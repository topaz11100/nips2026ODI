from __future__ import annotations

from dataclasses import dataclass
import gzip
import math
import pickle
import struct
from pathlib import Path
from typing import Any, BinaryIO, ClassVar

import numpy as np

from workshop_jax.data.preparation.common import (
    canonical_source_root,
    create_dataset_writer,
    initialize_dataset_tree,
    write_dataset_metadata,
)
from workshop_jax.data.policy import DatasetPolicy, parse_static_image_selector


_MIN_STREAMED_CIFAR_ARRAY_BYTES = 3072


@dataclass(frozen=True)
class _IdxSpec:
    path: Path
    header: bytes
    header_size: int
    count: int
    item_shape: tuple[int, ...]

    @property
    def item_size(self) -> int:
        return math.prod(self.item_shape) if self.item_shape else 1

    @property
    def payload_size(self) -> int:
        return self.count * self.item_size


@dataclass(frozen=True)
class _PickleByteRange:
    offset: int
    nbytes: int


@dataclass(frozen=True)
class _CifarArrayReference:
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    fortran_order: bool


@dataclass(frozen=True)
class _CifarNdarrayPlaceholder:
    pass


class _CifarNdarrayToken:
    pass


def _locate_mnist_source(raw_root: Path) -> Path:
    required = (
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    )
    dataset_root = canonical_source_root(raw_root, "mnist")
    candidates = (
        dataset_root,
        dataset_root / "raw",
        dataset_root / "MNIST" / "raw",
        dataset_root / "MNIST",
    )
    seen: set[Path] = set()
    for source in candidates:
        source = source.resolve()
        if source in seen or not source.is_dir():
            continue
        seen.add(source)
        if all(_resolve_mnist_idx_file(source, name) is not None for name in required):
            return source
    accepted = [f"{name} or {name}.gz" for name in required]
    raise FileNotFoundError(
        "MNIST source is missing official IDX files; accepted names are "
        f"{accepted} inside <raw_data>/mnist/ (optionally raw/ or MNIST/raw/)"
    )


def _locate_cifar_source(raw_root: Path, source_dataset_id: str) -> Path:
    dataset_root = canonical_source_root(raw_root, source_dataset_id)
    wrapper = (
        "cifar-10-batches-py" if source_dataset_id == "cifar-10" else "cifar-100-python"
    )
    candidates = (dataset_root / wrapper, dataset_root)
    required = (
        tuple(f"data_batch_{index}" for index in range(1, 6)) + ("test_batch",)
        if source_dataset_id == "cifar-10"
        else ("train", "test")
    )
    for source in candidates:
        if source.is_dir() and all((source / name).is_file() for name in required):
            return source.resolve(strict=False)
    raise FileNotFoundError(
        f"{source_dataset_id} source is missing official Python batches under "
        f"<raw_data>/{source_dataset_id}/; required={list(required)}"
    )


def prepare_mnist_dataset(
    raw_root: Path, policy: DatasetPolicy, dataset_root: Path
) -> None:
    source = _locate_mnist_source(raw_root)
    split_specs = _mnist_split_specs(source)
    counts = {split: image_spec.count for split, (image_spec, _) in split_specs.items()}

    initialize_dataset_tree(dataset_root)
    split_class_counts: dict[str, tuple[int, ...]] = {}

    for split, (image_spec, label_spec) in split_specs.items():
        writer = create_dataset_writer(
            dataset_root,
            policy,
            split=split,
            sample_count=image_spec.count,
        )
        with (
            _open_idx_payload(image_spec) as image_stream,
            _open_idx_payload(label_spec) as label_stream,
        ):
            for index in range(image_spec.count):
                image = _read_idx_chunk(
                    image_stream,
                    image_spec,
                    item_count=1,
                )[0].astype(np.float64) / np.float64(255.0)
                label = _read_idx_chunk(
                    label_stream,
                    label_spec,
                    item_count=1,
                )[0]
                transformed = _transform_mnist_sample(image, policy)
                writer.write_sample(
                    index,
                    transformed,
                    np.float64(label),
                    f"mnist:{split}:{index:08d}",
                )
                del image, transformed
        writer.commit()
        split_class_counts[split] = writer.class_counts

    write_dataset_metadata(
        dataset_root,
        policy,
        source={
            "adapter_id": policy.adapter_id,
            "source_root": str(source),
            "format": "official_idx_plain_or_gzip",
        },
        preprocessing=_image_preprocessing(policy),
        split_counts=counts,
        split_class_counts=split_class_counts,
    )


def prepare_cifar_dataset(
    raw_root: Path, policy: DatasetPolicy, dataset_root: Path
) -> None:
    source = _locate_cifar_source(raw_root, policy.source_dataset_id)
    split_files = _cifar_split_files(source, policy.source_dataset_id)
    split_batches = {
        split: tuple(
            (path, *_load_cifar_batch(path, policy.source_dataset_id)) for path in paths
        )
        for split, paths in split_files.items()
    }
    counts = {
        split: sum(data.shape[0] for _, data, _ in batches)
        for split, batches in split_batches.items()
    }
    initialize_dataset_tree(dataset_root)
    split_class_counts: dict[str, tuple[int, ...]] = {}
    for split, batches in split_batches.items():
        writer = create_dataset_writer(
            dataset_root,
            policy,
            split=split,
            sample_count=counts[split],
        )
        offset = 0
        for path, data, labels in batches:
            for local_index in range(data.shape[0]):
                image = data[local_index].reshape(3, 32, 32).transpose(1, 2, 0).astype(
                    np.float64
                ) / np.float64(255.0)
                transformed = _transform_cifar_sample(image, policy)
                writer.write_sample(
                    offset,
                    transformed,
                    np.float64(labels[local_index]),
                    f"{policy.source_dataset_id}:{split}:{offset:08d}",
                )
                del image, transformed
                offset += 1
            del data, labels
        writer.commit()
        split_class_counts[split] = writer.class_counts

    write_dataset_metadata(
        dataset_root,
        policy,
        source={
            "adapter_id": policy.adapter_id,
            "source_root": str(source),
            "format": "official_python_batches",
        },
        preprocessing=_image_preprocessing(policy),
        split_counts=counts,
        split_class_counts=split_class_counts,
    )


def _mnist_split_specs(source: Path) -> dict[str, tuple[_IdxSpec, _IdxSpec]]:
    split_files = {
        "train": (
            _mnist_idx_file(source, "train-images-idx3-ubyte"),
            _mnist_idx_file(source, "train-labels-idx1-ubyte"),
        ),
        "test": (
            _mnist_idx_file(source, "t10k-images-idx3-ubyte"),
            _mnist_idx_file(source, "t10k-labels-idx1-ubyte"),
        ),
    }
    split_specs: dict[str, tuple[_IdxSpec, _IdxSpec]] = {}
    for split, (image_path, label_path) in split_files.items():
        image_spec = _inspect_idx_images(image_path)
        label_spec = _inspect_idx_labels(label_path)
        split_specs[split] = (image_spec, label_spec)
    return split_specs


def _inspect_idx_images(path: Path) -> _IdxSpec:
    header = _read_idx_header(path, header_size=16)
    magic, count, rows, cols = struct.unpack(">IIII", header)
    spec = _IdxSpec(
        path=path,
        header=header,
        header_size=16,
        count=int(count),
        item_shape=(int(rows), int(cols)),
    )
    return spec


def _inspect_idx_labels(path: Path) -> _IdxSpec:
    header = _read_idx_header(path, header_size=8)
    magic, count = struct.unpack(">II", header)
    spec = _IdxSpec(
        path=path,
        header=header,
        header_size=8,
        count=int(count),
        item_shape=(),
    )
    return spec


def _read_idx_header(path: Path, *, header_size: int) -> bytes:
    with _open_idx_source(path) as handle:
        header = _read_exact(
            handle,
            header_size,
            failure=f"truncated MNIST IDX header: {path}",
        )
    return header


class _IdxPayloadStream:
    def __init__(self, spec: _IdxSpec) -> None:
        self._spec = spec
        self._context: Any | None = None
        self._handle: BinaryIO | None = None

    def __enter__(self) -> BinaryIO:
        self._context = _open_idx_source(self._spec.path)
        handle = self._context.__enter__()
        _read_exact(
            handle,
            self._spec.header_size,
            failure=f"truncated MNIST IDX header: {self._spec.path}",
        )
        self._handle = handle
        return handle

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> Any:
        if self._context is None:
            return None
        return self._context.__exit__(exc_type, exc, traceback)


def _open_idx_payload(spec: _IdxSpec) -> _IdxPayloadStream:
    return _IdxPayloadStream(spec)


def _open_idx_source(path: Path) -> Any:
    if path.suffix == ".gz":
        return gzip.open(path, "rb")
    return path.open("rb")


def _read_exact(handle: BinaryIO, size: int, *, failure: str) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = handle.read(size - len(chunks))
        if not chunk:
            break
        chunks.extend(chunk)
    if len(chunks) != size:
        raise ValueError(failure)
    return bytes(chunks)


def _read_idx_chunk(
    handle: BinaryIO,
    spec: _IdxSpec,
    *,
    item_count: int,
) -> np.ndarray:
    payload = _read_exact(
        handle,
        item_count * spec.item_size,
        failure=f"truncated MNIST IDX payload: {spec.path}",
    )
    return np.frombuffer(payload, dtype=np.uint8).reshape(
        item_count,
        *spec.item_shape,
    )


def _resolve_mnist_idx_file(source: Path, basename: str) -> Path | None:
    for candidate in (source / basename, source / f"{basename}.gz"):
        if candidate.is_file():
            return candidate
    return None


def _mnist_idx_file(source: Path, basename: str) -> Path:
    path = _resolve_mnist_idx_file(source, basename)
    if path is None:
        raise FileNotFoundError(source / basename)
    return path


def _transform_mnist_sample(image: np.ndarray, policy: DatasetPolicy) -> np.ndarray:
    if policy.selector != "s-mnist":
        raise ValueError(f"unsupported MNIST paper dataset: {policy.selector}")
    return image.reshape(784, 1, 1, 1)


def _transform_cifar_sample(image: np.ndarray, policy: DatasetPolicy) -> np.ndarray:
    num_steps = _static_image_steps(policy)
    return np.repeat(image[None, :, :, :], num_steps, axis=0)


def _static_image_steps(policy: DatasetPolicy) -> int:
    static_selection = parse_static_image_selector(policy.selector)
    if static_selection is None:
        raise ValueError(f"unsupported static paper dataset: {policy.selector}")
    _, selector_steps = static_selection
    return selector_steps


def _cifar_split_files(source: Path, dataset_id: str) -> dict[str, tuple[Path, ...]]:
    if dataset_id == "cifar-10":
        return {
            "train": tuple(source / f"data_batch_{index}" for index in range(1, 6)),
            "test": (source / "test_batch",),
        }
    return {"train": (source / "train",), "test": (source / "test",)}


def _reconstruct_cifar_ndarray(
    array_type: type[Any],
    shape: tuple[int, ...],
    typecode: bytes,
) -> _CifarNdarrayPlaceholder:
    return _CifarNdarrayPlaceholder()


class _CifarProtocol2Unpickler(pickle._Unpickler):
    dispatch: ClassVar[Any] = pickle._Unpickler.dispatch.copy()

    def __init__(self, handle: BinaryIO) -> None:
        super().__init__(handle, fix_imports=True, encoding="bytes")
        self._source_handle = handle

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) in {
            ("numpy.core.multiarray", "_reconstruct"),
            ("numpy._core.multiarray", "_reconstruct"),
        }:
            return _reconstruct_cifar_ndarray
        if (module, name) == ("numpy", "ndarray"):
            return _CifarNdarrayToken
        if (module, name) == ("numpy", "dtype"):
            return np.dtype
        raise pickle.UnpicklingError(
            f"unsupported global in official CIFAR batch: {module}.{name}"
        )

    def load_binstring(self) -> None:
        runtime: Any = self
        length = struct.unpack("<i", runtime.read(4))[0]
        if length >= _MIN_STREAMED_CIFAR_ARRAY_BYTES:
            offset = self._source_handle.tell()
            self._source_handle.seek(length, 1)
            runtime.append(_PickleByteRange(offset=offset, nbytes=length))
            return
        runtime.append(runtime._decode_string(runtime.read(length)))

    dispatch[pickle.BINSTRING[0]] = load_binstring

    def load_build(self) -> None:
        runtime: Any = self
        state = runtime.stack[-1]
        instance = runtime.stack[-2]
        if isinstance(instance, _CifarNdarrayPlaceholder):
            runtime.stack.pop()
            byte_range = state[4]
            reference = _CifarArrayReference(
                offset=byte_range.offset,
                nbytes=byte_range.nbytes,
                shape=tuple(state[1]),
                dtype=state[2],
                fortran_order=state[3],
            )
            runtime.stack[-1] = reference
            for memo_key, memo_value in tuple(runtime.memo.items()):
                if memo_value is instance:
                    runtime.memo[memo_key] = reference
            return
        base_unpickler: Any = pickle._Unpickler
        base_unpickler.load_build(self)

    dispatch[pickle.BUILD[0]] = load_build


def _load_pickle(path: Path) -> dict[Any, Any]:
    with path.open("rb") as handle:
        return _CifarProtocol2Unpickler(handle).load()


def _load_cifar_batch(path: Path, dataset_id: str) -> tuple[np.ndarray, np.ndarray]:
    payload = _load_pickle(path)
    raw_data = _required_cifar_value(payload, "data", path=path)
    data: np.ndarray
    if isinstance(raw_data, _CifarArrayReference):
        data = np.memmap(
            path,
            mode="r",
            dtype=raw_data.dtype,
            offset=raw_data.offset,
            shape=raw_data.shape,
            order="F" if raw_data.fortran_order else "C",
        )
    elif isinstance(raw_data, np.ndarray):
        data = raw_data
    else:
        raise ValueError(f"CIFAR data is not an ndarray: {path}")

    label_name = "labels" if dataset_id == "cifar-10" else "fine_labels"
    raw_labels = _required_cifar_value(payload, label_name, path=path)
    labels = np.asarray(raw_labels)
    return data, labels.astype(np.int64, copy=False)


def _required_cifar_value(
    payload: dict[Any, Any],
    name: str,
    *,
    path: Path,
) -> Any:
    keys = (name.encode("ascii"), name)
    present = [key for key in keys if key in payload]
    if not present:
        raise ValueError(f"CIFAR batch has no {name} field: {path}")
    return payload[present[0]]


def _image_preprocessing(policy: DatasetPolicy) -> dict[str, Any]:
    return {
        "policy_id": policy.policy_id,
        "normalization": "uint8_divide_255_float64",
        "floating_compute_dtype": "float64",
        "storage_dtype": "float64",
        "temporal_rule": (
            "row_major_sequence"
            if policy.selector == "s-mnist"
            else f"static_repeat_{policy.num_steps}"
        ),
    }
