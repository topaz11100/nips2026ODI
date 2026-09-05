from __future__ import annotations

from dataclasses import dataclass
import unicodedata
from typing import Any, Iterable

import jax


def canonical_semantic_path(path: str | int | Iterable[str | int]) -> tuple[str, ...]:
    if isinstance(path, (str, int)):
        raw_components: tuple[str | int, ...] = (path,)
    else:
        raw_components = tuple(path)
    return tuple(unicodedata.normalize("NFKC", str(item)) for item in raw_components)


def _fold_integer(key: Any, value: int) -> Any:
    return jax.random.fold_in(key, int(value))


def _fold_path(key: Any, path: tuple[str, ...]) -> Any:
    key = _fold_integer(key, len(path))
    for component in path:
        encoded = component.encode("ascii")
        key = _fold_integer(key, len(encoded))
        for byte in encoded:
            key = _fold_integer(key, byte)
    return key


@dataclass(frozen=True)
class InitializationKeyspace:
    root_key: Any
    topology_id: str

    def __post_init__(self) -> None:
        topology_id = canonical_semantic_path(self.topology_id)
        object.__setattr__(self, "topology_id", topology_id[0])

    def topology_key(self, parameter_path: str | int | Iterable[str | int]) -> Any:
        paths = (
            canonical_semantic_path(self.topology_id),
            canonical_semantic_path(parameter_path),
        )
        key = _fold_integer(self.root_key, 1)
        key = _fold_integer(key, len(paths))
        for path in paths:
            key = _fold_path(key, path)
        return _fold_integer(key, 0)


__all__ = ["InitializationKeyspace", "canonical_semantic_path"]
