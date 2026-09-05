from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from workshop_jax.data.policy import parse_static_image_selector

_FIXED_IMAGE_SELECTORS = {
    "cifar10-dvs": ("cifar10-dvs", (20, 128, 128, 2), 10),
    "dvs128-gesture": ("dvs128-gesture", (20, 128, 128, 2), 11),
}
_STATIC_IMAGE_GEOMETRY = {
    "mnist": ((28, 28, 1), 10),
    "cifar-10": ((32, 32, 3), 10),
    "cifar-100": ((32, 32, 3), 100),
}
VGG11_SMALL_CONV_CHANNELS = (32, 64, 128, 128, 256, 256, 256, 256)
RESNET_SMALL_BLOCK_CHANNELS = (32, 32, 32, 32, 64, 64, 128)


@dataclass(frozen=True)
class NativeImageGeometry:
    dataset_selector: str
    base_dataset_id: str
    timesteps: int
    height: int
    width: int
    input_channels: int
    num_classes: int


def resolve_native_image_geometry(
    dataset_selector: Any,
    canonical_input_shape: tuple[int, ...],
) -> NativeImageGeometry:
    selector = str(dataset_selector).strip().lower()
    static = parse_static_image_selector(selector)
    if static is None:
        try:
            base, expected_shape, classes = _FIXED_IMAGE_SELECTORS[selector]
        except KeyError as error:
            raise ValueError(f"unsupported spatial dataset: {selector}") from error
    else:
        base, steps = static
        tail, classes = _STATIC_IMAGE_GEOMETRY[base]
        expected_shape = (steps, *tail)
    shape = tuple(int(value) for value in canonical_input_shape)
    if shape != expected_shape:
        raise ValueError(f"invalid canonical shape for {selector}: {shape}")
    steps, height, width, channels = expected_shape
    return NativeImageGeometry(
        dataset_selector=selector,
        base_dataset_id=base,
        timesteps=steps,
        height=height,
        width=width,
        input_channels=channels,
        num_classes=classes,
    )


@dataclass(frozen=True)
class Vgg11Topology:
    geometry: NativeImageGeometry
    conv_channels: tuple[int, ...]
    pool_after: tuple[int, ...]
    pool_kernels: tuple[int, ...]
    pool_strides: tuple[int, ...]
    classifier_width: int
    pooled_height: int
    pooled_width: int

    @property
    def hidden_site_widths(self) -> tuple[int, ...]:
        return (*self.conv_channels, self.classifier_width)

    @property
    def attributes(self) -> tuple[tuple[str, Any], ...]:
        return (
            ("conv_channels", self.conv_channels),
            ("pool_after", self.pool_after),
            ("pool_kernels", self.pool_kernels),
            ("pool_strides", self.pool_strides),
            ("classifier_width", self.classifier_width),
            ("pooled_grid", (self.pooled_height, self.pooled_width)),
        )


def resolve_vgg11_topology(
    dataset_selector: Any,
    canonical_input_shape: tuple[int, ...],
) -> Vgg11Topology:
    geometry = resolve_native_image_geometry(dataset_selector, canonical_input_shape)
    pool_after = (0, 1, 3, 5, 7)
    if geometry.base_dataset_id in {"cifar10-dvs", "dvs128-gesture"}:
        pool_kernels = (2, 2, 2, 2, 8)
    else:
        pool_kernels = (2, 2, 2, 2, 2)
    pool_strides = pool_kernels
    height, width = geometry.height, geometry.width
    for kernel, stride in zip(pool_kernels, pool_strides, strict=True):
        height = (height - kernel) // stride + 1
        width = (width - kernel) // stride + 1
    return Vgg11Topology(
        geometry=geometry,
        conv_channels=VGG11_SMALL_CONV_CHANNELS,
        pool_after=pool_after,
        pool_kernels=pool_kernels,
        pool_strides=pool_strides,
        classifier_width=VGG11_SMALL_CONV_CHANNELS[-1],
        pooled_height=height,
        pooled_width=width,
    )


@dataclass(frozen=True)
class ResNetBlockSpec:
    block_index: int
    input_channels: int
    output_channels: int
    adapter_site: int | None
    adapter_kernel: int | None
    adapter_padding: int | None
    main1_site: int
    main2_site: int
    pool_kernel: int | None
    pool_stride: int | None
    pool_padding: int | None

    @property
    def block_id(self) -> str:
        return f"block_{self.block_index + 1}"


@dataclass(frozen=True)
class ResNetTopology:
    geometry: NativeImageGeometry
    realization_id: str
    resolution_profile: str
    source_equivalence_status: str
    selection_basis: str
    block_channels: tuple[int, ...]
    pool_after: tuple[int, ...]
    pool_kernels: tuple[int, ...]
    pool_strides: tuple[int, ...]
    blocks: tuple[ResNetBlockSpec, ...]
    pooled_height: int
    pooled_width: int
    output_width: int

    @property
    def hidden_site_widths(self) -> tuple[int, ...]:
        site_count = 0
        for block in self.blocks:
            site_count = max(site_count, block.main1_site + 1, block.main2_site + 1)
            if block.adapter_site is not None:
                site_count = max(site_count, block.adapter_site + 1)
        widths: list[int | None] = [None] * site_count
        for block in self.blocks:
            if block.adapter_site is not None:
                widths[block.adapter_site] = block.output_channels
            widths[block.main1_site] = block.output_channels
            widths[block.main2_site] = block.output_channels
        return tuple(int(value) for value in widths if value is not None)

    @property
    def attributes(self) -> tuple[tuple[str, Any], ...]:
        return (
            ("realization_id", self.realization_id),
            ("resolution_profile", self.resolution_profile),
            ("block_channels", self.block_channels),
            ("pool_after", self.pool_after),
            ("pool_kernels", self.pool_kernels),
            ("pool_strides", self.pool_strides),
            ("pooled_grid", (self.pooled_height, self.pooled_width)),
        )


def _resnet_blocks(
    geometry: NativeImageGeometry,
    block_channels: tuple[int, ...],
    pool_after: tuple[int, ...],
    pool_kernels: tuple[int, ...],
    pool_strides: tuple[int, ...],
) -> tuple[tuple[ResNetBlockSpec, ...], int, int]:
    schedule = dict(
        zip(
            pool_after,
            zip(pool_kernels, pool_strides, strict=True),
            strict=True,
        )
    )
    blocks: list[ResNetBlockSpec] = []
    site_index = 0
    input_channels = geometry.input_channels
    height, width = geometry.height, geometry.width
    for block_index, output_channels in enumerate(block_channels):
        adapter_site = None
        adapter_kernel = None
        adapter_padding = None
        if input_channels != output_channels:
            adapter_site = site_index
            site_index += 1
            adapter_kernel = 1
            adapter_padding = 0
        main1_site = site_index
        main2_site = site_index + 1
        site_index += 2
        pool = schedule.get(block_index)
        if pool is None:
            pool_kernel = None
            pool_stride = None
            pool_padding = None
        else:
            pool_kernel, pool_stride = pool
            pool_padding = 0
            height = (height - pool_kernel) // pool_stride + 1
            width = (width - pool_kernel) // pool_stride + 1
        blocks.append(
            ResNetBlockSpec(
                block_index=block_index,
                input_channels=input_channels,
                output_channels=output_channels,
                adapter_site=adapter_site,
                adapter_kernel=adapter_kernel,
                adapter_padding=adapter_padding,
                main1_site=main1_site,
                main2_site=main2_site,
                pool_kernel=pool_kernel,
                pool_stride=pool_stride,
                pool_padding=pool_padding,
            )
        )
        input_channels = output_channels
    return tuple(blocks), height, width


def resolve_resnet_topology(
    dataset_selector: Any,
    canonical_input_shape: tuple[int, ...],
) -> ResNetTopology:
    geometry = resolve_native_image_geometry(dataset_selector, canonical_input_shape)
    pool_after = (0, 1, 3, 5, 6)
    if geometry.base_dataset_id in {"cifar10-dvs", "dvs128-gesture"}:
        resolution_profile = "event_128_geometry"
        pool_kernels = (2, 2, 2, 2, 8)
    else:
        resolution_profile = "small_geometry"
        pool_kernels = (2, 2, 2, 2, 2)
    pool_strides = pool_kernels
    blocks, height, width = _resnet_blocks(
        geometry,
        RESNET_SMALL_BLOCK_CHANNELS,
        pool_after,
        pool_kernels,
        pool_strides,
    )
    return ResNetTopology(
        geometry=geometry,
        realization_id="fixed_small_capacity_7block_sew_resnet_shared_pooling",
        resolution_profile=resolution_profile,
        source_equivalence_status=(
            "source_concept_adapted_fixed_small_capacity_shared_depth_pooling"
        ),
        selection_basis=(
            "seven_sew_add_blocks_with_shared_pool_boundaries_and_"
            "fixed_small_capacity_resolution_specific_pool_profile"
        ),
        block_channels=RESNET_SMALL_BLOCK_CHANNELS,
        pool_after=pool_after,
        pool_kernels=pool_kernels,
        pool_strides=pool_strides,
        blocks=blocks,
        pooled_height=height,
        pooled_width=width,
        output_width=RESNET_SMALL_BLOCK_CHANNELS[-1],
    )


__all__ = [
    "NativeImageGeometry",
    "ResNetBlockSpec",
    "ResNetTopology",
    "Vgg11Topology",
    "resolve_native_image_geometry",
    "resolve_resnet_topology",
    "resolve_vgg11_topology",
]
