from __future__ import annotations

from dataclasses import dataclass
import re


_STATIC_SELECTOR = re.compile(
    r"^(?P<source>mnist|cifar-10|cifar-100)_(?P<steps>[1-9][0-9]*)$"
)


@dataclass(frozen=True)
class DatasetPolicy:
    selector: str
    policy_id: str
    sample_shape: tuple[int, ...]
    num_classes: int
    adapter_id: str
    source_dataset_id: str

    @property
    def dataset_id(self) -> str:
        return self.selector

    @property
    def num_steps(self) -> int:
        return int(self.sample_shape[0])


@dataclass(frozen=True)
class FixedEventProfile:
    preprocessing_version: str
    duration_us: int
    time_steps: int
    input_channels: int
    raw_channels: int | None = None
    channel_pool_size: int = 1

    @property
    def duration_s(self) -> float:
        return self.duration_us / 1_000_000.0

    @property
    def bin_width_us(self) -> int:
        return self.duration_us // self.time_steps

    @property
    def time_bin_s(self) -> float:
        return self.bin_width_us / 1_000_000.0


SHD_PREPROCESSING_PROFILE = FixedEventProfile(
    preprocessing_version="shd_spike_count_4ms_pool5_fixed_1s_v1",
    duration_us=1_000_000,
    time_steps=250,
    input_channels=140,
    raw_channels=700,
    channel_pool_size=5,
)
CIFAR10_DVS_PREPROCESSING_PROFILE = FixedEventProfile(
    preprocessing_version="cifar10dvs_count20_fixed_first_1p3s_90_10_v1",
    duration_us=1_300_000,
    time_steps=20,
    input_channels=2,
)
DVS128_GESTURE_PREPROCESSING_PROFILE = FixedEventProfile(
    preprocessing_version="dvs128gesture_count20_fixed_first_6s_official_csv_v1",
    duration_us=6_000_000,
    time_steps=20,
    input_channels=2,
)


def parse_static_image_selector(selector: str) -> tuple[str, int] | None:
    match = _STATIC_SELECTOR.fullmatch(str(selector))
    if match is None:
        return None
    return match.group("source"), int(match.group("steps"))


def _policy(
    selector: str,
    policy_id: str,
    sample_shape: tuple[int, ...],
    classes: int,
    adapter: str,
    source: str,
) -> DatasetPolicy:
    return DatasetPolicy(
        selector=selector,
        policy_id=policy_id,
        sample_shape=sample_shape,
        num_classes=classes,
        adapter_id=adapter,
        source_dataset_id=source,
    )


_POLICIES = {
    "s-mnist": _policy(
        "s-mnist", "s_mnist_row_major", (784, 1, 1, 1), 10, "mnist", "mnist"
    ),
    "shd": _policy(
        "shd",
        SHD_PREPROCESSING_PROFILE.preprocessing_version,
        (250, 1, 1, 140),
        20,
        "shd",
        "shd",
    ),
    "cifar-10_16": _policy(
        "cifar-10_16", "cifar10_repeat16", (16, 32, 32, 3), 10, "cifar", "cifar-10"
    ),
    "dvs128-gesture": _policy(
        "dvs128-gesture",
        DVS128_GESTURE_PREPROCESSING_PROFILE.preprocessing_version,
        (20, 128, 128, 2),
        11,
        "dvs128-gesture",
        "dvs128-gesture",
    ),
    "cifar-100_16": _policy(
        "cifar-100_16",
        "cifar100_repeat16",
        (16, 32, 32, 3),
        100,
        "cifar",
        "cifar-100",
    ),
    "cifar10-dvs": _policy(
        "cifar10-dvs",
        CIFAR10_DVS_PREPROCESSING_PROFILE.preprocessing_version,
        (20, 128, 128, 2),
        10,
        "cifar10-dvs",
        "cifar10-dvs",
    ),
}


def dataset_policy(selector: str) -> DatasetPolicy:
    try:
        return _POLICIES[str(selector)]
    except KeyError as error:
        raise ValueError(f"unsupported paper dataset: {selector}") from error


__all__ = [
    "CIFAR10_DVS_PREPROCESSING_PROFILE",
    "DVS128_GESTURE_PREPROCESSING_PROFILE",
    "DatasetPolicy",
    "FixedEventProfile",
    "SHD_PREPROCESSING_PROFILE",
    "dataset_policy",
    "parse_static_image_selector",
]
