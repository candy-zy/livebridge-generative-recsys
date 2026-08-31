"""Lightweight validation for the public KuaiLive-M3 file layout."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


REQUIRED_FILES = (
    "user_id_set.csv",
    "author_profile.csv",
    "photo_interaction.csv",
    "live_interaction.csv",
)

OPTIONAL_FILES = (
    "photo_play.parquet",
    "photo_meta.parquet",
    "photo_tag.csv",
    "photo_emb_128.parquet",
    "live_show.parquet",
    "live_room_meta.parquet",
    "live_emb_64.parquet",
    "live_comment.csv",
    "live_like.csv",
    "live_share.csv",
    "live_questionnaire.csv",
)

OPTIONAL_DIRECTORIES = ("live_emb_128_ts",)


@dataclass(frozen=True)
class DataValidationReport:
    data_dir: str
    valid_minimal_cdr: bool
    missing_required: tuple[str, ...]
    present_required: tuple[str, ...]
    present_optional: tuple[str, ...]
    missing_optional: tuple[str, ...]
    wrong_type: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def validate_data_dir(data_dir: str | Path) -> DataValidationReport:
    root = Path(data_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"KuaiLive-M3 data directory does not exist: {root}")

    missing_required: list[str] = []
    present_required: list[str] = []
    present_optional: list[str] = []
    missing_optional: list[str] = []
    wrong_type: list[str] = []

    for name in REQUIRED_FILES:
        path = root / name
        if path.is_file():
            present_required.append(name)
        elif path.exists():
            wrong_type.append(name)
            missing_required.append(name)
        else:
            missing_required.append(name)

    for name in OPTIONAL_FILES:
        path = root / name
        if path.is_file():
            present_optional.append(name)
        elif path.exists():
            wrong_type.append(name)
            missing_optional.append(name)
        else:
            missing_optional.append(name)

    for name in OPTIONAL_DIRECTORIES:
        path = root / name
        if path.is_dir():
            present_optional.append(f"{name}/")
        elif path.exists():
            wrong_type.append(f"{name}/")
            missing_optional.append(f"{name}/")
        else:
            missing_optional.append(f"{name}/")

    return DataValidationReport(
        data_dir=str(root),
        valid_minimal_cdr=not missing_required and not wrong_type,
        missing_required=tuple(missing_required),
        present_required=tuple(present_required),
        present_optional=tuple(present_optional),
        missing_optional=tuple(missing_optional),
        wrong_type=tuple(wrong_type),
    )
