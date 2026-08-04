from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Sample:
    sample_id: str
    pattern_id: str
    normal_path: str
    defect_type: str
    split: str = "train"
    seed: int = 2026
    reference_path: str | None = None
    reference_mask_path: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Sample":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{key: value[key] for key in allowed if key in value})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ROIPlan:
    xyxy: tuple[int, int, int, int]
    padding: tuple[int, int, int, int]
    source_size: tuple[int, int]
    model_size: int
    estimated_thickness: float
    scale: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

