from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .schemas import Sample


def read_manifest(path: str | Path) -> list[Sample]:
    path = Path(path)
    samples: list[Sample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                samples.append(Sample.from_dict(json.loads(line)))
            except Exception as exc:
                raise ValueError(f"Invalid manifest line {line_number}: {exc}") from exc
    ids = [sample.sample_id for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError("sample_id values must be unique")
    return samples


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

