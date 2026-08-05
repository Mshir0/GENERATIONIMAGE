from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .fusion import strict_composite, wavelet_fuse
from .masks import build_placement_mask, generate_process_mask
from .procedural import render_procedural
from .roi import extract_roi, paste_roi, plan_roi
from .schemas import Sample


class FabricGenerationPipeline:
    def __init__(self, config: dict, output_dir: str | Path):
        self.config = config
        self.output_dir = Path(output_dir)
        for name in ("images", "masks", "originals", "metadata"):
            (self.output_dir / name).mkdir(parents=True, exist_ok=True)
        self._backend = None

    def _get_backend(self):
        if self._backend is None:
            from .backend import O2MAGBackend

            self._backend = O2MAGBackend(self.config)
        return self._backend

    def generate(self, sample: Sample, allowed_routes: set[str] | None = None) -> dict:
        rng = np.random.default_rng(sample.seed)
        normal = Image.open(sample.normal_path).convert("RGB")
        defaults = self.config.get("mask_defaults", {})
        type_defaults = self.config.get("mask_by_type", {}).get(sample.defect_type, {})
        mask_params = {**defaults, **type_defaults, **sample.parameters.get("mask", {})}
        placement_config = mask_params.get("placement")
        placement_mask = None
        if placement_config:
            explicit_mask_path = placement_config.get("mask_path")
            if explicit_mask_path:
                placement_mask = np.asarray(
                    Image.open(explicit_mask_path).convert("L").resize(normal.size, Image.Resampling.NEAREST)
                )
            else:
                placement_mask = build_placement_mask(np.asarray(normal), placement_config)
        generated_mask = generate_process_mask(
            (normal.height, normal.width),
            sample.defect_type,
            rng,
            mask_params,
            placement_mask=placement_mask,
        )
        mask = Image.fromarray(generated_mask.mask, mode="L")
        route = self.config.get("routes", {}).get(sample.defect_type, "procedural")
        if allowed_routes and route not in allowed_routes:
            raise RuntimeError(f"Route {route} is disabled by --routes")

        roi_config = self.config.get("roi", {})
        use_roi = bool(roi_config.get("enabled", True)) and route == "o2mag"
        plan = None
        full_resize = False
        normal_input, mask_input = normal, mask
        if use_roi:
            plan = plan_roi(
                generated_mask.mask,
                model_size=int(roi_config.get("model_size", 512)),
                min_defect_px=int(roi_config.get("min_defect_px", 24)),
                context_ratio=float(roi_config.get("context_ratio", 3.0)),
                min_crop_size=int(roi_config.get("min_crop_size", 64)),
            )
            normal_input = extract_roi(normal, plan)
            mask_input = extract_roi(mask, plan, is_mask=True)
        elif route == "o2mag":
            model_size = int(roi_config.get("model_size", 512))
            normal_input = normal.resize((model_size, model_size), Image.Resampling.LANCZOS)
            mask_input = mask.resize((model_size, model_size), Image.Resampling.NEAREST)
            full_resize = True

        if route == "procedural":
            generated = render_procedural(
                normal_input, mask_input, sample.defect_type, rng, sample.parameters.get("render", {})
            )
        elif route == "o2mag":
            if not sample.reference_path or not sample.reference_mask_path:
                raise ValueError(f"{sample.sample_id}: O2MAG route requires reference paths")
            reference = Image.open(sample.reference_path).convert("RGB")
            reference_mask = Image.open(sample.reference_mask_path).convert("L")
            reference = reference.resize(normal_input.size, Image.Resampling.LANCZOS)
            reference_mask = reference_mask.resize(mask_input.size, Image.Resampling.NEAREST)
            generated = self._get_backend().generate(
                normal_input, mask_input, reference, reference_mask, sample.defect_type, sample.seed
            )
        else:
            raise ValueError(f"Unknown route: {route}")

        if full_resize:
            generated = generated.resize(normal.size, Image.Resampling.LANCZOS)
            normal_input, mask_input = normal, mask

        fusion_config = self.config.get("fusion", {})
        if fusion_config.get("enabled", True):
            lambdas = fusion_config.get("lambda_high", {})
            generated = wavelet_fuse(
                normal_input,
                generated,
                mask_input,
                float(lambdas.get(sample.defect_type, 0.75)),
                fusion_config.get("wavelet", "db2"),
                int(fusion_config.get("levels", 2)),
                int(fusion_config.get("feather_radius", 5)),
            )
        else:
            generated = strict_composite(normal_input, generated, mask_input)

        if plan:
            generated = paste_roi(normal, generated, plan)
        image_path = self.output_dir / "images" / f"{sample.sample_id}.png"
        mask_path = self.output_dir / "masks" / f"{sample.sample_id}.png"
        original_path = self.output_dir / "originals" / f"{sample.sample_id}.png"
        metadata_path = self.output_dir / "metadata" / f"{sample.sample_id}.json"
        generated.save(image_path)
        mask.save(mask_path)
        normal.save(original_path)
        record = {
            **sample.to_dict(),
            "generated_path": str(image_path.resolve()),
            "mask_path": str(mask_path.resolve()),
            "original_path": str(original_path.resolve()),
            "metadata_path": str(metadata_path.resolve()),
            "route": route,
            "mask_parameters": generated_mask.parameters,
            "roi": plan.to_dict() if plan else None,
            "fusion": fusion_config,
        }
        metadata_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        return record
