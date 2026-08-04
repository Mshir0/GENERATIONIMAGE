from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


class O2MAGBackend:
    """Loads O2MAG once and exposes a stable image-level API."""

    def __init__(self, config: dict):
        import torch
        from diffusers import DDIMScheduler

        from triag.diffuser_utils import LocalBlend, McaPipeline_Replace

        self.torch = torch
        self.LocalBlend = LocalBlend
        self.device = torch.device(config.get("device", "cuda:0"))
        self.steps = int(config.get("steps", 50))
        self.guidance = float(config.get("guidance_scale", 7.5))
        self.prompt_config = config.get("prompt", {})
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
        )
        self.pipe = McaPipeline_Replace.from_pretrained(
            str(Path(config["model_path"]).expanduser()), scheduler=scheduler
        ).to(self.device)
        self.pipe.vae.to(dtype=torch.float32)
        self.pipe.enable_vae_slicing()

    def _image_tensor(self, image: Image.Image):
        torch = self.torch
        value = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
        return torch.from_numpy(value).permute(2, 0, 1).unsqueeze(0).to(self.device)

    def _mask_tensor(self, mask: Image.Image):
        torch = self.torch
        value = np.asarray(mask.convert("L"), dtype=np.float32) > 127
        return torch.from_numpy(value.astype(np.float32)).to(self.device)

    def generate(
        self,
        normal: Image.Image,
        target_mask: Image.Image,
        reference: Image.Image,
        reference_mask: Image.Image,
        defect_type: str,
        seed: int,
    ) -> Image.Image:
        from triag.mca_p2p import McaControlReplace
        from triag.mca_utils import regiter_attention_editor_diffusers
        from triag.ptp_utils import get_equalizer

        torch = self.torch
        torch.manual_seed(seed)
        normal_t = self._image_tensor(normal)
        reference_t = self._image_tensor(reference)
        target_mask_t = self._mask_tensor(target_mask)
        reference_mask_t = self._mask_tensor(reference_mask)
        normal_prompt = self.prompt_config.get("normal", "a close-up photo of printed textile fabric")
        defect_label = defect_type.replace("_", " ")
        defect_prompt = self.prompt_config.get(
            "defect", "a close-up photo of printed textile fabric with {defect}"
        ).format(defect=defect_label)
        prompts = [defect_prompt, normal_prompt, defect_prompt]
        ref_latent, ref_history = self.pipe.invert(
            reference_t, "", self.steps, self.guidance, return_intermediates=True
        )
        src_latent, src_history = self.pipe.invert(
            normal_t, "", self.steps, self.guidance, return_intermediates=True
        )
        focus_word = defect_label.split()[-1]
        equalizer = get_equalizer(defect_prompt, (focus_word,), (100,), self.pipe.tokenizer).to(
            self.device
        )
        editor = McaControlReplace(
            prompts,
            self.pipe.tokenizer,
            [0, 1, 2, 3],
            self_replace_steps=0.1,
            cross_replace_steps=(0.4, 0.8),
            equalizer=equalizer,
            start_step=4,
            end_step=self.steps,
            start_layer=9,
            end_layer=16,
            total_steps=self.steps,
            mask_s=reference_mask_t,
            mask_t=target_mask_t,
        )
        regiter_attention_editor_diffusers(self.pipe, editor)
        output = self.pipe(
            prompts,
            latents=src_latent.expand(len(prompts), -1, -1, -1),
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            ref_intermediate_latents=[ref_history, src_history],
            lbl=self.LocalBlend(target_mask_t),
            neg_prompt=self.prompt_config.get("negative", ""),
            mask_r=reference_mask_t,
            mask_t=target_mask_t,
            output_type="pt",
        )
        tensor = output[-1].detach().cpu().clamp(0, 1)
        array = np.round(tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        return Image.fromarray(array)
