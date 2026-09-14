#!/usr/bin/env python3
"""Single-image outpainting with PowerPaint v2.1.

The input is centered on a canvas twice its width and height, which adds 50%
of the original size on every side.  Inference can run at a smaller working
resolution; the final image is restored to exactly 2W x 2H and the original
pixels are composited back into the center.
"""

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image
from safetensors.torch import load_model
from transformers import CLIPTextModel

from diffusers import UniPCMultistepScheduler
from powerpaint.models.BrushNet_CA import BrushNetModel
from powerpaint.models.unet_2d_condition import UNet2DConditionModel
from powerpaint.pipelines.pipeline_PowerPaint_Brushnet_CA import (
    StableDiffusionPowerPaintBrushNetPipeline,
)
from powerpaint.utils.utils import TokenizerWrapper, add_tokens


DEFAULT_POWERPAINT_DIR = (
    "/mnt/DataPart/jianghongda/checkpoint/PowerPaint-v2-1/PowerPaint_Brushnet"
)
DEFAULT_BASE_MODEL_PATH = (
    "/mnt/DataPart/jianghongda/checkpoint/PowerPaint-v2-1/realisticVisionV60B1_v51VAE"
)
BASE_MODEL_FOLDER = "realisticVisionV60B1_v51VAE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Outpaint one image by 50% on all four sides with PowerPaint v2.1."
    )
    parser.add_argument("--input", required=True, help="Input image path.")
    parser.add_argument("--output", default="outpaint_result.png", help="Output image path.")
    parser.add_argument(
        "--powerpaint_model_dir",
        default=DEFAULT_POWERPAINT_DIR,
        help="Directory containing diffusion_pytorch_model.safetensors and pytorch_model.bin.",
    )
    parser.add_argument(
        "--base_model_path",
        default=DEFAULT_BASE_MODEL_PATH,
        help=(
            "Diffusers-format realisticVisionV60B1_v51VAE directory."
        ),
    )
    parser.add_argument("--prompt", default="", help="Optional scene description.")
    parser.add_argument("--negative_prompt", default="", help="Optional negative prompt.")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance_scale", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--model_input_short_side",
        type=int,
        default=512,
        help="Working short side before expansion. Use 0 to infer at the original resolution.",
    )
    parser.add_argument(
        "--seam_overlap",
        type=int,
        default=16,
        help="Working-resolution pixels masked inside the original image to improve seams.",
    )
    parser.add_argument(
        "--final_feather",
        type=int,
        default=24,
        help="Final-resolution pixels used to blend the original image back into the center.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument(
        "--cpu_offload",
        action="store_true",
        help="Reduce VRAM usage at the cost of speed.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Do not access Hugging Face while loading the base model.",
    )
    return parser.parse_args()


def find_powerpaint_weights(model_dir: str) -> Tuple[Path, Path]:
    root = Path(model_dir).expanduser()
    candidates = (root, root / "PowerPaint_Brushnet")
    for folder in candidates:
        brushnet = folder / "diffusion_pytorch_model.safetensors"
        text_encoder = folder / "pytorch_model.bin"
        if brushnet.is_file() and text_encoder.is_file():
            return brushnet, text_encoder
    raise FileNotFoundError(
        f"Cannot find both PowerPaint weights under {root}. Expected "
        "diffusion_pytorch_model.safetensors and pytorch_model.bin."
    )


def find_base_model(model_dir: str, explicit_path: Optional[str]) -> str:
    if explicit_path:
        return explicit_path

    root = Path(model_dir).expanduser()
    candidates = (
        root / BASE_MODEL_FOLDER,
        root.parent / BASE_MODEL_FOLDER,
    )
    for candidate in candidates:
        if (candidate / "model_index.json").is_file():
            return str(candidate)

    raise FileNotFoundError(
        "The two PowerPaint files are adapter/task weights, not a complete diffusion model. "
        f"Download the '{BASE_MODEL_FOLDER}' folder from JunhaoZhuang/PowerPaint-v2-1, "
        "then pass it with --base_model_path."
    )


def load_text_encoder_weights(module: torch.nn.Module, path: Path) -> None:
    try:
        state_dict = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:  # PyTorch versions before weights_only was added.
        state_dict = torch.load(path, map_location="cpu")
    incompatible = module.load_state_dict(state_dict, strict=False)
    if incompatible.unexpected_keys:
        print(f"Warning: ignored {len(incompatible.unexpected_keys)} unexpected text-encoder keys.")


def build_pipeline(args: argparse.Namespace):
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False.")

    dtype = getattr(torch, args.dtype)
    brushnet_path, text_encoder_path = find_powerpaint_weights(args.powerpaint_model_dir)
    base_model_path = find_base_model(args.powerpaint_model_dir, args.base_model_path)
    print(f"Base model: {base_model_path}")
    print(f"PowerPaint weights: {brushnet_path.parent}")

    source_unet = UNet2DConditionModel.from_pretrained(
        base_model_path,
        subfolder="unet",
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    text_encoder_brushnet = CLIPTextModel.from_pretrained(
        base_model_path,
        subfolder="text_encoder",
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    brushnet = BrushNetModel.from_unet(source_unet)
    del source_unet

    pipe = StableDiffusionPowerPaintBrushNetPipeline.from_pretrained(
        base_model_path,
        brushnet=brushnet,
        text_encoder_brushnet=text_encoder_brushnet,
        torch_dtype=dtype,
        low_cpu_mem_usage=False,
        safety_checker=None,
        local_files_only=args.local_files_only,
    )
    # The repository uses its custom UNet implementation for BrushNet residuals.
    pipe.unet = UNet2DConditionModel.from_pretrained(
        base_model_path,
        subfolder="unet",
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    pipe.tokenizer = TokenizerWrapper(
        from_pretrained=base_model_path,
        subfolder="tokenizer",
        revision=None,
        torch_type=dtype,
        local_files_only=args.local_files_only,
    )
    add_tokens(
        tokenizer=pipe.tokenizer,
        text_encoder=pipe.text_encoder_brushnet,
        placeholder_tokens=["P_ctxt", "P_shape", "P_obj"],
        initialize_tokens=["a", "a", "a"],
        num_vectors_per_token=10,
    )
    load_model(pipe.brushnet, str(brushnet_path))
    load_text_encoder_weights(pipe.text_encoder_brushnet, text_encoder_path)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    if args.cpu_offload:
        pipe.enable_model_cpu_offload(gpu_id=int(args.device.split(":")[-1]) if ":" in args.device else 0)
    else:
        pipe.to(args.device)
    return pipe


def aligned_working_size(width: int, height: int, short_side: int) -> Tuple[int, int]:
    if short_side > 0:
        scale = short_side / min(width, height)
        width = int(round(width * scale))
        height = int(round(height * scale))
    # Alignment is needed by the VAE. Keeping the dimensions even also centers exactly.
    width = max(8, int(round(width / 8)) * 8)
    height = max(8, int(round(height / 8)) * 8)
    return width, height


def prepare_canvas(image: Image.Image, short_side: int, seam_overlap: int):
    work_w, work_h = aligned_working_size(*image.size, short_side)
    resized = image.resize((work_w, work_h), Image.Resampling.LANCZOS)
    canvas_w, canvas_h = work_w * 2, work_h * 2
    left, top = work_w // 2, work_h // 2

    canvas = Image.new("RGB", (canvas_w, canvas_h), (127, 127, 127))
    canvas.paste(resized, (left, top))

    mask = np.full((canvas_h, canvas_w, 3), 255, dtype=np.uint8)
    overlap = max(0, min(seam_overlap, work_w // 4, work_h // 4))
    mask[top + overlap : top + work_h - overlap, left + overlap : left + work_w - overlap] = 0
    mask = Image.fromarray(mask, mode="RGB")
    return resized, canvas, mask


def composite_original(
    generated: Image.Image, original: Image.Image, feather: int
) -> Image.Image:
    original_w, original_h = original.size
    target = generated.resize((original_w * 2, original_h * 2), Image.Resampling.LANCZOS)
    left, top = original_w // 2, original_h // 2

    feather = max(0, min(feather, original_w // 4, original_h // 4))
    if feather == 0:
        target.paste(original, (left, top))
        return target

    y, x = np.ogrid[:original_h, :original_w]
    horizontal = np.minimum(x, original_w - 1 - x)
    vertical = np.minimum(y, original_h - 1 - y)
    distance = np.minimum(horizontal, vertical).astype(np.float32)
    alpha = np.clip(distance / feather, 0.0, 1.0)
    alpha = Image.fromarray(np.uint8(alpha * 255), mode="L")
    target.paste(original, (left, top), alpha)
    return target


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    original = Image.open(args.input).convert("RGB")
    pipe = build_pipeline(args)
    _, canvas, mask = prepare_canvas(original, args.model_input_short_side, args.seam_overlap)

    positive = f"{args.prompt.strip()} empty scene".strip()
    prompt_a = f"{positive} P_ctxt"
    prompt_b = prompt_a
    negative_a = f"{args.negative_prompt.strip()} P_obj".strip()
    negative_b = negative_a

    mask_array = np.asarray(mask, dtype=np.float32) / 255.0
    masked_array = np.asarray(canvas, dtype=np.float32) * (1.0 - mask_array)
    masked_image = Image.fromarray(masked_array.astype(np.uint8), mode="RGB")
    generator = torch.Generator(device=args.device).manual_seed(args.seed)

    result = pipe(
        promptA=prompt_a,
        promptB=prompt_b,
        promptU=positive,
        tradoff=1.0,
        tradoff_nag=1.0,
        image=masked_image,
        mask=mask,
        num_inference_steps=args.steps,
        generator=generator,
        brushnet_conditioning_scale=1.0,
        negative_promptA=negative_a,
        negative_promptB=negative_b,
        negative_promptU=args.negative_prompt,
        guidance_scale=args.guidance_scale,
        width=canvas.width,
        height=canvas.height,
    ).images[0]

    final = composite_original(result, original, args.final_feather)
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    final.save(output_path)
    print(f"Saved {final.width}x{final.height} result to: {output_path}")


if __name__ == "__main__":
    main()
