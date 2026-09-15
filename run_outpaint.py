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
from PIL import Image, ImageFilter
from safetensors.torch import load_model
from transformers import CLIPTextModel

from diffusers import AutoencoderKL, UniPCMultistepScheduler
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
        "--outpaint_mode",
        choices=("two_stage", "one_stage"),
        default="two_stage",
        help="two_stage expands left/right first and then top/bottom for better large outpainting.",
    )
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
        "--save_intermediate",
        action="store_true",
        help="Save the horizontal first-pass result next to the final output.",
    )
    parser.add_argument(
        "--final_feather",
        type=int,
        default=24,
        help="Final-resolution seam width when restoring the exact original pixels.",
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

    unet = UNet2DConditionModel.from_pretrained(
        base_model_path,
        subfolder="unet",
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    vae = AutoencoderKL.from_pretrained(
        base_model_path,
        subfolder="vae",
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        base_model_path,
        subfolder="text_encoder",
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    text_encoder_brushnet = CLIPTextModel.from_pretrained(
        base_model_path,
        subfolder="text_encoder",
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    brushnet = BrushNetModel.from_unet(unet)
    scheduler = UniPCMultistepScheduler.from_pretrained(
        base_model_path,
        subfolder="scheduler",
        local_files_only=args.local_files_only,
    )
    tokenizer = TokenizerWrapper(
        from_pretrained=base_model_path,
        subfolder="tokenizer",
        revision=None,
        torch_type=dtype,
        local_files_only=args.local_files_only,
    )

    # Construct the custom pipeline explicitly. Newer Diffusers versions infer
    # a stock diffusers.UNet2DConditionModel from model_index.json when using
    # Pipeline.from_pretrained(), which is incompatible with PowerPaint's custom
    # UNet and can also replace/mis-register the supplied BrushNet component.
    pipe = StableDiffusionPowerPaintBrushNetPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        brushnet=brushnet,
        text_encoder_brushnet=text_encoder_brushnet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        image_encoder=None,
        requires_safety_checker=False,
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


def prepare_directional_canvas(
    image: Image.Image, direction: str, seam_overlap: int
) -> Tuple[Image.Image, Image.Image]:
    width, height = image.size
    overlap = max(0, min(seam_overlap, width // 4, height // 4))

    if direction == "horizontal":
        canvas = Image.new("RGB", (width * 2, height), (127, 127, 127))
        left = width // 2
        canvas.paste(image, (left, 0))
        mask_array = np.full((height, width * 2, 3), 255, dtype=np.uint8)
        mask_array[:, left + overlap : left + width - overlap] = 0
    elif direction == "vertical":
        canvas = Image.new("RGB", (width, height * 2), (127, 127, 127))
        top = height // 2
        canvas.paste(image, (0, top))
        mask_array = np.full((height * 2, width, 3), 255, dtype=np.uint8)
        mask_array[top + overlap : top + height - overlap, :] = 0
    else:
        raise ValueError(f"Unsupported outpainting direction: {direction}")

    return canvas, Image.fromarray(mask_array, mode="RGB")


def run_powerpaint(
    pipe,
    canvas: Image.Image,
    mask: Image.Image,
    positive: str,
    negative_prompt: str,
    args: argparse.Namespace,
    seed: int,
) -> Image.Image:
    prompt_a = f"{positive} P_ctxt"
    negative_a = f"{negative_prompt.strip()} P_obj".strip()

    mask_array = np.asarray(mask, dtype=np.float32) / 255.0
    masked_array = np.asarray(canvas, dtype=np.float32) * (1.0 - mask_array)
    masked_image = Image.fromarray(masked_array.astype(np.uint8), mode="RGB")
    generator = torch.Generator(device=args.device).manual_seed(seed)

    return pipe(
        promptA=prompt_a,
        promptB=prompt_a,
        promptU=positive,
        tradoff=1.0,
        tradoff_nag=1.0,
        image=masked_image,
        mask=mask,
        num_inference_steps=args.steps,
        generator=generator,
        brushnet_conditioning_scale=1.0,
        negative_promptA=negative_a,
        negative_promptB=negative_a,
        negative_promptU=negative_prompt,
        guidance_scale=args.guidance_scale,
        width=canvas.width,
        height=canvas.height,
    ).images[0]


def preserve_known_region(
    generated: Image.Image,
    known_canvas: Image.Image,
    generation_mask: Image.Image,
    feather_radius: int,
) -> Image.Image:
    """Keep unmasked pixels exact and feather only the mask boundary."""
    blend_mask = generation_mask.convert("L")
    if feather_radius > 0:
        blend_mask = blend_mask.filter(
            ImageFilter.GaussianBlur(radius=max(1, feather_radius // 2))
        )
    return Image.composite(generated.convert("RGB"), known_canvas.convert("RGB"), blend_mask)


def restore_original_center(
    generated: Image.Image, original: Image.Image, feather: int
) -> Image.Image:
    """Restore original-resolution center pixels with a narrow feathered seam."""
    target_size = (original.width * 2, original.height * 2)
    result = generated.resize(target_size, Image.Resampling.LANCZOS)
    left, top = original.width // 2, original.height // 2

    feather = max(0, min(feather, original.width // 4, original.height // 4))
    if feather == 0:
        result.paste(original, (left, top))
        return result

    y, x = np.ogrid[: original.height, : original.width]
    horizontal = np.minimum(x, original.width - 1 - x)
    vertical = np.minimum(y, original.height - 1 - y)
    distance = np.minimum(horizontal, vertical).astype(np.float32)
    alpha = np.clip(distance / feather, 0.0, 1.0)
    alpha_mask = Image.fromarray(np.uint8(alpha * 255), mode="L")
    result.paste(original, (left, top), alpha_mask)
    return result


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    original = Image.open(args.input).convert("RGB")
    pipe = build_pipeline(args)
    work_w, work_h = aligned_working_size(
        *original.size, args.model_input_short_side
    )
    working_image = original.resize((work_w, work_h), Image.Resampling.LANCZOS)
    positive = f"{args.prompt.strip()} empty scene".strip()
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.outpaint_mode == "two_stage":
        horizontal_canvas, horizontal_mask = prepare_directional_canvas(
            working_image, "horizontal", args.seam_overlap
        )
        horizontal_generated = run_powerpaint(
            pipe,
            horizontal_canvas,
            horizontal_mask,
            positive,
            args.negative_prompt,
            args,
            args.seed,
        )
        horizontal_result = preserve_known_region(
            horizontal_generated,
            horizontal_canvas,
            horizontal_mask,
            args.seam_overlap,
        )
        if args.save_intermediate:
            intermediate_path = output_path.with_name(
                f"{output_path.stem}_horizontal{output_path.suffix}"
            )
            horizontal_result.save(intermediate_path)
            print(f"Saved horizontal pass to: {intermediate_path}")

        vertical_canvas, vertical_mask = prepare_directional_canvas(
            horizontal_result, "vertical", args.seam_overlap
        )
        vertical_generated = run_powerpaint(
            pipe,
            vertical_canvas,
            vertical_mask,
            positive,
            args.negative_prompt,
            args,
            args.seed + 1,
        )
        result = preserve_known_region(
            vertical_generated,
            vertical_canvas,
            vertical_mask,
            args.seam_overlap,
        )
    else:
        _, canvas, mask = prepare_canvas(
            original, args.model_input_short_side, args.seam_overlap
        )
        generated = run_powerpaint(
            pipe,
            canvas,
            mask,
            positive,
            args.negative_prompt,
            args,
            args.seed,
        )
        result = preserve_known_region(
            generated, canvas, mask, args.seam_overlap
        )

    # The diffusion model works at a reduced resolution. Restore the requested
    # 2W x 2H size, then put the untouched source pixels back into the center.
    final = restore_original_center(result, original, args.final_feather)
    final.save(output_path)
    print(f"Saved {final.width}x{final.height} result to: {output_path}")


if __name__ == "__main__":
    main()
