# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms


class FARM(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.freq_controller = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_channels * 2),
            nn.GELU(),
            nn.Conv2d(
                in_channels * 2,
                in_channels * 2,
                kernel_size=3,
                padding=1,
                groups=in_channels * 2,
                bias=False,
            ),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.alpha = nn.Parameter(torch.tensor([0.1], dtype=torch.float32))

    def forward(self, x: torch.Tensor, return_vis: bool = False):
        # Same FFT normalization and log-amplitude formulation as GitHub FARM.py
        fft_x = torch.fft.fft2(x, norm="ortho")
        fft_x_shifted = torch.fft.fftshift(fft_x, dim=(-2, -1))

        amplitude = torch.log1p(torch.abs(fft_x_shifted))
        m_freq = self.freq_controller(amplitude)

        fft_x_filtered = fft_x_shifted * m_freq
        fft_x_ishifted = torch.fft.ifftshift(fft_x_filtered, dim=(-2, -1))
        x_restored = torch.fft.ifft2(fft_x_ishifted, norm="ortho").real

        output = x + self.alpha * x_restored
        output = torch.clamp(output, 0.0, 1.0)

        if return_vis:
            return output, amplitude, m_freq
        return output


def _unwrap_checkpoint(obj: Any) -> Dict[str, torch.Tensor]:
    """
    Supports common checkpoint formats:
      - raw state_dict
      - {"state_dict": ...}
      - {"model_state_dict": ...}
      - {"model": ...}
      - {"net": ...}
    """
    if not isinstance(obj, dict):
        raise TypeError("Checkpoint is not a dictionary/state_dict.")

    for key in ("state_dict", "model_state_dict", "model", "net"):
        if key in obj and isinstance(obj[key], dict):
            return obj[key]

    # raw state_dict
    if all(isinstance(k, str) for k in obj.keys()):
        return obj

    raise RuntimeError("Unsupported checkpoint structure.")


def extract_farm_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = (
        "farm.",
        "module.farm.",
        "model.farm.",
        "module.model.farm.",
        "network.farm.",
    )

    for prefix in prefixes:
        subset = {
            k[len(prefix):]: v
            for k, v in state_dict.items()
            if k.startswith(prefix)
        }
        if subset:
            return subset

    if (
        "alpha" in state_dict
        or any(k.startswith("freq_controller.") for k in state_dict)
    ):
        return state_dict

    sample_keys = list(state_dict.keys())[:30]
    raise RuntimeError(
        "No FARM parameters were found in the checkpoint.\n"
        "The first checkpoint keys are:\n  "
        + "\n  ".join(sample_keys)
    )


def load_trained_farm(weight_path: Path, device: torch.device) -> FARM:
    if not weight_path.is_file():
        raise FileNotFoundError(
            f"Cannot find checkpoint:\n{weight_path}\n\n"
            "Put best_vmamba_laryngeal.pth in the same folder as this script."
        )

    print(f"[INFO] Loading checkpoint: {weight_path}")
    raw = torch.load(str(weight_path), map_location=device)
    state_dict = _unwrap_checkpoint(raw)
    farm_state = extract_farm_state_dict(state_dict)

    model = FARM(in_channels=3).to(device)

    try:
        model.load_state_dict(farm_state, strict=True)
    except RuntimeError as e:
        print("\n[ERROR] FARM weights do not exactly match this FARM structure.")
        print("[INFO] Extracted FARM keys:")
        for k in farm_state.keys():
            print("   ", k)
        raise e

    model.eval()
    print("[INFO] FARM weights loaded successfully.")
    print(f"[INFO] Learned alpha = {model.alpha.item():.6f}")
    return model


CLASS_ORDER = [
    ("Laryngeal_carcinoma.jpg", "Laryngeal carcinoma"),
    ("Vocal_fold_polyp.jpg", "Vocal fold polyp"),
    ("Vocal_fold_leukoplakia.jpg", "Vocal fold leukoplakia"),
    ("Chorditis_vocalis.jpg", "Chorditis vocalis"),
    ("Normal_vocal_folds.jpg", "Normal vocal folds"),
    ("Sulcus_vocalis.jpg", "Sulcus vocalis"),
]


def find_image(input_dir: Path, preferred_filename: str) -> Path:
    stem = Path(preferred_filename).stem.lower()

    candidates = []
    for p in input_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
            continue
        if p.stem.lower() == stem:
            return p
        candidates.append(p)

    raise FileNotFoundError(
        f"Cannot find image for '{preferred_filename}' in:\n{input_dir}\n"
        f"Available image files: {[p.name for p in candidates]}"
    )


def load_image(image_path: Path, image_size: int = 256) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    return transform(image)


def tensor_to_rgb(x: torch.Tensor) -> np.ndarray:
    return (
        x.detach()
         .squeeze(0)
         .permute(1, 2, 0)
         .cpu()
         .numpy()
    )


def create_before_after_figure(
    input_dir: Path,
    model: FARM,
    device: torch.device,
    output_dir: Path,
    image_size: int = 256,
    dpi: int = 400,
):
    originals = []
    processed = []
    labels = []

    with torch.no_grad():
        for filename, label in CLASS_ORDER:
            image_path = find_image(input_dir, filename)
            x = load_image(image_path, image_size=image_size).unsqueeze(0).to(device)

            y, amplitude, freq_mask = model(x, return_vis=True)

            ori_np = np.clip(tensor_to_rgb(x), 0.0, 1.0)
            out_np = np.clip(tensor_to_rgb(y), 0.0, 1.0)

            originals.append(ori_np)
            processed.append(out_np)
            labels.append(label)

            print(
                f"[INFO] {filename:<30} "
                f"mask mean={freq_mask.mean().item():.4f}, "
                f"min={freq_mask.min().item():.4f}, "
                f"max={freq_mask.max().item():.4f}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        2, 6,
        figsize=(15.8, 5.25),
        dpi=dpi,
        gridspec_kw={"wspace": 0.035, "hspace": 0.10},
    )

    for col, label in enumerate(labels):
        axes[0, col].imshow(originals[col])
        axes[0, col].set_title(label, fontsize=10, pad=5)
        axes[0, col].axis("off")

        axes[1, col].imshow(processed[col])
        axes[1, col].axis("off")

    axes[0, 0].text(
        -0.11, 0.5, "Before FARM",
        transform=axes[0, 0].transAxes,
        rotation=90, va="center", ha="center",
        fontsize=11, fontweight="bold",
    )
    axes[1, 0].text(
        -0.11, 0.5, "After FARM",
        transform=axes[1, 0].transAxes,
        rotation=90, va="center", ha="center",
        fontsize=11, fontweight="bold",
    )

    plt.subplots_adjust(left=0.055, right=0.995, top=0.94, bottom=0.02)

    png_path = output_dir / "farm_before_after_2x6.png"
    svg_path = output_dir / "farm_before_after_2x6.svg"
    pdf_path = output_dir / "farm_before_after_2x6.pdf"

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)

    print("\n[OK] Figure generation completed.")
    print(f"[OK] PNG: {png_path}")
    print(f"[OK] SVG: {svg_path}")
    print(f"[OK] PDF: {pdf_path}")


def main():
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Generate a 2x6 before/after FARM figure using trained FARM weights."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(script_dir / "best_vmamba_laryngeal.pth"),
        help="Full FM-VMamba checkpoint. Default: best_vmamba_laryngeal.pth beside the script.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=str(script_dir / "test_image"),
        help="Folder containing the six test images. Default: ./test_image",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(script_dir / "farm_vis_output"),
        help="Output folder. Default: ./farm_vis_output",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Image resize used by the GitHub visualization code. Default: 256.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="PNG resolution. Default: 400 dpi.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )
    args = parser.parse_args()

    device = torch.device(
        "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"[INFO] Device: {device}")

    checkpoint = Path(args.checkpoint)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Cannot find test-image folder:\n{input_dir}\n\n"
            "Create a 'test_image' folder beside the script and put the six images inside."
        )

    model = load_trained_farm(checkpoint, device)

    create_before_after_figure(
        input_dir=input_dir,
        model=model,
        device=device,
        output_dir=output_dir,
        image_size=args.image_size,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
