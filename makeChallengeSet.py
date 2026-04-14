
"""
Challenge Set generator for LLM-as-Judge experiments.

- If you provide RGB + binary masks (0/255), script will make:
  - clean overlays
  - structural corruptions: line breaks, spurious lines, dilate, erode
  - weather/photometric/camera corruptions applied to the overlay

- If you only have overlays, script will still make:
  - weather/photometric/camera corruptions applied to the overlay

Outputs:
- Folder tree under OUT_ROOT with subfolders per corruption type
- CSV manifest listing: image_id, corruption, severity, out_path, source

Usage examples:
  python make_challenge_set.py \
      --rgb_dir data/rgb --mask_dir data/masks_bin --out_root data/challenge_set

  # overlays only
  python make_challenge_set.py \
      --overlay_dir data/overlays --out_root data/challenge_set

"""

import argparse
from pathlib import Path
import random
import math
import cv2
import numpy as np
from PIL import Image
import pandas as pd
import albumentations as A

# ----------------------------
# helpers
# ----------------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def to_uint8(x):
    x = np.clip(x, 0, 255)
    return x.astype(np.uint8)

def blend_overlay(rgb, mask_bin, color=(255, 0, 0), alpha=0.45):
    """Blend a binary mask (0/255) onto rgb as colored overlay (default: red)."""
    if mask_bin.ndim == 3:
        mask_bin = cv2.cvtColor(mask_bin, cv2.COLOR_BGR2GRAY)
    mask = (mask_bin > 127).astype(np.uint8)
    overlay = rgb.copy()
    color_arr = np.zeros_like(rgb)
    # OpenCV uses BGR; we convert tuple (R,G,B) -> (B,G,R)
    color_arr[..., 0] = color[2]  # B
    color_arr[..., 1] = color[1]  # G
    color_arr[..., 2] = color[0]  # R
    overlay = np.where(
        mask[..., None] == 1,
        (alpha * color_arr + (1 - alpha) * overlay).astype(np.uint8),
        overlay,
    )
    return overlay

def random_breaks_on_mask(mask_bin, severity=2, base_len=30):
    """
    Remove short segments along the mask to simulate continuity breaks.
    severity scales number/length of breaks.
    """
    mask = (mask_bin > 127).astype(np.uint8)
    broken = mask.copy()
    ys, xs = np.where(mask == 1)
    if len(xs) == 0:
        return (broken * 255).astype(np.uint8)

    n_breaks = {1: 3, 2: 6, 3: 10}[severity]
    max_len = {1: int(0.5 * base_len), 2: base_len, 3: int(1.5 * base_len)}[severity]
    for _ in range(n_breaks):
        idx = np.random.randint(0, len(xs))
        cx, cy = xs[idx], ys[idx]
        angle = np.random.uniform(0, 2 * math.pi)
        length = np.random.randint(max(6, max_len // 2), max_len)
        x2 = int(cx + length * math.cos(angle))
        y2 = int(cy + length * math.sin(angle))
        cv2.line(broken, (cx, cy), (x2, y2), 0, thickness=np.random.randint(2, 4))
    return (broken * 255).astype(np.uint8)

def add_spurious_lines(mask_bin, severity=2):
    """
    Draw thin false-positive lines at random locations.
    """
    sp = (mask_bin > 127).astype(np.uint8)
    h, w = sp.shape
    n_lines = {1: 3, 2: 6, 3: 10}[severity]
    for _ in range(n_lines):
        x1, y1 = np.random.randint(0, w), np.random.randint(0, h)
        x2, y2 = np.random.randint(0, w), np.random.randint(0, h)
        th = np.random.randint(1, 3)
        cv2.line(sp, (x1, y1), (x2, y2), 1, thickness=th)
    return (sp * 255).astype(np.uint8)

def morph_adjust(mask_bin, severity=2, mode='dilate'):
    k = {1: 1, 2: 2, 3: 3}[severity]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * k + 1, 2 * k + 1))
    if mode == 'dilate':
        out = cv2.dilate(mask_bin, kernel, iterations=1)
    else:
        out = cv2.erode(mask_bin, kernel, iterations=1)
    return out

def build_weather_transforms(severity=2):
    """
    Weather/lighting/camera corruptions via Albumentations.
    Tuned by severity level in {1,2,3}.
    """
    # map severity to param ranges
    fog = {1: (0.02, 0.05), 2: (0.05, 0.09), 3: (0.09, 0.14)}[severity]
    blur = {1: (5, 7), 2: (9, 11), 3: (13, 15)}[severity]
    jpeg = {1: (85, 95), 2: (70, 85), 3: (40, 70)}[severity]
    snow_upper = {1: 0.2, 2: 0.3, 3: 0.45}[severity]
    bright = 0.1 * severity / 3
    contr = 0.1 * severity / 3

    t = {
        "rain": A.RandomRain(
    brightness_coefficient=1,     # Slight brightening, more natural
    drop_length=6 + 4 * severity,    # MUCH shorter raindrops
    drop_width=1,                    # Keep raindrops very thin
    blur_value=1 + severity,         # Light blur, not streaky
    rain_type="drizzle",             # Produces finer, more realistic rain
    p=1.0,
),
        "snow": A.RandomSnow(
            brightness_coeff=2.0 + 0.5 * severity,
            snow_point_lower=0.1,
            snow_point_upper=snow_upper,
            p=1.0,
        ),
        "fog": A.RandomFog(
            fog_coef_lower=fog[0], fog_coef_upper=fog[1], alpha_coef=0.08, p=1.0
        ),
        "sunflare": A.RandomSunFlare(
            flare_roi=(0, 0, 1, 0.5),
            angle_lower=0,
            p=1.0,
            src_radius=120 + 40 * severity,
            src_color=(255, 255, 255),
        ),
        "shadow": A.RandomShadow(
            num_shadows_lower=1, num_shadows_upper=2, shadow_dimension=5, p=1.0
        ),
        "brightness": A.RandomBrightnessContrast(
            brightness_limit=bright, contrast_limit=contr, p=1.0
        ),
        "motion_blur": A.MotionBlur(blur_limit=blur, p=1.0),
        "defocus_blur": A.Defocus(radius=(2, 2 + 2 * severity), p=1.0),
        "gauss_noise": A.GaussNoise(var_limit=(5, 5 + 30 * severity), p=1.0),
        "jpeg": A.ImageCompression(
            quality_lower=jpeg[0], quality_upper=jpeg[1], p=1.0
        ),
    }
    return t

def load_rgb_mask_or_overlay(basename, rgb_dir, mask_dir, overlay_dir):
    """
    Returns (rgb, mask_bin, overlay_if_exists)
    - rgb: HxWx3 RGB or None
    - mask_bin: HxW uint8 (0/255) or None
    - overlay: HxWx3 RGB or None
    """
    stem = Path(basename).stem
    rgb, mask, overlay = None, None, None

    if rgb_dir:
        rp = rgb_dir / basename
        if rp.exists():
            rgb = cv2.cvtColor(cv2.imread(str(rp)), cv2.COLOR_BGR2RGB)

    if mask_dir:
        # Try all common mask extensions
        for ext in [".png", ".jpg", ".jpeg", ".webp"]:
            mp = mask_dir / f"{stem}{ext}"
            if mp.exists():
                m = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
                if m is not None:
                    mask = (m > 127).astype(np.uint8) * 255
                break

    if overlay_dir:
        op = overlay_dir / basename
        if op.exists():
            overlay = cv2.cvtColor(cv2.imread(str(op)), cv2.COLOR_BGR2RGB)

    return rgb, mask, overlay


# ----------------------------
# main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate Challenge Set with structural + weather corruptions.")
    ap.add_argument("--rgb_dir", type=Path, default=None, help="Dir with RGB images (optional if overlays only).")
    ap.add_argument("--mask_dir", type=Path, default=None, help="Dir with binary masks 0/255 (optional).")
    ap.add_argument("--overlay_dir", type=Path, default=None, help="Dir with existing overlays (optional).")
    ap.add_argument("--out_root", type=Path, required=True, help="Output root directory.")
    ap.add_argument("--mask_color", type=str, default="255,0,0", help="Overlay color as R,G,B (default: red).")
    ap.add_argument("--alpha", type=float, default=0.45, help="Overlay opacity (0..1).")
    ap.add_argument("--severities", type=str, default="1,2,3", help="Comma list of severities to generate.")
    ap.add_argument("--exts", type=str, default=".png,.jpg,.jpeg,.webp", help="Image extensions to consider.")
    args = ap.parse_args()

    color = tuple(int(x) for x in args.mask_color.split(","))
    severities = [int(x) for x in args.severities.split(",")]
    exts = set(e.strip().lower() for e in args.exts.split(","))

    # Validate inputs
    if args.overlay_dir is None and (args.rgb_dir is None or args.mask_dir is None):
        print("Provide either --overlay_dir OR both --rgb_dir and --mask_dir.")
        return

    # Collect file list
    if args.overlay_dir:
        file_list = sorted([p for p in args.overlay_dir.iterdir() if p.suffix.lower() in exts])
        source = "overlay"
    else:
        file_list = sorted([p for p in args.rgb_dir.iterdir() if p.suffix.lower() in exts])
        source = "rgb+mask"

    # Prepare outputs
    OUTS = {
        "clean": args.out_root / "clean",
        # structural (masks required)
        "breaks": args.out_root / "breaks",
        "spurious": args.out_root / "spurious",
        "dilate": args.out_root / "dilate",
        "erode": args.out_root / "erode",
        # weather/photometric/camera
        "rain": args.out_root / "rain",
        "snow": args.out_root / "snow",
        "fog": args.out_root / "fog",
        "sunflare": args.out_root / "sunflare",
        "shadow": args.out_root / "shadow",
        "brightness": args.out_root / "brightness",
        "motion_blur": args.out_root / "motion_blur",
        "defocus_blur": args.out_root / "defocus_blur",
        "gauss_noise": args.out_root / "gauss_noise",
        "jpeg": args.out_root / "jpeg",
    }
    for p in OUTS.values():
        ensure_dir(p)

    rows = []

    for src in file_list:
        base = src.name
        rgb, mask_bin, overlay = load_rgb_mask_or_overlay(
            base, args.rgb_dir, args.mask_dir, args.overlay_dir
        )

        if rgb is None and overlay is None:
            print(f"[skip] cannot load {base}")
            continue

        # Build a clean overlay
        if overlay is not None:
            ov_clean = overlay
        else:
            if mask_bin is None:
                print(f"[skip] missing mask for {base}")
                continue
            ov_clean = blend_overlay(rgb, mask_bin, color=color, alpha=args.alpha)

        out_clean = OUTS["clean"] / base
        Image.fromarray(ov_clean).save(out_clean)
        rows.append({
            "image_id": base,
            "corruption": "clean",
            "severity": 0,
            "out_path": str(out_clean),
            "source": source
        })

        # Structural corruptions if mask is available
        if mask_bin is not None:
            for sev in severities:
                # breaks
                m_broken = random_breaks_on_mask(mask_bin, severity=sev)
                ov = blend_overlay(rgb, m_broken, color=color, alpha=args.alpha)
                p = OUTS["breaks"] / f"{src.stem}_brk_s{sev}{src.suffix}"
                Image.fromarray(ov).save(p)
                rows.append({"image_id": base, "corruption": "breaks", "severity": sev, "out_path": str(p), "source": source})

                # spurious
                m_sp = add_spurious_lines(mask_bin.copy(), severity=sev)
                ov = blend_overlay(rgb, m_sp, color=color, alpha=args.alpha)
                p = OUTS["spurious"] / f"{src.stem}_spu_s{sev}{src.suffix}"
                Image.fromarray(ov).save(p)
                rows.append({"image_id": base, "corruption": "spurious", "severity": sev, "out_path": str(p), "source": source})

                # dilate / erode
                md = morph_adjust(mask_bin, severity=sev, mode="dilate")
                me = morph_adjust(mask_bin, severity=sev, mode="erode")
                ov = blend_overlay(rgb, md, color=color, alpha=args.alpha)
                p = OUTS["dilate"] / f"{src.stem}_dil_s{sev}{src.suffix}"
                Image.fromarray(ov).save(p)
                rows.append({"image_id": base, "corruption": "dilate", "severity": sev, "out_path": str(p), "source": source})

                ov = blend_overlay(rgb, me, color=color, alpha=args.alpha)
                p = OUTS["erode"] / f"{src.stem}_ero_s{sev}{src.suffix}"
                Image.fromarray(ov).save(p)
                rows.append({"image_id": base, "corruption": "erode", "severity": sev, "out_path": str(p), "source": source})

        # Weather/photometric/camera corruptions on overlay image
        for sev in severities:
            tf = build_weather_transforms(severity=sev)
            for cname, aug in tf.items():
                img2 = aug(image=ov_clean)["image"]
                p = OUTS[cname] / f"{src.stem}_{cname}_s{sev}{src.suffix}"
                Image.fromarray(img2).save(p)
                rows.append({"image_id": base, "corruption": cname, "severity": sev, "out_path": str(p), "source": source})

    # Save manifest
    manifest = pd.DataFrame(rows)
    ensure_dir(args.out_root)
    manifest_path = args.out_root / "challenge_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\n[OK] Challenge set saved under: {args.out_root}")
    print(f"[OK] Manifest: {manifest_path}  (rows={len(manifest)})")

if __name__ == "__main__":
    main()
