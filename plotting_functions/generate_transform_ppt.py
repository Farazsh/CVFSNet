"""Generate a PowerPoint review deck for CVFSNet transform outputs.

For each TICI label (T0, T1, T2a, T2b, T3), this script randomly picks one
subject ID from ``AmTICIS_processed_CVFS_paper/<LABEL>`` and builds slides for
both AP and sagittal views.

Slide design:
- 5 rows total.
- Row 1: non-transformed clip (all 8 frames).
- Rows 2-5: one transform per row, using only frames 3, 5, 7.
- Left 1/4 of slide: one text box per row with transform details.
- Right 3/4 of slide: frame thumbnails.
- Title: subject ID + TICI score.

Transforms are grouped as 4 per slide (1-4, 5-8, 9-12, 13-16).
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List, Tuple

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt
from PIL import Image


ROOT = Path(__file__).resolve().parent
BASE_DIR = ROOT / "AmTICIS_processed_CVFS_paper"
TRANSFORM_DIR = ROOT / "AmTICIS_transforms"
DEFAULT_OUTPUT = ROOT / "AmTICIS_transforms_review.pptx"

LABELS = ["T0", "T1", "T2a", "T2b", "T3"]
VIEWS = ["AP", "sagittal"]

# For transformed rows, show only these frames.
TRANSFORM_FRAME_NAMES = ["frame_003.png", "frame_005.png", "frame_007.png"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output PPTX path.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used to select one subject per label.",
    )
    return parser.parse_args()


def get_transform_dirs() -> List[Path]:
    dirs = [d for d in TRANSFORM_DIR.iterdir() if d.is_dir()]
    dirs.sort(key=lambda p: p.name)
    if not dirs:
        raise FileNotFoundError(f"No transform folders found in {TRANSFORM_DIR}")
    return dirs


def choose_subjects(seed: int) -> Dict[str, str]:
    rng = random.Random(seed)
    chosen: Dict[str, str] = {}
    for label in LABELS:
        label_dir = BASE_DIR / label
        if not label_dir.exists():
            raise FileNotFoundError(f"Missing label directory: {label_dir}")
        subjects = sorted([d.name for d in label_dir.iterdir() if d.is_dir()])
        if not subjects:
            raise ValueError(f"No subjects found for label {label}")
        chosen[label] = rng.choice(subjects)
    return chosen


def read_params_txt(params_file: Path) -> Dict[str, str]:
    if not params_file.exists():
        return {}
    lines = [ln.strip() for ln in params_file.read_text().splitlines() if ln.strip()]
    lines = [ln for ln in lines if not set(ln) <= {"-"} and ":" in ln]
    excluded_keys = {
        "transform",
        "output_frames",
        "source_nifti",
        "native_shape_HWT",
        "resampled_to",
        "note",
    }
    values_only: Dict[str, str] = {}
    for ln in lines:
        key, value = [x.strip() for x in ln.split(":", 1)]
        if key in excluded_keys:
            continue
        values_only[key] = value
    return values_only


def get_text_lines(transform_name: str, params: Dict[str, str]) -> List[str]:
    """Return exactly 3 text lines: name + two key values."""
    if transform_name == "No Transform":
        return [transform_name, "value1: none", "value2: none"]

    preferred = {
        "01_RandomErode": ["kernel", "dataloader_p"],
        "02_RandomDilate": ["kernel", "dataloader_p"],
        "03_TioClamp": ["out_min", "out_max"],
        "04_TioRandomFlip": ["axes", "flip_probability"],
        "05_TioRandomAnisotropy": ["axis", "downsampling_factor"],
        "06_TioRandomMotion": ["degrees_max", "translation_max"],
        "07_TioRandomGhosting": ["num_ghosts", "axis"],
        "08_TioRandomSpike": ["num_spikes", "intensity_range"],
        "09_TioRandomBiasField": ["coefficients_max", "order"],
        "10_TioRandomBlur": ["std", "std_range"],
        "11_TioRandomNoise": ["std", "std_range"],
        "12_TioRandomGamma": ["log_gamma", "gamma"],
        "13_RandomRotation": ["angle_deg", "degrees_range"],
        "14_Crop": ["x_left", "x_right"],
        "15_Resize": ["t", "visual"],
        "16_TioZNormalization": ["input_mean", "input_std"],
    }

    keys = preferred.get(transform_name, [])
    vals = []
    for k in keys:
        if k in params:
            vals.append(f"{k}: {params[k]}")
    if len(vals) < 2:
        for k, v in params.items():
            if k in {"history", "seed"}:
                continue
            candidate = f"{k}: {v}"
            if candidate not in vals:
                vals.append(candidate)
            if len(vals) == 2:
                break
    while len(vals) < 2:
        vals.append("value: none")
    return [transform_name, vals[0], vals[1]]


def get_base_frames(label: str, subject: str, view: str) -> List[Path]:
    view_dir = BASE_DIR / label / subject / view
    frames = sorted(view_dir.glob("frame_*.png"))
    if len(frames) < 8:
        raise ValueError(f"Expected >=8 frames in {view_dir}, got {len(frames)}")
    return [frames[3], frames[5], frames[7]]


def get_transform_frames(transform_dir: Path, label: str, subject: str, view: str) -> List[Path]:
    leaf = transform_dir / label / subject / view
    if not leaf.exists():
        raise FileNotFoundError(f"Missing transform output leaf: {leaf}")

    preferred = [leaf / name for name in TRANSFORM_FRAME_NAMES]
    if all(p.exists() for p in preferred):
        return preferred

    frames = sorted(leaf.glob("frame_*.png"))
    if len(frames) < 8:
        raise ValueError(f"Expected >=8 frames in {leaf}, got {len(frames)}")
    return [frames[3], frames[5], frames[7]]


def add_text_box(slide, left, top, width, height, lines: List[str], font_size: int) -> None:
    """Add a 3-line text box: transform name + two values."""
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True

    p = tf.paragraphs[0]
    p.text = lines[0]
    p.font.bold = True
    p.font.size = Pt(font_size + 1)
    p.font.color.rgb = RGBColor(20, 20, 20)

    for line in lines[1:3]:
        p2 = tf.add_paragraph()
        p2.text = line
        p2.level = 0
        p2.font.size = Pt(font_size)
        p2.font.color.rgb = RGBColor(50, 50, 50)


def _fit_size_keep_aspect(
    img_wh: Tuple[int, int], max_w, max_h
) -> Tuple[int, int]:
    iw, ih = img_wh
    scale = min(max_w / iw, max_h / ih)
    return int(iw * scale), int(ih * scale)


def add_image_row(
    slide,
    frame_paths: List[Path],
    left,
    top,
    width,
    row_height,
    preserve_aspect: bool = False,
    align_right: bool = False,
) -> None:
    n = len(frame_paths)
    gap = Inches(0.05)
    inner_top_pad = Inches(0.02)
    max_h = row_height - Inches(0.04)

    if not preserve_aspect:
        total_gap = gap * (n - 1)
        cell_w = (width - total_gap) / n
        for i, fpath in enumerate(frame_paths):
            x = left + i * (cell_w + gap)
            y = top + inner_top_pad
            slide.shapes.add_picture(str(fpath), x, y, width=cell_w, height=max_h)
        return

    # Preserve native aspect ratios and right-align the row content.
    sizes = []
    for fpath in frame_paths:
        with Image.open(fpath) as im:
            sizes.append((im.width, im.height))
    fitted = [_fit_size_keep_aspect(s, max_h, max_h) for s in sizes]
    content_w = sum(w for w, _ in fitted) + gap * (n - 1)
    if content_w > width:
        shrink = width / content_w
        fitted = [(int(w * shrink), int(h * shrink)) for w, h in fitted]
        content_w = width
    start_x = left + (width - content_w if align_right else 0)
    cur_x = start_x
    for (fw, fh), fpath in zip(fitted, frame_paths):
        y = top + inner_top_pad + (max_h - fh) / 2
        slide.shapes.add_picture(str(fpath), cur_x, y, width=fw, height=fh)
        cur_x += fw + gap


def build_slides(
    prs: Presentation, transform_groups: List[List[Path]], label: str, subject: str, view: str
) -> None:
    blank = prs.slide_layouts[6]

    slide_w = prs.slide_width
    slide_h = prs.slide_height

    left_panel_w = int(slide_w * 0.25)
    right_panel_x = left_panel_w
    right_panel_w = slide_w - left_panel_w

    top_margin = Inches(0.70)
    bottom_margin = Inches(0.20)
    usable_h = slide_h - top_margin - bottom_margin
    row_h = usable_h / 5

    base_frames = get_base_frames(label, subject, view)

    for group in transform_groups:
        slide = prs.slides.add_slide(blank)

        title = slide.shapes.add_textbox(Inches(0.2), Inches(0.05), slide_w - Inches(0.4), Inches(0.5))
        ttf = title.text_frame
        ttf.clear()
        p = ttf.paragraphs[0]
        p.text = f"Subject {subject}  |  TICI {label}"
        p.font.size = Pt(24)
        p.font.bold = True
        p.alignment = PP_ALIGN.CENTER

        row0_top = top_margin
        add_image_row(
            slide,
            base_frames,
            right_panel_x + Inches(0.05),
            row0_top,
            right_panel_w - Inches(0.10),
            row_h,
            preserve_aspect=True,
            align_right=True,
        )
        base_lines = get_text_lines("No Transform", {})
        add_text_box(
            slide,
            Inches(0.08),
            row0_top + Inches(0.02),
            left_panel_w - Inches(0.16),
            row_h - Inches(0.04),
            base_lines,
            font_size=10,
        )

        for idx, tdir in enumerate(group):
            row_top = top_margin + row_h * (idx + 1)
            tf_frames = get_transform_frames(tdir, label, subject, view)
            add_image_row(
                slide,
                tf_frames,
                right_panel_x + Inches(0.05),
                row_top,
                right_panel_w - Inches(0.10),
                row_h,
                preserve_aspect=True,
                align_right=True,
            )

            params_dict = read_params_txt(tdir / label / subject / view / "params.txt")
            params_lines = get_text_lines(tdir.name, params_dict)
            add_text_box(
                slide,
                Inches(0.08),
                row_top + Inches(0.02),
                left_panel_w - Inches(0.16),
                row_h - Inches(0.04),
                params_lines,
                font_size=9,
            )


def chunked(items: List[Path], size: int) -> List[List[Path]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def main() -> None:
    args = parse_args()

    transform_dirs = get_transform_dirs()
    groups = chunked(transform_dirs, 4)
    chosen = choose_subjects(args.seed)

    prs = Presentation()

    for label in LABELS:
        subject = chosen[label]
        for view in VIEWS:
            build_slides(prs, groups, label, subject, view)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(args.output))

    print(f"Saved PPT: {args.output}")
    for label in LABELS:
        print(f"  {label}: subject {chosen[label]}")


if __name__ == "__main__":
    main()
