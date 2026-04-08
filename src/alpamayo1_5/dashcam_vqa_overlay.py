# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Burn VQA captions from JSONL onto an MP4 (PyAV decode/encode + Pillow overlay).

Each JSONL line must be ``{"anchor_sec": <float>, "answer": <str>}``. For each
decoded frame at time ``t``, the caption is the answer from the latest row with
``anchor_sec <= t``; before the first such anchor, a placeholder is shown.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


def load_vqa_jsonl(path: Path) -> list[tuple[float, str]]:
    """Load and validate JSONL; return sorted (anchor_sec, answer) rows."""
    rows: list[tuple[float, str]] = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj: dict[str, Any] = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON: {e}") from e
            if "anchor_sec" not in obj or "answer" not in obj:
                raise ValueError(f"{path}:{lineno}: expected keys anchor_sec and answer")
            anchor = obj["anchor_sec"]
            answer = obj["answer"]
            if not isinstance(anchor, (int, float)):
                raise ValueError(f"{path}:{lineno}: anchor_sec must be numeric")
            if not isinstance(answer, str):
                raise ValueError(f"{path}:{lineno}: answer must be a string")
            rows.append((float(anchor), answer))
    rows.sort(key=lambda x: x[0])
    return rows


def _video_frame_total(container: av.container.Container, stream: av.VideoStream) -> int | None:
    """Best-effort total frame count for progress (may be None if unknown)."""
    n = getattr(stream, "frames", None)
    if n is not None and int(n) > 0:
        return int(n)
    dur_us = getattr(container, "duration", None)
    if dur_us is None or dur_us <= 0:
        return None
    try:
        fps = float(stream.average_rate)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    if fps <= 0:
        return None
    est = int(round(float(dur_us) / 1_000_000 * fps))
    return est if est > 0 else None


def _approx_fps(rate: Any) -> int:
    """Integer FPS for mux defaults; handles Fraction / PyAV rational types."""
    try:
        x = float(rate)
    except (TypeError, ValueError, ZeroDivisionError):
        return 30
    if x <= 0:
        return 30
    return max(1, int(round(x)))


def frame_time_sec(frame: av.VideoFrame, stream: av.VideoStream) -> float:
    """Presentation time in seconds for a decoded video frame."""
    tb = getattr(frame, "time_base", None) or stream.time_base
    if frame.pts is not None and tb is not None:
        return float(frame.pts * tb)
    t = getattr(frame, "time", None)
    if t is not None and not (isinstance(t, float) and np.isnan(t)):
        return float(t)
    return 0.0


def _font_candidates() -> list[str]:
    return [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        str(Path.home() / ".local/share/fonts/DejaVuSans.ttf"),
    ]


def load_font(size_px: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for p in _font_candidates():
        if os.path.isfile(p):
            try:
                return ImageFont.truetype(p, size=size_px)
            except OSError:
                continue
    return ImageFont.load_default()


def wrap_lines(
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_width: int,
    draw: ImageDraw.ImageDraw,
) -> list[str]:
    """Word-wrap ``text`` into lines that fit ``max_width`` (px)."""
    lines: list[str] = []
    for paragraph in text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            lines.append("")
            continue
        words = paragraph.split()
        current: list[str] = []
        for w in words:
            trial = (" ".join(current + [w])).strip()
            bbox = draw.textbbox((0, 0), trial, font=font)
            w_px = bbox[2] - bbox[0]
            if w_px <= max_width or not current:
                current.append(w)
            else:
                lines.append(" ".join(current))
                current = [w]
        if current:
            lines.append(" ".join(current))
    return lines if lines else [""]


def line_height(font: ImageFont.FreeTypeFont | ImageFont.ImageFont, draw: ImageDraw.ImageDraw) -> int:
    bbox = draw.textbbox((0, 0), "Ay", font=font)
    return max(1, bbox[3] - bbox[1])


def draw_caption_bar(
    image: Image.Image,
    caption: str,
    *,
    position: str,
    font_px: int,
    max_lines: int,
    padding: int,
    bar_alpha: int,
) -> Image.Image:
    """Return RGB image with semi-transparent caption bar and wrapped text."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    font = load_font(font_px)
    draw_tmp = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    max_text_w = max(8, w - 2 * padding)
    lines = wrap_lines(caption, font, max_text_w, draw_tmp)
    if len(lines) > max_lines:
        lines = lines[: max_lines - 1]
        if lines:
            ell = " …"
            while lines[-1] and draw_tmp.textbbox((0, 0), lines[-1] + ell, font=font)[2] > max_text_w:
                lines[-1] = lines[-1][:-1]
            lines[-1] = (lines[-1].rstrip() + ell).strip()
        else:
            lines = ["…"]

    lh = line_height(font, draw_tmp)
    line_gap = max(2, font_px // 8)
    text_block_h = len(lines) * lh + (len(lines) - 1) * line_gap
    bar_h = padding * 2 + text_block_h
    bar_h = min(bar_h, h // 2)

    base = image.convert("RGBA")
    overlay = Image.new("RGBA", (w, bar_h), (0, 0, 0, 0))
    bar_draw = ImageDraw.Draw(overlay)
    bar_draw.rectangle((0, 0, w, bar_h), fill=(0, 0, 0, bar_alpha))

    y0 = padding
    for i, line in enumerate(lines):
        bbox = bar_draw.textbbox((0, 0), line, font=font)
        tw = bbox[2] - bbox[0]
        x = (w - tw) // 2
        y = y0 + i * (lh + line_gap)
        bar_draw.text((x, y), line, font=font, fill=(255, 255, 255, 255))

    if position == "top":
        base.paste(overlay, (0, 0), overlay)
    else:
        base.paste(overlay, (0, h - bar_h), overlay)
    return base.convert("RGB")


def run_overlay(
    video_path: Path,
    jsonl_path: Path,
    output_path: Path,
    *,
    placeholder: str,
    font_size: int | None,
    position: str,
    max_lines: int,
    bar_alpha: int,
    show_progress: bool = True,
) -> None:
    timeline = load_vqa_jsonl(jsonl_path)
    event_i = 0
    current_answer: str | None = None

    input_container = av.open(str(video_path))
    try:
        in_stream = input_container.streams.video[0]
        in_stream.thread_type = "NONE"

        w = in_stream.width
        h = in_stream.height
        if w is None or h is None:
            raise ValueError("Input video stream has no width/height")

        tb_in = in_stream.time_base
        rate = in_stream.average_rate
        if rate is None:
            rate = 30
        else:
            try:
                if float(rate) <= 0:
                    rate = 30
            except (TypeError, ValueError, ZeroDivisionError):
                rate = 30

        output_container = av.open(str(output_path), mode="w")
        try:
            out_stream = output_container.add_stream("libx264", rate=rate)
            out_stream.width = w
            out_stream.height = h
            out_stream.pix_fmt = "yuv420p"
            if tb_in is not None:
                out_stream.time_base = tb_in
            else:
                out_stream.time_base = Fraction(1, _approx_fps(rate))

            frame_total = _video_frame_total(input_container, in_stream)
            decode_iter = input_container.decode(in_stream)
            if show_progress:
                decode_iter = tqdm(
                    decode_iter,
                    total=frame_total,
                    desc="Overlay",
                    unit="frame",
                    dynamic_ncols=True,
                )

            for frame in decode_iter:
                if not isinstance(frame, av.VideoFrame):
                    continue
                t = frame_time_sec(frame, in_stream)
                while event_i < len(timeline) and timeline[event_i][0] <= t:
                    current_answer = timeline[event_i][1]
                    event_i += 1

                caption = current_answer if current_answer is not None else placeholder
                fpx = font_size if font_size is not None else max(16, int(h * 0.022))

                arr = frame.to_ndarray(format="rgb24")
                pil = Image.fromarray(arr, mode="RGB")
                pil = draw_caption_bar(
                    pil,
                    caption,
                    position=position,
                    font_px=fpx,
                    max_lines=max_lines,
                    padding=max(8, fpx // 2),
                    bar_alpha=bar_alpha,
                )
                rgb = np.asarray(pil, dtype=np.uint8)
                out_frame = av.VideoFrame.from_ndarray(rgb, format="rgb24")
                out_frame.pts = frame.pts
                out_frame.time_base = frame.time_base or tb_in or out_stream.time_base

                yuv = out_frame.reformat(format="yuv420p", width=w, height=h)
                yuv.pts = out_frame.pts
                yuv.time_base = out_frame.time_base

                for packet in out_stream.encode(yuv):
                    output_container.mux(packet)

            for packet in out_stream.encode(None):
                output_container.mux(packet)
        finally:
            output_container.close()
    finally:
        input_container.close()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Burn VQA JSONL captions onto an MP4 using Pillow + PyAV.",
    )
    p.add_argument("--video", type=Path, required=True, help="Input MP4 path")
    p.add_argument("--jsonl", type=Path, required=True, help="VQA answers JSONL path")
    p.add_argument("--output", type=Path, required=True, help="Output MP4 path")
    p.add_argument(
        "--placeholder",
        type=str,
        default="warming up…",
        help="Caption before the first anchor_sec is reached (default: warming up…)",
    )
    p.add_argument(
        "--font-size",
        type=int,
        default=None,
        help="Font size in pixels (default: max(16, int(frame_height * 0.022)))",
    )
    p.add_argument(
        "--position",
        choices=("bottom", "top"),
        default="bottom",
        help="Caption bar position (default: bottom)",
    )
    p.add_argument(
        "--max-lines",
        type=int,
        default=8,
        help="Maximum wrapped lines; extra text is truncated with … (default: 8)",
    )
    p.add_argument(
        "--bar-alpha",
        type=int,
        default=200,
        help="Caption bar opacity 0–255 (default: 200)",
    )
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bar",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if not args.video.is_file():
        print(f"error: video not found: {args.video}", file=sys.stderr)
        return 1
    if not args.jsonl.is_file():
        print(f"error: jsonl not found: {args.jsonl}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run_overlay(
        args.video,
        args.jsonl,
        args.output,
        placeholder=args.placeholder,
        font_size=args.font_size,
        position=args.position,
        max_lines=args.max_lines,
        bar_alpha=max(0, min(255, args.bar_alpha)),
        show_progress=not args.no_progress,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
