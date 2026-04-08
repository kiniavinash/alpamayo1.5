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

"""Run Alpamayo 1.5 VQA on a dashcam MP4 and write JSONL for ``dashcam_vqa_overlay``.

Each output line is ``{"anchor_sec": <float>, "answer": <str>}``, matching
:func:`alpamayo1_5.dashcam_vqa_overlay.load_vqa_jsonl`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import av
import numpy as np
import torch
from tqdm import tqdm

from alpamayo1_5 import helper
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5


def frame_time_sec(frame: av.VideoFrame, stream: av.VideoStream) -> float:
    """Presentation time in seconds for a decoded video frame (matches overlay)."""
    tb = getattr(frame, "time_base", None) or stream.time_base
    if frame.pts is not None and tb is not None:
        return float(frame.pts * tb)
    t = getattr(frame, "time", None)
    if t is not None and not (isinstance(t, float) and np.isnan(t)):
        return float(t)
    return 0.0


def compute_anchor_times(
    *,
    duration_sec: float | None,
    sample_every: float,
    start_sec: float,
    end_sec: float | None,
) -> list[float]:
    """Return sorted anchor times in seconds (inclusive of start, step ``sample_every``)."""
    if sample_every <= 0:
        raise ValueError("sample_every must be positive")
    if start_sec < 0:
        raise ValueError("start_sec must be non-negative")

    upper = end_sec if end_sec is not None else duration_sec
    if upper is None:
        raise ValueError("Need duration_sec or end_sec to bound anchors")
    if upper < start_sec:
        return []

    anchors: list[float] = []
    t = start_sec
    # Float-safe stepping
    n = 0
    max_n = int(np.ceil((upper - start_sec) / sample_every)) + 2
    while t <= upper + 1e-9 and n < max_n:
        anchors.append(round(t, 6))
        n += 1
        t = start_sec + n * sample_every
    return anchors


def select_window_frames(
    buffer: list[tuple[float, torch.Tensor]],
    anchor_sec: float,
    num_frames: int,
) -> list[torch.Tensor]:
    """Pick the last ``num_frames`` frames with timestamp <= ``anchor_sec``."""
    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    eligible = [(ti, fi) for (ti, fi) in buffer if ti <= anchor_sec + 1e-9]
    if not eligible:
        return []
    last_n = eligible[-num_frames:]
    frames = [fi for _, fi in last_n]
    while len(frames) < num_frames and frames:
        frames.insert(0, frames[0])
    return frames


def _first_answer(extra: dict[str, Any]) -> str:
    """Extract a single answer string from ``generate_text`` output."""
    ans = extra.get("answer")
    if ans is None:
        return ""
    flat = np.asarray(ans).ravel()
    if flat.size == 0:
        return ""
    out = flat[0]
    return out if isinstance(out, str) else str(out)


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run VQA on sampled MP4 frames; write JSONL for dashcam_vqa_overlay.",
    )
    p.add_argument("--video", type=Path, required=True, help="Input MP4 path")
    p.add_argument(
        "--output-jsonl",
        type=Path,
        required=True,
        help="Output JSONL path (one object per line: anchor_sec, answer)",
    )
    p.add_argument(
        "--question",
        type=str,
        required=True,
        help="VQA question (same for every anchor)",
    )
    p.add_argument(
        "--sample-every-seconds",
        type=float,
        default=1.0,
        help="Seconds between anchor times (default: 1.0)",
    )
    p.add_argument(
        "--num-frames",
        type=int,
        default=4,
        help="Temporal frames per prompt (last N frames at/before each anchor; default: 4)",
    )
    p.add_argument(
        "--buffer-frames",
        type=int,
        default=600,
        help="Max decoded frames kept in memory for window selection (default: 600)",
    )
    p.add_argument("--start-sec", type=float, default=0.0, help="First anchor time (default: 0)")
    p.add_argument(
        "--end-sec",
        type=float,
        default=None,
        help="Last anchor time upper bound (default: video duration)",
    )
    p.add_argument(
        "--max-anchors",
        type=int,
        default=None,
        help="Stop after this many anchors (smoke test)",
    )
    p.add_argument(
        "--model-id",
        type=str,
        default="nvidia/Alpamayo-1.5-10B",
        help="HuggingFace model id (default: nvidia/Alpamayo-1.5-10B)",
    )
    p.add_argument("--device", type=str, default="cuda", help="cuda or cpu (default: cuda)")
    p.add_argument(
        "--dtype",
        type=str,
        choices=("bfloat16", "float16", "float32"),
        default="bfloat16",
        help="Model dtype (default: bfloat16)",
    )
    p.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        help="Optional attn_implementation for from_pretrained (e.g. sdpa)",
    )
    p.add_argument("--top-p", type=float, default=0.98)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max-generation-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm")
    return p


def _dtype_from_str(name: str) -> torch.dtype:
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[name]


def _video_duration_sec(container: av.container.Container, stream: av.VideoStream) -> float | None:
    if stream.duration is not None and stream.time_base is not None:
        return float(stream.duration * stream.time_base)
    if container.duration is not None and container.duration > 0:
        # PyAV: duration in AV_TIME_BASE (microseconds)
        return float(container.duration) / 1_000_000.0
    return None


def _scan_max_frame_time(video_path: Path) -> float:
    """Decode entire stream to get the last frame presentation time (seconds)."""
    max_t = 0.0
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "NONE"
        for frame in container.decode(stream):
            if isinstance(frame, av.VideoFrame):
                max_t = max(max_t, frame_time_sec(frame, stream))
    finally:
        container.close()
    return max_t


def _build_anchors(args: argparse.Namespace) -> list[float]:
    """Compute anchor times from metadata and/or a full decode scan."""
    container = av.open(str(args.video))
    try:
        stream = container.streams.video[0]
        duration_sec = _video_duration_sec(container, stream)
    finally:
        container.close()

    end_bound = args.end_sec
    if end_bound is None:
        if duration_sec is not None:
            end_bound = duration_sec
        else:
            end_bound = _scan_max_frame_time(args.video) + 1e-3

    anchors = compute_anchor_times(
        duration_sec=end_bound,
        sample_every=args.sample_every_seconds,
        start_sec=args.start_sec,
        end_sec=end_bound,
    )
    if args.max_anchors is not None:
        anchors = anchors[: max(0, args.max_anchors)]
    return anchors


def _generate_vqa_answer(
    model: Alpamayo1_5,
    processor: Any,
    device: torch.device,
    dtype: torch.dtype,
    stacked_frames: torch.Tensor,
    question: str,
    *,
    top_p: float,
    temperature: float,
    max_generation_length: int,
) -> str:
    messages = helper.create_vqa_message(stacked_frames, question=question)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = helper.to_device({"tokenized_data": inputs}, device)
    if device.type == "cuda":
        with torch.autocast("cuda", dtype=dtype):
            extra = model.generate_text(
                data=model_inputs,
                top_p=top_p,
                temperature=temperature,
                num_samples=1,
                max_generation_length=max_generation_length,
            )
    else:
        extra = model.generate_text(
            data=model_inputs,
            top_p=top_p,
            temperature=temperature,
            num_samples=1,
            max_generation_length=max_generation_length,
        )
    return _first_answer(extra)


def run_inference(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    if not args.video.is_file():
        print(f"error: video not found: {args.video}", file=sys.stderr)
        return 1

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    anchors = _build_anchors(args)
    if not anchors:
        print("warning: no anchors to process", file=sys.stderr)
        args.output_jsonl.write_text("", encoding="utf-8")
        return 0

    dtype = _dtype_from_str(args.dtype)
    device = torch.device(args.device)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    fp_kwargs: dict[str, Any] = {"dtype": dtype}
    if args.attn_implementation:
        fp_kwargs["attn_implementation"] = args.attn_implementation

    model = Alpamayo1_5.from_pretrained(args.model_id, **fp_kwargs).to(device)
    model.eval()
    processor = helper.get_processor(model.tokenizer)

    buffer: list[tuple[float, torch.Tensor]] = []
    results: list[tuple[float, str]] = []

    container = av.open(str(args.video))
    try:
        in_stream = container.streams.video[0]
        in_stream.thread_type = "NONE"

        anchor_idx = 0
        decode_iter = container.decode(in_stream)
        if not args.no_progress:
            decode_iter = tqdm(decode_iter, desc="Decode", unit="pkt")

        for frame in decode_iter:
            if not isinstance(frame, av.VideoFrame):
                continue
            t = frame_time_sec(frame, in_stream)
            arr = frame.to_ndarray(format="rgb24")
            # (H,W,3) uint8 -> (N,3,H,W) matching dataset uint8 CHW
            tensor = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
            buffer.append((t, tensor))
            while len(buffer) > args.buffer_frames:
                buffer.pop(0)

            while anchor_idx < len(anchors) and anchors[anchor_idx] <= t + 1e-9:
                a = anchors[anchor_idx]
                anchor_idx += 1
                frames_list = select_window_frames(buffer, a, args.num_frames)
                if not frames_list:
                    answer = ""
                else:
                    stacked = torch.stack(frames_list, dim=0)
                    answer = _generate_vqa_answer(
                        model,
                        processor,
                        device,
                        dtype,
                        stacked,
                        args.question,
                        top_p=args.top_p,
                        temperature=args.temperature,
                        max_generation_length=args.max_generation_length,
                    )
                results.append((a, answer))

            if anchor_idx >= len(anchors):
                break

        while anchor_idx < len(anchors):
            a = anchors[anchor_idx]
            anchor_idx += 1
            frames_list = select_window_frames(buffer, a, args.num_frames)
            if not frames_list:
                answer = ""
            else:
                stacked = torch.stack(frames_list, dim=0)
                answer = _generate_vqa_answer(
                    model,
                    processor,
                    device,
                    dtype,
                    stacked,
                    args.question,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    max_generation_length=args.max_generation_length,
                )
            results.append((a, answer))

    finally:
        container.close()

    results.sort(key=lambda x: x[0])
    with args.output_jsonl.open("w", encoding="utf-8") as f:
        for anchor_sec, answer in results:
            line = json.dumps(
                {"anchor_sec": float(anchor_sec), "answer": answer},
                ensure_ascii=False,
            )
            f.write(line + "\n")

    return 0


def main(argv: list[str] | None = None) -> int:
    return run_inference(argv)


if __name__ == "__main__":
    raise SystemExit(main())
