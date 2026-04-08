# Alpamayo Trajectory Input Spec

## Purpose
This document defines the data you need to generate so Alpamayo 1.5 can run trajectory prediction on non-native data sources such as dashcam video plus VSLAM ego poses.

This is a portable reference spec. It describes the effective input contract derived from the repository code, not a new training format defined by the repo.

## Scope
This spec covers:
- the raw data you should generate upstream
- the exact tensor shapes Alpamayo expects at inference time
- how to transform VSLAM poses into the required ego-history frame
- how navigation instructions should be represented

This spec does not cover:
- how to run VSLAM itself
- how to train or finetune Alpamayo
- how to create a canonical on-disk format used by the repo, because the repo does not define one for trajectory inference

## High-Level Contract
At inference time, the model is called with a Python `data` dict containing:

```python
{
    "tokenized_data": ...,      # built from images + prompt text via the processor
    "ego_history_xyz": ...,     # shape (B, 1, 16, 3) by default
    "ego_history_rot": ...,     # shape (B, 1, 16, 3, 3) by default
}
```

For a single sample, the simplest valid shapes are:

```python
ego_history_xyz.shape == (1, 1, 16, 3)
ego_history_rot.shape == (1, 1, 16, 3, 3)
```

The inference path enforces `n_traj_group == 1`, so the second dimension should be `1`.

## What You Must Generate Upstream
For each prediction anchor time `t0`, generate:

- camera frames ending at `t0`
- ego pose history ending at `t0`
- optional navigation text

The upstream record you generate should contain at least:

```python
{
    "anchor_time_sec": float,              # or anchor_time_us as int
    "image_frames": uint8 array,           # see shape below
    "camera_indices": int64 array | None,  # optional for single-camera dashcam
    "ego_history_xyz": float32 array,      # shape (1, 1, 16, 3)
    "ego_history_rot": float32 array,      # shape (1, 1, 16, 3, 3)
    "nav_text": str | None,                # optional, plain text instruction
}
```

Optional evaluation-only fields:

```python
{
    "ego_future_xyz": float32 array,       # shape (1, 1, 64, 3)
    "ego_future_rot": float32 array,       # shape (1, 1, 64, 3, 3)
    "relative_timestamps": float32 array,  # metadata
    "absolute_timestamps": int64 array,    # metadata
}
```

The future tensors are not required to run inference. They are only useful if you want to compare predictions against ground truth.

## Timing Requirements

### Ego history
Default history parameters in the repo are:
- `num_history_steps = 16`
- `time_step = 0.1` seconds

This means the history timestamps are:

```text
t0 - 1.5s
t0 - 1.4s
...
t0 - 0.1s
t0
```

You should resample or interpolate your VSLAM poses onto these exact times.

### Future horizon
Default future parameters in the repo are:
- `num_future_steps = 64`
- `time_step = 0.1` seconds

This corresponds to a 6.4 second future horizon. You only need this if you are generating labels for evaluation.

### Image frames
The reference loader uses:
- `num_frames = 4`
- one frame every `0.1` seconds ending at `t0`

So the default image times per camera are:

```text
t0 - 0.3s
t0 - 0.2s
t0 - 0.1s
t0
```

For a single dashcam stream, this is the easiest setup to reproduce.

## Image Tensor Format
Before prompt construction, the reference loader stores images as:

```python
image_frames.shape == (N_cameras, num_frames, 3, H, W)
dtype == uint8
```

For a single dashcam camera with 4 frames:

```python
image_frames.shape == (1, 4, 3, H, W)
```

When passed into `helper.create_message(...)`, the frames are flattened to:

```python
frames.shape == (N_total, 3, H, W)
```

where:

```text
N_total = N_cameras * num_frames
```

### Camera indices
`camera_indices` is optional for single-camera input.

If you do provide it, it should have shape:

```python
camera_indices.shape == (N_cameras,)
dtype == int64
```

Known camera IDs in the repo are:
- `0`: front left camera
- `1`: front camera
- `2`: front right camera
- `3`: rear left camera
- `4`: rear camera
- `5`: rear right camera
- `6`: front telephoto camera

For a single forward-facing dashcam, using `camera_indices=None` is acceptable in the current prompt-building code.

## Ego History Format

### Required shapes
For one sample:

```python
ego_history_xyz.shape == (1, 1, 16, 3)
ego_history_rot.shape == (1, 1, 16, 3, 3)
ego_history_xyz.dtype == float32
ego_history_rot.dtype == float32
```

For batched inference:

```python
ego_history_xyz.shape == (B, 1, 16, 3)
ego_history_rot.shape == (B, 1, 16, 3, 3)
```

### Coordinate frame
This is the most important requirement.

The model expects history poses in the local ego frame at `t0`, not in world coordinates.

Let:
- `p_i` be the 3D position of history sample `i` in your world or VSLAM frame
- `R_i` be the 3x3 rotation matrix of history sample `i` in your world or VSLAM frame
- `p_0` be the position at `t0`
- `R_0` be the rotation at `t0`

Then you must convert every history pose into the local frame of `t0` as:

```text
p_i_local = R_0^T (p_i - p_0)
R_i_local = R_0^T R_i
```

After this transform:
- the last history position must be approximately `[0, 0, 0]`
- the last history rotation must be approximately the identity matrix

This is the same normalization used by the reference loader.

### Rotation representation
Rotations must be provided as 3x3 matrices:

```python
ego_history_rot[..., i, :, :] = R_i_local
```

If your upstream system gives quaternions, convert them to 3x3 matrices first.

If your upstream system gives only yaw, you may construct a planar rotation matrix:

```python
R_yaw = [
    [cos(yaw), -sin(yaw), 0.0],
    [sin(yaw),  cos(yaw), 0.0],
    [0.0,       0.0,      1.0],
]
```

For dashcam plus monocular VSLAM, yaw-only may be a reasonable approximation if pitch and roll are noisy or not meaningful in the vehicle frame.

### Units
- positions must be in meters
- timestamps must be on a consistent time base
- the history cadence must be 10 Hz if you want to match the reference setup

If your VSLAM system is monocular and does not recover metric scale, you must solve scale before using the poses. The model was trained with metric motion history.

## Vehicle Frame vs Camera Frame
The model is conditioned on ego motion, meaning vehicle motion, not arbitrary camera motion.

If your VSLAM poses are in the camera frame, convert them into a vehicle-centered pose stream before constructing history:

```text
T_world_vehicle = T_world_camera * T_camera_vehicle
```

where `T_camera_vehicle` is the fixed extrinsic from the camera to the vehicle body frame.

If you do not know camera-to-vehicle extrinsics, your best fallback is to use a stable pseudo-ego frame aligned to the camera, but this is a mismatch relative to the training distribution and may degrade behavior.

## Navigation Input Format
Navigation is not a separate numeric tensor.

It is plain text inserted into the prompt between:

```text
<|route_start|> ... <|route_end|>
```

Examples:
- `Turn left in 11m`
- `Turn right in 30m`
- `Continue straight for 120m`
- `Keep right in 200m`

Recommended guidance:
- keep it short and imperative
- include maneuver and distance when available
- avoid paragraphs or richly formatted route descriptions

If you do not want navigation conditioning, set `nav_text=None`.

## How History Is Used Internally
The prompt contains 48 history placeholder tokens.

That matches:

```text
16 history steps * 3 coordinates per step = 48
```

Internally, the history tokenizer encodes per-step XYZ deltas into discrete tokens and fuses them into the prompt token stream.

Implications for your generator:
- history must be temporally ordered
- history must end exactly at `t0`
- history should be smooth and consistently sampled
- noisy or irregularly sampled VSLAM tracks should be resampled and smoothed before export

## Minimal Model-Ready Flow
Once you have generated the upstream record, the model-ready flow is:

1. Flatten `image_frames` from `(N_cameras, num_frames, 3, H, W)` to `(N_total, 3, H, W)`.
2. Build the prompt with `helper.create_message(...)`, passing frames, optional `camera_indices`, and optional `nav_text`.
3. Call the processor's `apply_chat_template(...)` to create `tokenized_data`.
4. Call trajectory inference with:

```python
{
    "tokenized_data": tokenized_data,
    "ego_history_xyz": ego_history_xyz,
    "ego_history_rot": ego_history_rot,
}
```

## Recommended Portable On-Disk Format
The repo does not define a required on-disk format for trajectory inference, so if you want to generate data elsewhere and hand it off later, use a simple portable bundle.

Recommended structure per anchor:

### Metadata JSON

```json
{
  "sample_id": "clip123_t0005123",
  "anchor_time_sec": 5.123,
  "history_hz": 10.0,
  "num_history_steps": 16,
  "num_image_frames": 4,
  "nav_text": "Turn left in 11m",
  "camera_indices": null,
  "tensor_file": "clip123_t0005123.npz"
}
```

### Tensor NPZ
Store these arrays in `clip123_t0005123.npz`:
- `image_frames`
- `ego_history_xyz`
- `ego_history_rot`
- optional `camera_indices`
- optional `ego_future_xyz`
- optional `ego_future_rot`
- optional `relative_timestamps`
- optional `absolute_timestamps`

This is a recommendation for portability only. Alpamayo itself does not load this format directly. You will still need a thin adapter that reads these files and builds the in-memory `data` dict.

## Validation Checklist
Before running the model, verify:

- `ego_history_xyz.shape == (1, 1, 16, 3)`
- `ego_history_rot.shape == (1, 1, 16, 3, 3)`
- `image_frames.shape == (1, 4, 3, H, W)` for the single-camera default case
- history timestamps are exactly `0.1` seconds apart
- the last history position is approximately zero
- the last history rotation is approximately identity
- positions are in meters
- video timestamps and VSLAM timestamps are synchronized to the same clock
- `nav_text` is either `None` or a short route instruction string

## Practical Notes For Dashcam Plus VSLAM
- Single-camera dashcam input is supported by the prompt-building code, even though the reference dataset uses multiple cameras.
- Monocular VSLAM is usable only if you recover or calibrate metric scale.
- If the road is approximately planar and vertical motion is unreliable, a consistent planar history can be better than noisy 3D motion.
- The diffusion action head predicts future `x` and `y` and keeps future `z` anchored to the last history `z`, so keeping `z` stable is reasonable if you do not trust vertical estimates.

## Derived From
This spec is derived from:
- `/root/alpamayo1.5/src/alpamayo1_5/load_physical_aiavdataset.py`
- `/root/alpamayo1.5/src/alpamayo1_5/helper.py`
- `/root/alpamayo1.5/src/alpamayo1_5/test_inference.py`
- `/root/alpamayo1.5/src/alpamayo1_5/models/alpamayo1_5.py`
- `/root/alpamayo1.5/src/alpamayo1_5/models/base_model.py`
- `/root/alpamayo1.5/src/alpamayo1_5/models/delta_tokenizer.py`
- `/root/alpamayo1.5/src/alpamayo1_5/action_space/unicycle_accel_curvature.py`
- `/root/alpamayo1.5/src/alpamayo1_5/nav_utils.py`
