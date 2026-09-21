# Realtime ASAP FF deployment

Use `realtime_planner_node_ff.py`. The copied legacy files remain unchanged.

## Files

- `realtime_planner_node_ff.py`: ROS entry point and FF input construction.
- `realtime_runtime_ff.py`: checkpoint auto-detection and model-call helpers.
- `realtime_planner_node.py`: copied legacy implementation used as ROS/topic infrastructure only.

The FF entry point fixes the semantic palette, provides calibrated K/T, supplies
both AD-MLP state and four-step ego motion, and keeps the existing ROS path/MPC
outputs.

## Model files

The default planner checkpoint location is:

```text
real_time_ff/model/last.ckpt
```

It may instead be passed with `_checkpoint:=/absolute/path/model.ckpt`.
The checkpoint is self-contained: its `model.admlp_baseline.*` tensors are
strict-loaded, so the separate AD-MLP training checkpoint is not needed on the
deployment computer.

The checkpoint TAG selects the architecture automatically:

| TAG | Runtime variant | Coarse source |
|---|---|---|
| `Planning_ASAP_ff` | `all_admlp` | all commands use AD-MLP |
| `Planning_ASAP_hybrid_ff` | `hybrid` | FORWARD extrapolates ego motion; LEFT/RIGHT use AD-MLP |

Use `_expected_model_variant:=all_admlp` or `hybrid` as an optional guard. The
default `auto` trusts the checkpoint TAG.

## Depth-Anything files

Portable defaults:

```text
real_time_ff/third_party/Depth-Anything-V2/
real_time_ff/model/depth_anything_v2_vitl.pth
```

Alternative locations can be supplied with `_da_v2_repo` and `_da_v2_ckpt`.
Keep `_use_depth:=true`: the FF checkpoints were trained with Depth-Anything-V2
relative inverse depth. Disabling it supplies zero depth and is a distribution
mismatch.

## Ego input modes

Real odometry for every ego input:

```text
_ego_input_mode:=real_odom
```

Fixed-speed straight motion for every model ego input:

```text
_ego_input_mode:=fixed_speed _fixed_speed_mps:=1.0
```

In fixed mode, `future_egomotion`, `ego_history_egomotion`, and `admlp_input`
are generated consistently. Real odometry is still used to transform and
publish the local trajectory in the global odom frame.

Backward-compatible `auto` mode is also available: legacy `_fixed_speed>0`
selects fixed mode; otherwise it selects real odometry. Explicit mode is
recommended for experiments.

## Launch examples

Source ROS as usual, then run with the Python environment containing the FF
model dependencies:

```bash
source /opt/ros/noetic/setup.bash
/home/cyc/miniconda3/envs/stp3_env/bin/python \
  /home/cyc/ST-P3_please/real_time_ff/realtime_planner_node_ff.py \
  _checkpoint:=/path/to/all_admlp_ff.ckpt \
  _expected_model_variant:=all_admlp \
  _ego_input_mode:=real_odom \
  _save_plots:=false
```

Hybrid with a synthetic 1 m/s straight ego history:

```bash
source /opt/ros/noetic/setup.bash
/home/cyc/miniconda3/envs/stp3_env/bin/python \
  /home/cyc/ST-P3_please/real_time_ff/realtime_planner_node_ff.py \
  _checkpoint:=/path/to/hybrid_ff.ckpt \
  _expected_model_variant:=hybrid \
  _ego_input_mode:=fixed_speed \
  _fixed_speed_mps:=1.0 \
  _save_plots:=false
```

Omit `_expected_model_variant` when automatic TAG detection is desired.

## Inputs and coordinate conventions

- Image cadence remains 0.5 seconds and the buffer waits for 3 images plus 5 odometry poses.
- Semantic RGB uses the same PALETTE4 as training; semantic IDs remain 0..3.
- K/T are copied from `park_L2_ASAP_ff.py` for the driver-rotated ZED2i and 1.8 m height.
- Route is absent in this first version, so route attention falls back to the selected coarse trajectory.
- No HD map or external drivable mask is required; safety uses image-projected road evidence.
- Internal planner lateral is +right and is flipped once to ROS/public +left before publishing.

Startup logs report checkpoint TAG, detected variant, coarse policy, ego input
mode, semantic palette, and camera calibration. Per-inference model latency is
also logged with CUDA synchronization.
