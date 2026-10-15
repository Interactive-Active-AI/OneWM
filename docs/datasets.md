# Data preparation

Run all commands from the repository root. Change the relative paths in the YAML
recipes or use `--override key=value` when your data lives elsewhere. Data, model
weights, text caches and generated files are ignored by Git.

## LIBERO

The default recipe reads the four no-noop **LeRobot v2.1** datasets, with their
original numeric `action` and `observation.state` fields:

```text
data/libero/
  libero_spatial_no_noops_lerobot/
  libero_object_no_noops_lerobot/
  libero_goal_no_noops_lerobot/
  libero_10_no_noops_lerobot/
```

Obtain the LeRobot v2.1 conversion from your dataset provider. A public conversion
with this schema is available from [IPEC-COMMUNITY](https://huggingface.co/datasets/IPEC-COMMUNITY/libero_object_no_noops_lerobot).
Use the same conversion/version throughout training and evaluation; independently
converted datasets can contain different episode sets. These files are not
bundled with OneWM. Each root must contain `meta/info.json`, `meta/tasks.jsonl`,
`meta/episodes.jsonl`, `data/chunk-*/episode_*.parquet`, and matching
`videos/chunk-*/observation.images.{image,wrist_image}/episode_*.mp4`.
The original LIBERO HDF5 files are not directly accepted by this loader.

Each camera is resized to 224×224 and the two views are concatenated horizontally
into 224×448. A 32-action chunk aligns with nine frames at offsets
`[0,4,8,...,32]`. Actions use per-channel min/max normalization. The default
LIBERO model uses images and instruction without proprioceptive state.
Missing LIBERO statistics are computed from the configured roots at training
startup and written to `data/libero/assets/stats.json`; preserve that file for
inference with the resulting checkpoint.

```bash
python -m onewm.tools.precompute_text_cache --config configs/libero.yaml --device cuda:0
```

For simulator evaluation, install [LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO)
in a separate Python environment with its required robosuite/MuJoCo versions.
Place the checkout at `third_party/LIBERO` or pass `--libero-root` explicitly.
Install `numpy`, `msgpack`, `websockets>=12,<16`, `imageio` and `av` in that client
environment. The simulator client does not need OneWM weights or a CUDA PyTorch
installation. Run it from this repository so the `deploy` module is available.

## RoboCasa365

Use the official **Human300 pretraining** LeRobot data. The downloader selects
human demonstrations from the pinned upstream registry; the 50 target tasks and
synthetic demonstrations are excluded from training and normalization.

```bash
bash scripts/setup_robocasa365.sh
python deploy/robocasa365/scripts/download_human300.py --manifest-only
python deploy/robocasa365/scripts/download_human300.py --workers 3
python -m onewm.tools.robocasa365_data audit --root data/robocasa365/pretrain
python -m onewm.tools.robocasa365_data stats \
  --root data/robocasa365/pretrain --output data/robocasa365/assets/stats.json
python -m onewm.tools.precompute_text_cache --config configs/robocasa365.yaml --device cuda:0
```

The setup script requires Conda, installs a separate `onewm-robocasa365`
simulator environment and checks out RoboCasa and robosuite at the revisions used
by the data adapter. It leaves the current training environment alone.
The downloader requires `curl` and writes resumable archives under the ignored
data directory. Use `--destination-root` to select another location and update
`data.root` in the recipe accordingly.

The expected hierarchy is
`pretrain/{atomic|composite}/{task}/{date}/lerobot/` with exactly 300 task roots.
Normalization is computed from the training episode split in the transformed
coordinates. Preserve the resulting statistics with the checkpoint.

The three 256×256 views are ordered **left, wrist, right** with 16-pixel RGB-128
separators (256×800 total). Four historical observations at `[-6,-4,-2,0]` plus a
duplicated current frame form the clean prefix. Eight future frames use offsets
`[2,4,...,16]`. The model predicts 16 actions in EE-first 12D order and uses four
14D state-history tokens. The two binary channels are dimensions 6 and 11;
normalization keeps those channels unchanged.

Simulation also requires the upstream kitchen assets (a separate large download):

```bash
conda run -n onewm-robocasa365 python \
  deploy/robocasa365/scripts/download_kitchen_assets_resumable.py
```

## Dual Franka

The Franka loader accepts packed **LeRobot v3** recordings under
`data/franka/{prepare_breakfast,put_block_in_drawer,stack_blocks}`. These robot
recordings are supplied by the user; no recording or checkpoint is bundled.
Required numeric fields are `action`, `observation.state`, `frame_index`,
`episode_index`, `task_index` and `timestamp`. Metadata uses `meta/tasks.parquet`
and `meta/episodes/chunk-*/file-*.parquet`, with packed data/video offsets in the
episode records. The preparation tool also checks `meta/episode_provenance.json`
for duplicate source recordings.

```bash
python -m onewm.tools.prepare_franka --root data/franka --output data/franka/assets
python -m onewm.tools.precompute_text_cache --config configs/franka.yaml --device cuda:0
```

Preparation creates a purged train/validation window split, q01/q99 statistics,
and a local source inventory. The absolute 16D state and target order is left
seven joints, left gripper, right seven joints, right gripper. Recorded targets
must match the next state. All three synchronized RGB views are required:
`middle_zed`, `left_camera`, `right_camera`. Each view is letterboxed to 224×224
and concatenated to a 224×672 strip. The model predicts 32 target timesteps;
gripper dimensions 7 and 15 remain continuous. The recorded 30 Hz timestamp grid
is a dataset convention; robot execution timing belongs to the hardware client.
