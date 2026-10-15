# Deployment

One trained OneWM checkpoint supports both WAM action generation and WM
prediction conditioned on known actions. Use the same model geometry,
normalization statistics, tokenizer and text-cache settings as training.
Checkpoint paths are always supplied by the caller. A file (`model.pt`,
`pytorch_model.bin` or safetensors) or a training checkpoint directory containing
`pytorch_model/mp_rank_00_model_states.pt` is accepted. Loading checks all model
keys strictly. Franka loading preserves checkpoint storage with `assign=True`.

## Model-ready NPZ interface

`python -m onewm.tools.infer` accepts an NPZ request loaded without pickle:

| Field | Shape and meaning |
| --- | --- |
| `image` | uint8 `[H,W,3]`, already composed in the training camera layout |
| `input_video` | Alternative to `image`: float `[3,T,H,W]` in `[-1,1]`, clean history only |
| `instruction` | Scalar string; encoded with the recipe's prompt template/T5 cache |
| `context`, `context_mask` | Alternative to `instruction`: precomputed `[L,4096]` and Boolean `[L]` |
| `proprio` | Optional **normalized** state `[N,D]` for a state-conditioned model |
| `actions_normalized` | WM input only: full normalized `[horizon,action_dim]` in model order |

Use the benchmark adapter to compose raw camera views, reorder actions and
normalize state before using this low-level interface. It intentionally consumes
model-ready tensors. RoboCasa365 requires five clean raw history frames and
four state tokens for WAM; use its online adapter for raw simulator observations.
Franka WAM requires one 16D state token. The default LIBERO recipe is state-free.

For example, create a LIBERO observation request from two RGB images:

```python
import numpy as np
from PIL import Image
from deploy.libero.model_adapter import compose_policy_image

primary = np.asarray(Image.open('primary.png').convert('RGB'))
wrist = np.asarray(Image.open('wrist.png').convert('RGB'))
image = compose_policy_image(primary, wrist, view_size=(224, 224), concat_mode='horizontal')
rgb = image[0].add(1).mul(127.5).round().clamp(0, 255).byte().permute(1, 2, 0).numpy()
np.savez_compressed('observation.npz', image=rgb, instruction='pick up the bowl')
```

Generate a normalized action chunk:

```bash
python -m onewm.tools.infer --mode wam --config configs/libero.yaml \
  --checkpoint checkpoints/libero/your_checkpoint --input observation.npz \
  --output outputs/policy.npz
```

To predict its visual consequences, add `actions_normalized` from that output to
the observation NPZ and call the WM path:

```python
with np.load('observation.npz', allow_pickle=False) as obs, np.load('outputs/policy.npz') as action:
    np.savez_compressed('world_request.npz', **dict(obs), actions_normalized=action['actions_normalized'])
```

```bash
python -m onewm.tools.infer --mode wm --config configs/libero.yaml \
  --checkpoint checkpoints/libero/your_checkpoint --input world_request.npz \
  --output outputs/world.npz
```

WM output includes uint8 THWC `frames` and the unchanged normalized action
condition. WAM output is normalized; benchmark policy wrappers return
**denormalized** actions and handle their environment conventions.

## LIBERO online policy

In the model environment:

```bash
python -m deploy.libero.policy_server --config configs/libero.yaml \
  --checkpoint checkpoints/libero/your_checkpoint --host 127.0.0.1 --port 10093
```

In the separate LIBERO environment, from the repository root:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python -m deploy.libero.eval run \
  --benchmark standard --libero-root third_party/LIBERO --suite libero_spatial \
  --host 127.0.0.1 --port 10093 --num-trials 10 --output-dir outputs/libero-eval
```

The client preserves the dataset-to-LIBERO gripper conversion. It obtains camera
views and reobserves after the chosen replan interval. Rendering uses EGL; an
unavailable EGL stack must be fixed in the simulator environment.

## RoboCasa365 online policy

In the model environment:

```bash
python -m deploy.robocasa365.policy_server --config configs/robocasa365.yaml \
  --checkpoint checkpoints/robocasa365/your_checkpoint --port 8765 \
  --override data.text_embedding_cache_dir=data/robocasa365/assets/eval_text_cache
```

In the simulator environment prepared in [datasets.md](datasets.md):

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl conda run --no-capture-output -n onewm-robocasa365 \
  python -m deploy.robocasa365.eval_client --server ws://127.0.0.1:8765 \
  --rollouts-per-task 50 --split pretrain --output-dir outputs/robocasa365-eval
```

The server maintains observation history separately per rollout, returns eight
executed actions from each predicted 16-action chunk, and clears history on
reset. The client evaluates the official target50 task registry. `--split`
selects the kitchen split; it does not add target data to training. Training and
evaluation instruction caches use separate directories.

## Franka offline adapter

```python
from deploy.franka.policy_adapter import FrankaPolicy

policy = FrankaPolicy('configs/franka.yaml', 'checkpoints/franka/your_checkpoint')
targets = policy.predict(
    {'middle_zed': middle_rgb, 'left_camera': left_rgb, 'right_camera': right_rgb},
    absolute_state_16d, instruction='stack the blocks',
)
# targets: [32,16] absolute targets in the same joint/gripper order as the state.
```

The adapter returns predictions and does not connect to a robot controller.
A hardware client supplies synchronization, units, control rate, target limits
and the lab's supervised execution procedure.

## An Endless Dream (LIBERO)

```bash
python -m deploy.libero.endless_dream --config configs/libero.yaml \
  --checkpoint checkpoints/libero/your_checkpoint --input observation.npz \
  --output outputs/endless-dream --rounds 16 --steps 10 --seed 42
```

The loop predicts an action chunk, holds it clean during world prediction, then
feeds the final generated RGB frame into the next round. Feedback uses lossless
uint8 arrays; MP4 encoding is only for viewing. There are no later environment
observations or ground-truth actions. The output contains `rollout.npz`, per-round
NPZ files, `rollout.mp4`, and a short summary. Predicted actions are not executed
in the simulator.

Weights, task text, initial image and geometry are supplied by the caller. The
Python `imagine` function also accepts any model implementing the compatible
`infer_action` / `infer_world` interfaces; it has no checkpoint or episode IDs
embedded in the implementation. The command-line loader builds OneWM models
from the supplied recipe. For a state-conditioned model the optional initial
state stays fixed: visual feedback does not estimate new proprioception.
