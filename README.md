<div align="center">

# OneWM

### Can World and Action Modeling Benefit Each Other<br>under Unified Causal Training?

Yue Su*, Bokai Lin*, Xinkai Wang*, Chubin Zhang, Xiangnan Wu, Kuan Zhang,<br>
Weiyang Jin, Jiwen Yu, Yiming Li, Cewu Lu, Lixin Yang†, Xihui Liu†

MMLab@HKU · MVIG@SJTU · THU · SII · UCAS · IAAI Team<br>
<sub>* Equal contribution. † Corresponding authors.</sub>

**[Project Page](https://selen-suyue.github.io/OneWMNet/) · [Paper](assets/OneWM-preview.pdf) · [Demos](https://selen-suyue.github.io/OneWMNet/#experiments)**

**Training and deployment code for LIBERO, RoboCasa365 and dual Franka.**

[Original full-resolution PDF](assets/OneWM.pdf)

</div>

## Overview

**One causal model learns to predict the world and act in it.**

OneWM is a simple framework for unified causal training of world prediction and action generation. Within a shared video–action architecture, actions serve as both **conditions for predicting visual consequences** and **targets for policy learning**. We study whether these two learning objectives can benefit each other while keeping the architecture fixed.

Our experiments on **LIBERO, LIBERO-Plus, RoboCasa365, and real robots** support reciprocal benefits: learning action-conditioned forward dynamics improves policy performance and generalization, while action learning improves multi-view world-prediction fidelity.

We plan to integrate OneWM into the [OpenWAM](https://openwam-official.github.io/) framework soon, making it easier to train and deploy.

## How it works

<p align="center">
  <img src="assets/figures/intro.png" alt="Action-conditioning comparison: OneWM unifies all-clean, partial-clean, and all-noisy actions under causal attention." width="900">
</p>

OneWM connects world modeling and action generation by varying how much of the action sequence is known. A shared video–action architecture with **temporally causal attention** learns from three conditioning regimes:

- **All-clean:** specified actions condition future-video prediction.
- **Partial-clean:** a clean action prefix conditions future-video prediction and denoising of the remaining action suffix.
- **All-noisy:** the model learns to generate actions alongside future video.

At inference, the same model supports action generation from observations and instructions, or world prediction conditioned on specified actions.

## Install

Training and full-model inference require Linux, NVIDIA CUDA and a compatible
PyTorch installation. The commands below use Python 3.11 and CUDA 12.4 wheels;
adapt the PyTorch wheel to your CUDA driver when needed.

```bash
git clone https://github.com/Interactive-Active-AI/OneWM.git
cd OneWM
conda create -n onewm python=3.11 -y
conda activate onewm
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -e ".[train,deploy]"
python -m pip install huggingface_hub
hf download Wan-AI/Wan2.2-TI2V-5B --local-dir checkpoints/Wan2.2-TI2V-5B
```

The Wan directory must contain its DiT safetensor shards, model configuration,
`Wan2.2_VAE.pth`, `models_t5_umt5-xxl-enc-bf16.pth`, and `google/umt5-xxl` tokenizer.
Prepare the ActionDiT transformer initialization before training:

```bash
python -m onewm.tools.preprocess_action_dit_init \
  --config configs/libero.yaml --output checkpoints/action_dit_init.pt \
  --device cpu --dtype bfloat16
```

This interpolates the Wan transformer weights into the action expert. It is an
initialization payload, not a trained policy. Deployment loads the complete
trained checkpoint and does not require this initialization file.

## Prepare data and train

Follow [Data preparation](docs/datasets.md) for the dataset layout, text-cache
precomputation, RoboCasa365 downloads, and Franka split/statistics preparation.
The YAML files use portable relative paths under `data/`, `checkpoints/`, and
`outputs/`. Execute commands from the repository root.

```bash
# Choose one: libero, robocasa365, franka.
python scripts/train.py --config configs/libero.yaml --num-processes 8 --dry-run
python scripts/train.py --config configs/libero.yaml --num-processes 8
```

The launcher calculates gradient accumulation so
`micro_batch × GPU_count × accumulation = expected_global_batch_size`.
It rejects nondivisible combinations. For example, LIBERO uses 16×8×1=128;
RoboCasa365 uses 4×8×4=128; Franka uses 12×8×1=96. Reduce per-GPU batch size when
necessary and the launcher will recalculate accumulation:

```bash
python scripts/train.py --config configs/robocasa365.yaml --num-processes 4 \
  --override training.batch_size=2 training.output_dir=outputs/robocasa365
```

Defaults train the video and action experts jointly, freeze VAE/T5, and save
checkpoints under `training.output_dir`. LIBERO uses 20 epochs (a maximum of
42,320 updates), RoboCasa365 uses a maximum of 120,000 updates, and Franka uses
30 epochs. Change the budget explicitly through `training.max_steps` or
`training.num_epochs` for your dataset. Resume an intact training checkpoint with:

```bash
python scripts/train.py --config configs/libero.yaml --num-processes 8 \
  --override training.resume=outputs/libero/checkpoint-2000
```

Keep the matching recipe, normalization statistics and text-cache convention
with every trained model. W&B is disabled in all defaults and has no account,
entity or run identifiers embedded in the source. Optional logging can be
installed with `python -m pip install -e ".[logging]"` and enabled in a local
recipe. The logger records that run's configuration when enabled.

## Deploy WM and WAM

Both paths load the same joint checkpoint:

```bash
# WAM: observation + instruction -> normalized action chunk.
python -m onewm.tools.infer --mode wam --config configs/libero.yaml \
  --checkpoint checkpoints/libero/your_checkpoint --input observation.npz \
  --output outputs/policy.npz

# WM: observation + instruction + clean actions -> predicted video.
python -m onewm.tools.infer --mode wm --config configs/libero.yaml \
  --checkpoint checkpoints/libero/your_checkpoint --input world_request.npz \
  --output outputs/world.npz
```

[Deployment](docs/deployment.md) explains how to construct these NPZ requests,
start LIBERO/RoboCasa365 policy servers and simulator clients, and use the
Franka adapter. Checkpoint paths above are examples; supply your own trained
weights and their matching configuration. The low-level NPZ interface uses
normalized model inputs; benchmark adapters handle raw observation conversion
and return denormalized actions.

## Real-world manipulation

| Block stacking | Block insertion | Breakfast arrangement |
| :---: | :---: | :---: |
| [![Block stacking](assets/demos/stacking.jpg)](assets/demos/stacking.mp4) | [![Block insertion](assets/demos/storage.jpg)](assets/demos/storage.mp4) | [![Breakfast arrangement](assets/demos/breakfast.jpg)](assets/demos/breakfast.mp4) |

Click a preview to open its silent video, or [watch all three on the project page](https://selen-suyue.github.io/OneWMNet/#experiments).

## An Endless Dream

> **After the first observation, OneWM sees only what it imagines.**

Starting from an initial pair of camera views and a task instruction, OneWM **predicts an action chunk**, holds those predicted actions fixed to **imagine the next video**, then feeds the final generated views back into its next action prediction. The same model repeats this loop with **no further environment observations** and no additional training.

<p align="center">
  <a href="https://selen-suyue.github.io/OneWMNet/#dreaming"><img src="assets/figures/endless-dream.png" alt="Six LIBERO imagined rollouts, with initial main and wrist observations followed by generated future frames." width="900"></a>
</p>

The shown LIBERO rollouts span **16 rounds, 512 predicted actions, and 25.6 seconds of modeled time**. They demonstrate task-directed progression in imagination; the generated action sequences are not executed in the simulator in this demonstration.

[Explore the interactive action-and-video loop](https://selen-suyue.github.io/OneWMNet/#dreaming) · [Watch a sample rollout](assets/demos/endless-dream.mp4)

Run the same bounded imagination procedure with any compatible checkpoint:

```bash
python -m deploy.libero.endless_dream --config configs/libero.yaml \
  --checkpoint checkpoints/libero/your_checkpoint --input observation.npz \
  --output outputs/endless-dream --rounds 16
```

Inputs and output fields are documented in [Deployment](docs/deployment.md#an-endless-dream-libero).

## License and acknowledgments

OneWM code is released under [Apache-2.0](LICENSE). Upstream attributions and
retained license texts are listed in [NOTICE](NOTICE). We acknowledge
[Wan2.2](https://github.com/Wan-Video/Wan2.2),
[Fast-WAM](https://github.com/yuantianyuan01/FastWAM),
[LIBERO](https://github.com/Lifelong-Robot-Learning/LIBERO), and
[RoboCasa365](https://github.com/robocasa/robocasa).


## Citation

If you find this work useful, please cite:

```bibtex
@misc{su2026onewm,
  title  = {Can World and Action Modeling Benefit Each Other under Unified Causal Training?},
  author = {Yue Su and Bokai Lin and Xinkai Wang and Chubin Zhang and Xiangnan Wu and Kuan Zhang and Weiyang Jin and Jiwen Yu and Yiming Li and Cewu Lu and Lixin Yang and Xihui Liu},
  year   = {2026},
  url    = {https://github.com/Interactive-Active-AI/OneWM}
}
```
