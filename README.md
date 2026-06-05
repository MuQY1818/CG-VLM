# Conscious Gaze for Vision-Language Models

This repository provides a reference implementation of **Conscious Gaze (CG-VLM)**, a decoding-time method for mitigating hallucination in vision-language models.

The released code uses **InstructBLIP** as the example backbone. The CG-VLM logic is implemented in `core/conscious_gaze/` and integrated into a lightly patched InstructBLIP implementation under `core/instructblip/`. Other VLM backbones are not included in this release, but the method is organized so that the CDS and FCI components can be adapted to other architectures.

## What is included

- Conscious Gaze modules:
  - **CDS (Cognitive Demand Sensing)**: estimates whether the current decoding step needs stronger visual grounding.
  - **FCI (Focused Consensus Induction)**: boosts attention toward visual tokens when CDS enters focus mode.
- An InstructBLIP-based reference integration.
- POPE inference and evaluation scripts.
- Minimal runtime dependencies.

## Repository structure

```text
core/
  conscious_gaze/   CG-VLM configuration, CDS, FCI, and attention helpers
  instructblip/     InstructBLIP reference backend with CG-VLM integration
  sampling/         Shapley-style contrastive decoding utilities
run_pope.py         Run POPE inference
eval_pope.py        Compute POPE metrics from generated answers
requirements.txt    Python dependencies
```

## Environment setup

Create and activate a dedicated environment before installing dependencies:

```bash
conda create -n cg-vlm python=3.10
conda activate cg-vlm
pip install -r requirements.txt
```

Do not install dependencies into the base conda environment.

## Model weights

Provide your own InstructBLIP-compatible weights through `--model-path`.

The current reference path is designed around InstructBLIP with a decoder-only language model, such as a Vicuna-based InstructBLIP checkpoint. The code does not include pretrained weights.

## Run POPE with the InstructBLIP example

```bash
python run_pope.py \
  --model-path /path/to/instructblip_or_cgvlm_weights \
  --image-folder /path/to/coco/val2014 \
  --question-file data/coco/coco_pope_random.json \
  --answers-file outputs/pope_answer.jsonl \
  --seed 42 \
  --max-new-tokens 260 \
  --variance-threshold 1.0 \
  --alpha 0.5
```

Evaluate the generated answers:

```bash
python eval_pope.py \
  --gt_files data/coco/coco_pope_random.json \
  --gen_files outputs/pope_answer.jsonl
```

## Single-image inference example

```python
import torch
from PIL import Image

from core.conscious_gaze.cds import CognitiveStateTracker
from core.conscious_gaze.config import ConsciousGazeConfig
from core.instructblip import InstructBlipForConditionalGeneration, InstructBlipProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

cfg = ConsciousGazeConfig()
tracker = CognitiveStateTracker(cfg.entropy_window_size)

model = InstructBlipForConditionalGeneration.from_pretrained(
    "/path/to/instructblip_or_cgvlm_weights",
    torch_dtype=dtype,
)
processor = InstructBlipProcessor.from_pretrained("/path/to/instructblip_or_cgvlm_weights")

model.enable_conscious_gaze(cfg, tracker)
model.eval().to(device)

image = Image.open("your_image.png").convert("RGB")
prompt = "<ImageHere>describe the image."

inputs = processor(images=image, text=prompt, return_tensors="pt")
inputs = {key: value.to(device) for key, value in inputs.items()}
inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

inputs_just_q = processor(images=image, text="", return_tensors="pt")
inputs_just_q = {key: value.to(device) for key, value in inputs_just_q.items()}
inputs_just_q["pixel_values"] = inputs_just_q["pixel_values"].to(dtype)

image_rand = torch.rand_like(inputs["pixel_values"], device=device, dtype=dtype)

outputs = model.generate(
    **inputs,
    just_q=inputs_just_q,
    image_rand=image_rand,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
)

text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
print(text)
```

## Method overview

At each decoding step, the InstructBLIP example builds four variants of the multimodal input:

- full image and text input
- image-masked input
- text-masked input
- image-and-text-masked input

CDS compares the next-token logits from these variants and estimates the interaction variance between visual and textual evidence. If the variance exceeds `variance_threshold`, CDS sets the focus flag for that step.

When the focus flag is active, FCI increases the pre-softmax attention scores from the current decoding token to the visual token range in selected language-model layers. The boost strength is controlled by `alpha`.

## Key arguments

- `--variance-threshold`: threshold for CDS focus-mode activation.
- `--alpha`: visual attention boost strength used by FCI.
- `--fci-start` and `--fci-end`: half-open layer range for applying FCI.
- `--max-new-tokens`: maximum number of generated tokens.
- `--temperature`: sampling temperature.
