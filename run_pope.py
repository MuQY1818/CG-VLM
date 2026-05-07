import argparse
import json
import os
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import set_seed

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.conscious_gaze.config import ConsciousGazeConfig  # noqa: E402
from core.conscious_gaze.cds import CognitiveStateTracker  # noqa: E402
from core.instructblip import InstructBlipForConditionalGeneration, InstructBlipProcessor  # noqa: E402


def load_model(model_path: str, device: str, cfg: ConsciousGazeConfig, tracker: CognitiveStateTracker):
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    model = InstructBlipForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype)
    processor = InstructBlipProcessor.from_pretrained(model_path)
    model.to(device)
    model.enable_conscious_gaze(cfg, tracker)
    return model, processor


def run(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    cfg = ConsciousGazeConfig(
        variance_threshold=args.variance_threshold,
        alpha=args.alpha,
        fci_layers=list(range(args.fci_start, args.fci_end)),
    )
    tracker = CognitiveStateTracker(cfg.entropy_window_size)
    model, processor = load_model(args.model_path, device, cfg, tracker)

    questions = [json.loads(q) for q in open(os.path.expanduser(args.question_file), "r")]
    answers_path = Path(args.answers_file)
    answers_path.parent.mkdir(parents=True, exist_ok=True)

    with answers_path.open("w", encoding="utf-8") as ans_file:
        for item in questions:
            qid = item.get("question_id", item.get("id", 0))
            image_file = item["image"]
            question = item["text"]

            image_path = os.path.join(args.image_folder, image_file)
            image = Image.open(image_path).convert("RGB")
            prompt = "<ImageHere>" + question
            inputs = processor(images=image, text=prompt, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
            inputs_just_q = processor(images=image, text="", return_tensors="pt")
            inputs_just_q = {k: v.to(device) for k, v in inputs_just_q.items()}
            inputs_just_q["pixel_values"] = inputs_just_q["pixel_values"].to(dtype)
            image_rand = torch.rand_like(inputs["pixel_values"], device=device, dtype=dtype)

            outputs = model.generate(
                **inputs,
                do_sample=True,
                num_beams=1,
                max_new_tokens=args.max_new_tokens,
                top_p=1.0,
                temperature=args.temperature,
                repetition_penalty=1.5,
                length_penalty=1.0,
                just_q=inputs_just_q,
                image_rand=image_rand,
                seed=args.seed,
            )
            text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
            ans_file.write(
                json.dumps(
                    {
                        "question_id": qid,
                        "prompt": question,
                        "text": text,
                        "model_id": "cg-vlm",
                        "image": image_file,
                        "metadata": {},
                    }
                )
                + "\n"
            )
            ans_file.flush()


def parse_args():
    parser = argparse.ArgumentParser(description="Run POPE with CG-VLM InstructBLIP")
    parser.add_argument("--model-path", required=True, help="path or HF repo for weights")
    parser.add_argument("--image-folder", required=True)
    parser.add_argument("--question-file", required=True)
    parser.add_argument("--answers-file", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-tokens", type=int, default=260)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--variance-threshold", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--fci-start", type=int, default=5)
    parser.add_argument("--fci-end", type=int, default=19)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    run(args)
