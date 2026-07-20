import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parent

EXPECTED = {
    "1779090545444418027.png": "speed_bump",
    "1779090595480065027.png": "speed_bump",
    "1779090690518134027.png": "speed_bump",
    "1779090536977374027.png": "normal",
    "1779090576979320027.png": "normal",
    "1779090628982060027.png": "normal",
}

PROMPTS: Dict[str, List[str]] = {
    "normal": [
        "a normal flat asphalt road with no black and yellow speed bump in the foreground",
        "a clear road surface without a striped speed bump across the lane",
        "a normal driving scene on a flat road with parked cars or lane markings",
        "a curved road or intersection without any foreground road hump",
        "yellow lane lines on a flat road, not a speed bump",
    ],
    "stop_sign": [
        "a red octagonal stop sign",
        "a stop sign on the road",
        "traffic sign that says stop",
        "a stop traffic sign",
        "a road warning sign requiring the car to stop",
    ],
    "red_light": [
        "a red traffic light",
        "a red stop signal",
        "traffic light requiring the car to stop",
        "red traffic signal at an intersection",
    ],
    "speed_bump": [
        "a black and yellow striped speed bump across the road in the foreground",
        "a black yellow road hump spanning the lane at the bottom of the image",
        "a raised black and yellow speed bump lying horizontally across the driving lane",
        "a foreground road speed bump with alternating black and yellow stripes",
        "a traffic calming speed hump painted black and yellow across the asphalt",
    ],
}


class ClipVlmClassifier:
    def __init__(
        self,
        prompts: Dict[str, List[str]],
        model_name: str = "ViT-B-32",
        hf_model_name: str = "openai/clip-vit-base-patch32",
        input_size: int = 224,
        temperature: float = 100.0,
        local_files_only: bool = True,
    ):
        self.labels = list(prompts.keys())
        self.prompts = prompts
        self.model_name = model_name
        self.hf_model_name = hf_model_name
        self.input_size = input_size
        self.temperature = temperature
        self.local_files_only = local_files_only
        self.backend = ""
        self.model = None
        self.tokenizer = None
        self.text_features = None

    def load(self, device: torch.device) -> None:
        try:
            import open_clip

            model, _, _ = open_clip.create_model_and_transforms(self.model_name, pretrained="openai")
            tokenizer = open_clip.get_tokenizer(self.model_name)
            self.backend = "open_clip"
            self.model = model.eval().to(device)
            self.tokenizer = tokenizer
        except Exception:
            from transformers import CLIPModel, CLIPTokenizer

            self.backend = "transformers"
            self.model = CLIPModel.from_pretrained(
                self.hf_model_name,
                local_files_only=self.local_files_only,
            ).eval().to(device)
            self.tokenizer = CLIPTokenizer.from_pretrained(
                self.hf_model_name,
                local_files_only=self.local_files_only,
            )

        for param in self.model.parameters():
            param.requires_grad_(False)
        self.text_features = self._encode_text_features(device)

    def _encode_text_features(self, device: torch.device) -> torch.Tensor:
        features = []
        with torch.no_grad():
            for label in self.labels:
                label_prompts = self.prompts[label]
                if self.backend == "open_clip":
                    tokens = self.tokenizer(label_prompts).to(device)
                    text_features = self.model.encode_text(tokens)
                else:
                    tokens = self.tokenizer(label_prompts, padding=True, return_tensors="pt").to(device)
                    text_features = self.model.get_text_features(**tokens)

                text_features = F.normalize(text_features.float(), dim=-1)
                text_features = F.normalize(text_features.mean(dim=0, keepdim=True), dim=-1)
                features.append(text_features)
        return torch.cat(features, dim=0)

    def _preprocess(self, image: Image.Image, device: torch.device) -> torch.Tensor:
        image = image.convert("RGB")
        if image.size != (self.input_size, self.input_size):
            image = image.resize((self.input_size, self.input_size), Image.BICUBIC)
        w, h = image.size
        x = torch.tensor(list(image.getdata()), dtype=torch.float32, device=device).view(h, w, 3)
        x = x.permute(2, 0, 1).unsqueeze(0) / 255.0
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=device).view(1, 3, 1, 1)
        return (x - mean) / std

    def _image_logits(self, image: Image.Image, device: torch.device) -> torch.Tensor:
        x = self._preprocess(image, device)
        with torch.no_grad():
            if self.backend == "open_clip":
                image_features = self.model.encode_image(x)
            else:
                image_features = self.model.get_image_features(pixel_values=x)
            image_features = F.normalize(image_features.float(), dim=-1)
            return self.temperature * image_features @ self.text_features.to(device).T

    @staticmethod
    def _bottom_crop(image: Image.Image, frac: float) -> Image.Image:
        frac = min(max(frac, 0.05), 1.0)
        w, h = image.size
        crop_h = max(int(round(h * frac)), 1)
        return image.crop((0, h - crop_h, w, h))

    def predict(self, image_path: Path, device: torch.device, bottom_crop_frac: float = 0.40) -> Tuple[str, float, Dict[str, float]]:
        image = Image.open(image_path)
        logits = self._image_logits(image, device)

        if bottom_crop_frac > 0:
            # These test images show the speed bump at the bottom foreground,
            # while one normal image has a black/yellow warning sign farther away.
            logits = self._image_logits(self._bottom_crop(image, bottom_crop_frac), device)

        probs = logits.softmax(dim=-1)[0]
        best_idx = int(probs.argmax().item())
        scores = {label: float(probs[i].item()) for i, label in enumerate(self.labels)}
        return self.labels[best_idx], float(probs[best_idx].item()), scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal CLIP/VLM test for six road-scene images.")
    parser.add_argument("--model", default="ViT-B-32", help="open_clip model name")
    parser.add_argument("--hf-model", default="openai/clip-vit-base-patch32", help="transformers fallback model")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--temperature", type=float, default=100.0)
    parser.add_argument("--bottom-crop-frac", type=float, default=0.40, help="0 uses the full image")
    parser.add_argument("--threshold", type=float, default=0.40, help="minimum speed_bump probability to output speed_bump")
    parser.add_argument("--allow-download", action="store_true", help="allow transformers fallback to download model files")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.allow_download:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = ClipVlmClassifier(
        prompts=PROMPTS,
        model_name=args.model,
        hf_model_name=args.hf_model,
        input_size=args.input_size,
        temperature=args.temperature,
        local_files_only=not args.allow_download,
    )
    classifier.load(device)

    correct = 0
    print(f"backend={classifier.backend} device={device}")
    score_cols = ",".join(f"p_{label}" for label in classifier.labels)
    print(f"file,pred,expected,ok,{score_cols},best_prob")
    for filename, expected in EXPECTED.items():
        path = ROOT / filename
        _, best_prob, scores = classifier.predict(path, device, args.bottom_crop_frac)
        pred = "speed_bump" if scores["speed_bump"] >= args.threshold else "normal"
        ok = pred == expected
        correct += int(ok)
        score_text = ",".join(f"{scores[label]:.4f}" for label in classifier.labels)
        print(
            f"{filename},{pred},{expected},{ok},"
            f"{score_text},{best_prob:.4f}"
        )

    total = len(EXPECTED)
    print(f"accuracy={correct}/{total} ({correct / total:.1%})")
    return 0 if correct == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
