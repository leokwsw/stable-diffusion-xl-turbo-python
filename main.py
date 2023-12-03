import os
import torch

from diffusers import DiffusionPipeline
from dotenv import load_dotenv

load_dotenv()
CACHE_DIR = os.environ.get("CACHE_DIR", None)
mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
xpu_available = hasattr(torch, "xpu") and torch.xpu.is_available()
device = "cuda" if torch.cuda.is_available() else "xpu" if xpu_available else "cpu"

if mps_available:
    device = "mps"

pipe = DiffusionPipeline.from_pretrained(
    "stabilityai/sdxl-turbo",
    cache_dir=CACHE_DIR,
).to(device)

results = pipe(
    prompt="A cinematic shot of a baby racoon wearing an intricate italian priest robe",
    num_inference_steps=1,
    guidance_scale=0.0,
)
imga = results.images[0]
imga.save("image.png")
