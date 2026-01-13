import os
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline
from src.time_adaptive_freeu_schedule import DeltaFreeUSchedule
from src.time_adaptive_freeu_lunch_utils import (
    register_free_upblock2d,
    register_free_crossattn_upblock2d,
)
MODEL_ID = "Manojb/stable-diffusion-2-1-base"
DEVICE = "cuda"
DTYPE = torch.float16

SEED = 1011
PROMPT="a fluffy orange cat sitting on a wooden table."
STEPS = 25

# 固定 FreeU（SD2.1 常用）
FREEU_FIXED = dict(b1=1.4, b2=1.6, s1=0.9, s2=0.2)

# schedule ckpt（改成你实际保存的路径）
SCHEDULE_CKPT = "ckpt/ckpt_freeu_schedule_step90000.pt"

# schedule 超参
SCHED_K = 25
SCHED_MAX_PCT = 0.5
SCHED_BASE = dict(base_b1=1.4, base_b2=1.6, base_s1=0.9, base_s2=0.2)

#保存地址
OUT_PATH = "compare_freeu_90000_orange_cat.png"

# -----------------------------
def make_pipe():
    pipe = StableDiffusionPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        safety_checker=None,
    ).to(DEVICE)
    pipe.set_progress_bar_config(disable=True)
    return pipe

@torch.no_grad()
def gen(pipe, prompt, seed, steps):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    img = pipe(prompt, generator=g, num_inference_steps=steps).images[0]
    return img

def hstack(images):
    w, h = images[0].size
    canvas = Image.new("RGB", (w * len(images), h))
    for i, im in enumerate(images):
        canvas.paste(im, (i * w, 0))
    return canvas

def draw_title_bar(img, titles):
    from PIL import ImageDraw, ImageFont
    w, h = img.size
    bar_h = 50
    out = Image.new("RGB", (w, h + bar_h), (0, 0, 0))
    out.paste(img, (0, bar_h))

    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.load_default()
    except:
        font = None

    seg_w = w // len(titles)
    for i, t in enumerate(titles):
        x = i * seg_w + 10
        y = 15
        draw.text((x, y), t, fill=(255, 255, 255), font=font)

    return out

def main():
    # Baseline SD
    pipe_sd = make_pipe()
    img_sd = gen(pipe_sd, PROMPT, SEED, STEPS)

    # Fixed FreeU
    pipe_fixed = make_pipe()
    register_free_upblock2d(pipe_fixed, **FREEU_FIXED)
    register_free_crossattn_upblock2d(pipe_fixed, **FREEU_FIXED)
    img_fixed = gen(pipe_fixed, PROMPT, SEED, STEPS)

    # Trained schedule FreeU
    if not os.path.exists(SCHEDULE_CKPT):
        raise FileNotFoundError(f"Schedule ckpt not found: {SCHEDULE_CKPT}")

    pipe_dyn = make_pipe()
    schedule = DeltaFreeUSchedule(
        K=SCHED_K,
        max_pct=SCHED_MAX_PCT,
        **SCHED_BASE,
        T=1000, 
    ).to(DEVICE)
    sd = torch.load(SCHEDULE_CKPT, map_location=DEVICE)
    schedule.load_state_dict(sd, strict=True)

    register_free_upblock2d(pipe_dyn, schedule=schedule)             # 用 schedule
    register_free_crossattn_upblock2d(pipe_dyn, schedule=schedule)   # 用 schedule
    img_dyn = gen(pipe_dyn, PROMPT, SEED, STEPS)

    # 拼图保存
    comp = hstack([img_sd, img_fixed, img_dyn])
    comp = draw_title_bar(comp, ["SD", "FreeU fixed", "FreeU trained"])
    comp.save(OUT_PATH)
    print(f"Saved: {OUT_PATH}")
    print("Prompt:", PROMPT)
    print("Seed:", SEED, "Steps:", STEPS)

if __name__ == "__main__":
    main()