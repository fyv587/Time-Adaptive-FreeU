import torch
import torch.nn.functional as F
from diffusers import StableDiffusionPipeline, DDPMScheduler
from src.time_adaptive_freeu_schedule import DeltaFreeUSchedule
from src.time_adaptive_freeu_lunch_utils import (
    register_free_upblock2d,
    register_free_crossattn_upblock2d,
)
from torch.utils.data import DataLoader
from coco_hf_dataset import CocoHFParquet
import torch.fft as fft
import os, datetime
import csv
device = "cuda"

def dump_bs_csv_append(path, train_step, infer_ts, schedule):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(["train_step","kk","t","bin","b1","b2","s1","s2"])

        with torch.no_grad():
            for kk in range(len(infer_ts)):
                ttest = infer_ts[kk].view(1).to(next(schedule.parameters()).device)
                b1,b2,s1,s2 = schedule(ttest)
                idx = int(schedule._idx_from_t(ttest)[0].item())
                w.writerow([train_step, kk, int(ttest.item()), idx,
                            float(b1.item()), float(b2.item()), float(s1.item()), float(s2.item())])

def hf_lf_ratio(z, thresh=8, eps=1e-8):
    zf = fft.fftshift(fft.fftn(z.float(), dim=(-2,-1)), dim=(-2,-1))
    mag2 = (zf.real**2 + zf.imag**2)  

    B,C,H,W = mag2.shape
    crow, ccol = H//2, W//2

    lf = mag2[..., crow-thresh:crow+thresh, ccol-thresh:ccol+thresh].mean(dim=(-1,-2))
    hf = mag2.mean(dim=(-1,-2)) - lf

    ratio = hf / (lf + eps)  
    return ratio.mean()    

def highpass(z):
    k = torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]], device=z.device, dtype=z.dtype).view(1,1,3,3)
    k = k.repeat(z.shape[1], 1, 1, 1) 
    return F.conv2d(z, k, padding=1, groups=z.shape[1])

coco_local = "data/coco2017_hf"
dataset = CocoHFParquet(coco_local, split="train", size=512)
loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=8,    
    pin_memory=True,
    drop_last=True,
    persistent_workers=True,
    prefetch_factor=4,
)

it = iter(loader)

# 加载 SD 
pipe = StableDiffusionPipeline.from_pretrained(
    "Manojb/stable-diffusion-2-1-base",
    torch_dtype=torch.float16,
    safety_checker=None,
).to(device)

pipe.unet.train()
pipe.vae.eval()
pipe.text_encoder.eval()

# 冻结原模型
for m in [pipe.unet, pipe.vae, pipe.text_encoder]:
    for p in m.parameters():
        p.requires_grad_(False)

K = 25  # 设置 num_inference_steps
pipe.scheduler.set_timesteps(K, device=device)
infer_ts = pipe.scheduler.timesteps.to(device) 
infer_ts = infer_ts.long()
print("infer timesteps:", infer_ts[:5], "...", infer_ts[-5:])

#注册 FreeU
schedule = DeltaFreeUSchedule(
    K=25,
    max_pct=0.5,          # 微调幅度50%
    base_b1=1.4, base_b2=1.6,
    base_s1=0.9, base_s2=0.2
).to(device)
register_free_upblock2d(pipe, schedule=schedule)
register_free_crossattn_upblock2d(pipe, schedule=schedule)

# ========== 3. Optimizer ==========
optimizer = torch.optim.AdamW(schedule.parameters(), lr=1e-3)
scheduler = DDPMScheduler.from_config(pipe.scheduler.config)
alphas_cumprod = scheduler.alphas_cumprod.to(device)

import datetime

run_name = datetime.datetime.now().strftime("freeu_%Y%m%d_%H%M%S")
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, f"{run_name}.log")
BS_CSV_PATH = os.path.join(LOG_DIR, "bs_at_infer25_track.csv")
CKPT_DIR = "ckpt"
os.makedirs(CKPT_DIR, exist_ok=True)
log_f = open(log_path, "a", buffering=1)  

def log(msg: str):
    print(msg)
    log_f.write(msg + "\n")

# ========== 5. 训练循环 ==========
# ========== 5. 训练循环 ==========
for step in range(100000):
    try:
        images, captions = next(it)
    except StopIteration:
        it = iter(loader)
        images, captions = next(it)

    images = images.to(device, dtype=torch.float16, non_blocking=True)
    if isinstance(captions, tuple):
        captions = list(captions)

    if step == 0:
        print("caption example:", captions[0])
        print("image min/max:", images.min().item(), images.max().item())

    with torch.no_grad():
        latents = pipe.vae.encode(images).latent_dist.sample()
        latents = latents * pipe.vae.config.scaling_factor

        tokens = pipe.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=pipe.tokenizer.model_max_length,
            return_tensors="pt",
        ).to(device)

        text_emb = pipe.text_encoder(tokens.input_ids)[0]

    B = latents.shape[0]
    t = torch.randint(0, 1000, (B,), device=device, dtype=torch.long)
    noise = torch.randn_like(latents)
    x_t = scheduler.add_noise(latents, noise, t)

    for blk in pipe.unet.up_blocks:
        blk._cur_t = t

    eps_pred = pipe.unet(x_t, t, encoder_hidden_states=text_emb).sample

    loss_eps = F.mse_loss(eps_pred, noise)

    a = alphas_cumprod[t].view(B,1,1,1).to(device=device, dtype=eps_pred.dtype)
    x0_pred = (x_t - torch.sqrt(1 - a) * eps_pred) / (torch.sqrt(a) + 1e-8)

    loss_x0 = F.mse_loss(x0_pred, latents)
    loss_hf = F.l1_loss(highpass(x0_pred), highpass(latents))

    # 低频一致性
    lp_pred = F.avg_pool2d(x0_pred, 9, 1, 4)
    lp_gt   = F.avg_pool2d(latents, 9, 1, 4)
    loss_lp = F.l1_loss(lp_pred, lp_gt)

    # 频谱能量比
    ratio_pred = hf_lf_ratio(x0_pred, thresh=8)
    ratio_gt   = hf_lf_ratio(latents, thresh=8)
    loss_ratio = (ratio_pred - ratio_gt).abs()


    loss = (
    + 0.1 * loss_hf
    + loss_lp
    + loss_ratio
)
    
    if step % 100 == 0:
        log(
            f"step {step} | total {float(loss):.4f} | "
            f"eps {float(loss_eps):.4f} | hf {float(loss_hf):.4f} | "
            f"lp {float(loss_lp):.4f} | ratio {float(loss_ratio):.4f}"
        )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        log(f"step {step} loss {float(loss):.6f}")
    
        with torch.no_grad():
            for kk in [0, K//2, K-1]:
                ttest = infer_ts[kk].view(1)
                b1,b2,s1,s2 = schedule(ttest)
                idx = int(schedule._idx_from_t(ttest)[0])
                log(f"kk {kk} t {int(ttest.item())} -> bin {idx} b1 {float(b1):.4f} b2 {float(b2):.4f} s1 {float(s1):.4f} s2 {float(s2):.4f}")
    # 每 N 步保存一次
    if step % 1000 == 0 and step > 0:
        torch.save(
            schedule.state_dict(),
            os.path.join(CKPT_DIR, f"ckpt_freeu_schedule_step{step}.pt")
        )
        dump_bs_csv_append(BS_CSV_PATH, step, infer_ts, schedule)

log_f.close()