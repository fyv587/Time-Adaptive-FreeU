# Time-Adaptive FreeU

本项目为课程大作业，复现并改进了 **FreeU（Free Lunch in Diffusion U-Net）** 方法。

在原始 FreeU 使用固定的 **(b1, b2, s1, s2)** 超参数的基础上，本项目提出 **Time-Adaptive FreeU**：  
将 FreeU 的调节参数设计为随 diffusion timestep 变化的可学习 schedule，在不训练原始 Stable Diffusion 模型的前提下，通过少量参数学习实现更好的结构–纹理权衡。

---

## 一、方法简介

FreeU 通过在 U-Net 上采样阶段对：

- backbone 特征（b1, b2）
- skip connection 特征（s1, s2）

进行缩放，从而改善生成质量。

本项目的改进点是：

- 将固定的 (b1, b2, s1, s2) 扩展为 timestep-dependent schedule
- 使用离散 timestep bin（K=25）并对每个 bin 学习增量参数
- 冻结 Stable Diffusion 的所有参数，仅训练 schedule
- 保持 FreeU 的 plug-and-play 特性，不修改模型结构

## 二、项目结构
```bash
Time-Adaptive-FreeU/
├─ scripts/
│  ├─ init.py
│  ├─ train_time_adaptive_freeu.py      # 训练 Time-Adaptive FreeU schedule
│  └─ time_adaptive_freeu_cli.py        # 推理与对比脚本
├─ src/
│  ├─ init.py
│  ├─ time_adaptive_freeu_schedule.py   # 可学习 FreeU schedule 定义
│  └─ time_adaptive_freeu_lunch_utils.py# FreeU patch 实现
├─ coco_hf_dataset.py                   # COCO parquet 数据读取
├─ data/
│  └─ coco2017_hf/                       # 数据集（需自行准备）
├─ ckpt/                                # 训练得到的 schedule checkpoint
├─ logs/                                # 训练日志与 CSV 记录
├─ requirements.txt
└─ README.md
```


---

## 三、环境配置

推荐使用 **Python 3.9 / 3.10**，并需要支持 CUDA 的 GPU。

安装依赖：

```bash
pip install -r requirements.txt
```

## 四、数据集准备（重要）

### 1️⃣ 数据格式说明

训练使用的是 **COCO 2017 的 HuggingFace Parquet 格式数据集**。

请将数据集放置为如下目录结构：

```bash
data/coco2017_hf/
├─ .cache/
└─ data/
├─ train-00000-of-00011.parquet
├─ train-00001-of-00011.parquet
├─ train-00002-of-00011.parquet
├─ …
└─ train-00010-of-00011.parquet
```
每个 parquet 文件中包含图像及其对应的文本描述（caption）。

训练脚本中通过如下方式加载数据集：

```python
dataset = CocoHFParquet("data/coco2017_hf", split="train", size=512)
```

## 五、训练 Time-Adaptive FreeU

训练脚本用于在冻结 Stable Diffusion 主干参数的前提下，学习 Time-Adaptive FreeU 的调节参数 schedule。

### 运行方式

请在**仓库根目录**下，使用模块方式启动训练脚本：

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train_time_adaptive_freeu
```
训练过程中：
	•	Stable Diffusion（U-Net / VAE / Text Encoder）参数全部冻结
	•	仅优化 DeltaFreeUSchedule 中的可学习参数
	•	训练日志自动保存至 logs/
	•	schedule 的 checkpoint 周期性保存至 ckpt/
	•	同时记录 FreeU 参数在 inference timestep 上的变化（CSV 文件）

## 六、推理与效果对比

推理脚本用于对比以下三种生成结果：
	1.	原始 Stable Diffusion
	2.	使用固定参数的 FreeU
	3.	使用训练得到的 Time-Adaptive FreeU

### 运行方式

同样在仓库根目录下运行：

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.time_adaptive_freeu_cli
```
脚本会生成三张图像并进行横向拼接，最终保存为对比结果图像，例如：

```bash
compare_freeu_90000ckpts_orange_cat.png
```

