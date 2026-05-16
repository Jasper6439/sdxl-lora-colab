# SDXL LoRA 训练 Colab Notebook

这个仓库包含一个完整的 Google Colab Notebook，用于训练 **车柱完** 角色的 SDXL LoRA 模型。

## 🚀 快速开始

### 1. 在 Colab 中打开 Notebook

点击下面的按钮在 Google Colab 中直接打开 Notebook：

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Jasper6439/sdxl-lora-colab/blob/main/colab_sdxl_lora_training.ipynb)

或者直接访问：
```
https://colab.research.google.com/github/Jasper6439/sdxl-lora-colab/blob/main/colab_sdxl_lora_training.ipynb
```

### 2. 设置 GPU 运行时

1. 点击右上角 **"连接到托管运行时"**
2. 菜单：**运行时 → 更改运行时类型**
3. 选择：**硬件加速器: GPU**（推荐 T4 或 A100）
4. 点击 **保存**

### 3. 按顺序运行单元格

Notebook 包含 9 个步骤：

1. ✅ 检查 GPU 配置
2. ✅ 安装 kohya_ss 训练工具
3. ✅ 下载 SDXL 1.0 模型
4. ✅ 创建训练数据目录
5. ✅ 上传训练数据（手动或 Google Drive）
6. ✅ 创建训练配置
7. ✅ 测试训练环境
8. 🚀 开始训练（20 epochs）
9. 📥 下载训练好的 LoRA 模型

## 📋 系统要求

- **Google 账号**（用于访问 Google Colab）
- **GPU 运行时**（T4 免费，A100 需要 Pro）
- **训练数据**：93 张图片 + 标签文件（已优化）
- **时间**：约 1-2 小时（20 epochs）

## 📊 训练配置

- **模型**：SDXL 1.0（6.5GB）
- **Batch Size**：2（T4 16GB VRAM 够用）
- **Network Dim**：32
- **Epochs**：20
- **分辨率**：512x512
- **混合精度**：FP16

## 📁 训练数据

**触发词**：`车柱完`

**标签优化**：
- ✅ 统一触发词为 `车柱完`
- ✅ 移除英文触发词变体
- ✅ 添加角色特征标签（black_hair, brown_eyes, short_hair）
- ✅ 按标签组排序（角色特征 → 服装 → 动作 → 背景）

## ⚙️ 输出

训练完成后会生成：

- `chezhuhuan_sdxl_000005.safetensors`（5 epoch）
- `chezhuhuan_sdxl_000010.safetensors`（10 epoch）
- `chezhuhuan_sdxl_000015.safetensors`（15 epoch）
- `chezhuhuan_sdxl_000020.safetensors`（20 epoch，最终模型）

## 💡 使用训练好的 LoRA

1. 下载 `.safetensors` 文件
2. 放入 Stable Diffusion WebUI 的 `models/Lora/` 目录
3. 在提示词中使用：`<lora:chezhuhuan_sdxl:0.7> 车柱完`
4. 调整 LoRA 权重（建议 0.6-0.8）

## ⚠️ 注意事项

1. **Colab 时间限制**
   - 免费版：12 小时
   - Pro 版：24 小时
   - 建议每 5 个 epoch 保存一次（已配置）

2. **数据持久化**
   - Colab 虚拟机关机后数据会丢失
   - **务必下载训练好的 LoRA 模型！**

3. **网络稳定性**
   - 可以关闭浏览器，训练会在后台继续
   - 但网络断开超过 30 分钟可能会断开连接

## 🔧 常见问题

### CUDA Out of Memory

**解决方法：**
- 修改 Notebook 中的配置：`train_batch_size = 1`
- 添加：`gradient_accumulation_steps = 2`
- 降低：`network_dim = 16`

### 训练速度慢

**解决方法：**
- 确保安装了 xformers
- 在配置文件中启用：`xformers = true`

## 📚 相关资源

- [kohya_ss GitHub](https://github.com/bmaltais/kohya_ss)
- [SDXL 模型下载](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)
- [Google Colab](https://colab.research.google.com/)

## 📄 许可证

MIT License

---

**作者**：Jasper6439  
**创建时间**：2026-05-17  
**用途**：车柱完角色 LoRA 模型训练
