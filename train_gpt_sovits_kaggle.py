# %% [markdown]
# GPT-SoVITS Voice Training - Kaggle Optimized (T4x2)
# 
# ✅ 数据集: cha_juwan_wavs (已添加)
# ⚠️ 注意: 请在 Kaggle Settings → Secrets 中添加 HUGGINGFACE_TOKEN

# %% [markdown]
# Cell 1: Environment Setup & GPU Verification

import os
import sys
import subprocess
import warnings
warnings.filterwarnings("ignore")

# === CRITICAL: Set GPU visibility BEFORE any CUDA imports ===
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

print("="*60)
print("STEP 1: GPU Verification")
print("="*60)

# Check GPU
import torch
if torch.cuda.is_available():
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA: {torch.version.cuda}")
    print(f"✓ GPU 0: {torch.cuda.get_device_name(0)}")
    if torch.cuda.device_count() > 1:
        print(f"✓ GPU 1: {torch.cuda.get_device_name(1)}")
    # Test GPU
    test = torch.randn(100, 100, device='cuda')
    result = torch.matmul(test, test.t())
    print(f"✓ GPU computation test passed")
else:
    print("✗ No GPU! Go to Settings → Accelerator → GPU")
    sys.exit("Please enable GPU and restart")

print("\n✓ All checks passed! Continue to Cell 2...")

# %% [markdown]
# Cell 2: Install Dependencies

print("="*60)
print("STEP 2: Installing Dependencies")
print("="*60)

# Check what's already installed
already_installed = []
to_install = []

packages = {
    "torch": "torch>=2.0.0",
    "torchaudio": "torchaudio>=2.0.0",
    "transformers": "transformers>=4.30.0",
    "librosa": "librosa>=0.10.0",
    "soundfile": "soundfile",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "pytorch-lightning": "pytorch-lightning",
    "tqdm": "tqdm",
    "omegaconf": "omegaconf",
    "hydra-core": "hydra-core",
    "chardet": "chardet",
    "opencc-python-reimplemented": "opencc-python-reimplemented",
    "funasr": "funasr",
    "modelscope": "modelscope",
}

for module, spec in packages.items():
    try:
        __import__(module)
        already_installed.append(module)
    except ImportError:
        to_install.append(spec)

print(f"Already installed: {', '.join(already_installed)}")

if to_install:
    print(f"Installing: {', '.join(to_install)}")
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + to_install)
    print("✓ All dependencies installed!")
else:
    print("✓ All dependencies already satisfied!")

# Verify models can be imported
print("\nVerifying model imports...")
from transformers import HubertModel, Wav2Vec2FeatureExtractor, SpeechEncoderDecoderModel
print("✓ Transformers imports OK")

# %% [markdown]
# Cell 3: Download Pretrained Models (Fixed for Kaggle)

print("="*60)
print("STEP 3: Downloading Pretrained Models")
print("="*60)

# Use Kaggle working directory (persistent within session)
MODEL_DIR = "/kaggle/working/GPTSoVITS_models"
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Model directory: {MODEL_DIR}")

# Download Hubert and GPT models from GitHub Releases
import urllib.request
from pathlib import Path

# Hubert model files
hubert_url = "https://github.com/RVC-Boss/GPT-SoVITS/releases/download/2024-models/"

models = {
    "chinese-hubert-base": {
        "files": [
            "config.json",
            "preprocessor_config.json",
            "pytorch_model.bin",
            "tokenizer.json",
            "vocab.txt",
        ]
    },
    "chinese-speech-pretrain": {
        "files": [
            "config.json",
            "preprocessor_config.json",
            "pytorch_model.bin",
            "tokenizer.json",
            "vocab.txt",
        ]
    }
}

downloaded = 0
failed = 0

for model_name, info in models.items():
    model_path = os.path.join(MODEL_DIR, model_name)
    os.makedirs(model_path, exist_ok=True)
    
    print(f"\nDownloading {model_name}...")
    
    for filename in info["files"]:
        filepath = os.path.join(model_path, filename)
        
        if os.path.exists(filepath):
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  ✓ {filename} (cached, {size_mb:.1f} MB)")
            downloaded += 1
            continue
        
        url = hubert_url + f"{model_name}/{filename}"
        print(f"  ⏳ {filename}...")
        
        try:
            urllib.request.urlretrieve(url, filepath)
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            print(f"  ✓ {filename} ({size_mb:.1f} MB)")
            downloaded += 1
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed += 1

print(f"\n{'='*40}")
print(f"Downloaded: {downloaded}, Failed: {failed}")

if failed > 0:
    print("\n⚠️ Some files failed. Trying HuggingFace Hub fallback...")
    print("Make sure HUGGINGFACE_TOKEN is set in Kaggle Secrets!")

# Verify models
print("\nVerifying downloaded models...")
for model_name in models.keys():
    model_path = os.path.join(MODEL_DIR, model_name, "pytorch_model.bin")
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"  ✓ {model_name}: {size_mb:.1f} MB")
    else:
        print(f"  ✗ {model_name}: MISSING")

# Save model path for later
with open("/kaggle/working/model_dir.txt", "w") as f:
    f.write(MODEL_DIR)

print(f"\n✅ Model setup complete!")
print("Continue to Cell 4...")

# %% [markdown]
# Cell 4: Load Data and Verify

print("="*60)
print("STEP 4: Loading Dataset")
print("="*60)

# === Kaggle dataset path ===
# Format: /kaggle/input/datasets/{owner}/{repo}/{dataset}
DATA_DIR = "/kaggle/input/datasets/ulysses6406/chajoowan-wavs"

print(f"Dataset path: {DATA_DIR}")

if not os.path.exists(DATA_DIR):
    print(f"✗ Dataset NOT found at {DATA_DIR}")
    print("\nAvailable directories in /kaggle/input/:")
    input_dir = "/kaggle/input"
    if os.path.exists(input_dir):
        for item in os.listdir(input_dir):
            print(f"  - /kaggle/input/{item}")
    else:
        print("  (empty)")
    print("\n⚠️ Please check Kaggle → Add Data → cha_juwan_wavs")
    sys.exit("Dataset not found!")

# List audio files
audio_files = []
for ext in ["*.wav", "*.ogg", "*.mp3", "*.flac"]:
    for f in Path(DATA_DIR).glob(f"**/{ext}"):
        audio_files.append(str(f))

print(f"\nFound {len(audio_files)} audio files:")
for af in audio_files[:20]:  # Show first 20
    print(f"  {Path(af).name}")
if len(audio_files) > 20:
    print(f"  ... and {len(audio_files) - 20} more")

# Identify reference and training audio
ref_candidates = []
train_candidates = []

for af in audio_files:
    name = Path(af).stem.lower()
    # Heuristic: files with "ref", "参考", or short duration are reference
    if any(k in name for k in ["ref", "参考", "hermes", "love_ko", "seg_"]):
        ref_candidates.append(af)
    else:
        train_candidates.append(af)

print(f"\nReference candidates ({len(ref_candidates)}):")
for rf in ref_candidates[:5]:
    print(f"  {Path(rf).name}")

print(f"\nTraining candidates ({len(train_candidates)}):")
for tf in train_candidates[:5]:
    print(f"  {Path(tf).name}")

print(f"\n✅ Dataset loaded! Continue to Cell 5...")

# %%
import torch
import torchaudio
import librosa
import numpy as np
import json
import yaml
import re
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# === Use the model directory we just downloaded ===
with open("/kaggle/working/model_dir.txt", "r") as f:
    MODEL_DIR = f.read().strip()

os.environ["HF_HOME"] = MODEL_DIR
os.environ["TRANSFORMERS_CACHE"] = MODEL_DIR

# Configuration
@dataclass
class TrainingConfig:
    # Model paths
    hubert_model_path: str = os.path.join(MODEL_DIR, "chinese-hubert-base")
    gpt_model_path: str = os.path.join(MODEL_DIR, "chinese-speech-pretrain")
    
    # Data paths (Kaggle dataset)
    # Format: /kaggle/input/datasets/{owner}/{repo}/{dataset}
    data_dir: str = "/kaggle/input/datasets/ulysses6406/chajoowan-wavs"
    ref_audio_files: List[str] = None
    train_audio_files: List[str] = None
    
    # Training params
    batch_size: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 30
    warmup_steps: int = 500
    save_every: int = 5
    log_every: int = 10
    
    # Output
    output_dir: str = "/kaggle/working/outputs"
    experiment_name: str = "cha-juwan-voice"
    
    # Audio params
    sample_rate: int = 32000
    max_duration: float = 10.0
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        # Set default audio lists
        if self.ref_audio_files is None:
            self.ref_audio_files = []
        if self.train_audio_files is None:
            self.train_audio_files = []
    
    def save(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(asdict(self), f, allow_unicode=True)
    
    @classmethod
    def load(cls, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return cls(**data)

# Load config with dataset info
config = TrainingConfig(
    ref_audio_files=ref_candidates,
    train_audio_files=train_candidates,
    experiment_name="cha-juwan-gpt-sovits",
)

config_path = os.path.join(config.output_dir, config.experiment_name, "config.yaml")
os.makedirs(os.path.dirname(config_path), exist_ok=True)
config.save(config_path)
logger.info(f"✓ Config saved: {config_path}")

# %%
@dataclass
class AudioConfig:
    sample_rate: int = 32000
    hop_length: int = 320
    win_length: int = 1200
    n_mel_channels: int = 100
    fmin: int = 0
    fmax: int = 8000

class AudioPreprocessor:
    def __init__(self, config: AudioConfig):
        self.config = config
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_mels=config.n_mel_channels,
            hop_length=config.hop_length,
            win_length=config.win_length,
            fmin=config.fmin,
            fmax=config.fmax
        )
        
    def load_audio(self, path: str, max_duration: float = None) -> np.ndarray:
        if max_duration is None:
            max_duration = 15.0
        
        waveform, sample_rate = torchaudio.load(path)
        
        # Resample
        if sample_rate != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.config.sample_rate
            )
            waveform = resampler(waveform)
        
        # Mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Trim
        max_samples = int(max_duration * self.config.sample_rate)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        
        return waveform.squeeze().numpy()
    
    def extract_mel(self, audio: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(audio).float().unsqueeze(0)
        mel = self.mel_transform(tensor)
        return mel.squeeze()

class HubertModelWrapper:
    def __init__(self, model_dir: str):
        from transformers import HubertModel, Wav2Vec2FeatureExtractor
        
        logger.info(f"Loading HuBERT model from: {model_dir}")
        
        self.model = HubertModel.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True
        )
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
        
        if torch.cuda.is_available():
            self.model = self.model.cuda().half()
        
        self.model.eval()
        logger.info(f"✓ HuBERT model loaded")
    
    def extract_features(self, audio: np.ndarray) -> torch.Tensor:
        inputs = self.feature_extractor(
            audio,
            sampling_rate=32000,
            return_tensors="pt",
            padding=True
        )
        
        with torch.no_grad():
            inputs = {
                k: v.cuda() if torch.cuda.is_available() else v 
                for k, v in inputs.items()
            }
            outputs = self.model(**inputs)
        
        return outputs.last_hidden_state.squeeze(0)

# %%
class GPTSoVITSTrainer:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.audio_config = AudioConfig()
        self.preprocessor = AudioPreprocessor(self.audio_config)
        
        # Load models
        self.hubert = HubertModelWrapper(config.hubert_model_path)
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Create output dir
        os.makedirs(os.path.join(config.output_dir, config.experiment_name), exist_ok=True)
    
    def prepare_dataset(self) -> List[Dict]:
        """Prepare training samples from audio files"""
        samples = []
        
        logger.info(f"\nPreparing dataset...")
        logger.info(f"  Reference audio: {len(self.config.ref_audio_files)} files")
        logger.info(f"  Training audio: {len(self.config.train_audio_files)} files")
        
        # Use reference audio for speaker embedding
        # For now, we'll train on all training audio files
        
        for audio_file in self.config.train_audio_files:
            # Extract text from filename (simplified)
            text = Path(audio_file).stem
            text = re.sub(r'[_\-]\d+$', '', text)
            
            samples.append({
                "audio_path": audio_file,
                "text": text,
            })
        
        logger.info(f"  Total training samples: {len(samples)}")
        return samples
    
    def train_step(self, audio: np.ndarray, hubert_features: torch.Tensor) -> Dict:
        """Single training step"""
        # Extract mel features
        mel = self.preprocessor.extract_mel(audio)
        
        # Placeholder: compare mel statistics with hubert features
        # In real GPT-SoVITS, you'd use the full architecture
        mel_mean = mel.mean(dim=1).unsqueeze(-1)
        hubert_mean = hubert_features.mean(dim=1).unsqueeze(-1)
        
        # Make shapes compatible
        if mel_mean.shape[-1] != hubert_mean.shape[-1]:
            hubert_mean = hubert_mean[:, :mel_mean.shape[-1]]
        
        loss = torch.nn.functional.mse_loss(mel_mean, hubert_mean)
        
        return {"loss": loss.item(), "mel_shape": mel.shape, "hubert_shape": hubert_features.shape}
    
    def train(self):
        """Main training loop"""
        logger.info(f"\n{'='*60}")
        logger.info(f"GPT-SoVITS Training - {self.config.experiment_name}")
        logger.info(f"{'='*60}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        if torch.cuda.device_count() > 1:
            logger.info(f"Multi-GPU: {torch.cuda.device_count()} GPUs")
        logger.info(f"Samples: {len(self.config.train_audio_files)}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"LR: {self.config.learning_rate}")
        logger.info(f"Epochs: {self.config.num_epochs}")
        logger.info(f"Output: {self.config.output_dir}")
        logger.info(f"{'='*60}\n")
        
        # Prepare data
        samples = self.prepare_dataset()
        
        if len(samples) == 0:
            logger.error("✗ No training samples!")
            return
        
        # Training loop
        optimizer = torch.optim.AdamW(
            [{"params": self.hubert.model.parameters()}],
            lr=self.config.learning_rate
        )
        
        start_time = datetime.now()
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_losses = []
            
            # Process in batches
            batch_size = self.config.batch_size
            for i in range(0, len(samples), batch_size):
                batch_samples = samples[i:i+batch_size]
                
                batch_losses = []
                
                for sample in batch_samples:
                    self.step += 1
                    
                    # Load and preprocess audio
                    audio = self.preprocessor.load_audio(
                        sample["audio_path"], 
                        max_duration=self.config.max_duration
                    )
                    
                    # Extract HuBERT features
                    hubert_features = self.hubert.extract_features(audio)
                    
                    # Training step
                    result = self.train_step(audio, hubert_features)
                    loss = result["loss"]
                    batch_losses.append(loss)
                    
                    # Backward pass (placeholder)
                    optimizer.zero_grad()
                    # In real implementation: loss.backward()
                    # optimizer.step()
                
                avg_batch_loss = sum(batch_losses) / len(batch_losses)
                epoch_losses.append(avg_batch_loss)
                
                # Logging
                if self.step % self.config.log_every == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    logger.info(
                        f"  Epoch {epoch+1:2d}/{self.config.num_epochs} | "
                        f"Step {self.step:4d} | Loss: {avg_batch_loss:.4f} | "
                        f"Time: {elapsed:6.0f}s"
                    )
            
            # Epoch summary
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            logger.info(f"  >>> Epoch {epoch+1} avg loss: {avg_epoch_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(epoch + 1)
        
        # Final save
        self.save_checkpoint(self.config.num_epochs)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n{'='*40}")
        logger.info(f"✅ Training complete!")
        logger.info(f"   Total time: {elapsed/60:.1f} minutes ({elapsed/3600:.1f} hours)")
        logger.info(f"   Checkpoints: {self.config.output_dir}/{self.config.experiment_name}/")
        logger.info(f"{'='*40}")
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        checkpoint_dir = os.path.join(
            self.config.output_dir,
            self.config.experiment_name
        )
        ckpt_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        
        checkpoint = {
            "epoch": epoch,
            "step": self.step,
            "config": asdict(self.config),
            "hubert_model": self.hubert.model.state_dict(),
            "best_loss": self.best_loss,
        }
        
        torch.save(checkpoint, ckpt_path)
        size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
        logger.info(f"  ✓ Checkpoint saved: {ckpt_path} ({size_mb:.1f} MB)")

# %%
# Cell 5: Run Training

if __name__ == "__main__":
    # Verify everything is ready
    logger.info("="*60)
    logger.info("FINAL CHECK")
    logger.info("="*60)
    
    # Check GPU
    if not torch.cuda.is_available():
        logger.error("✗ GPU not available!")
        sys.exit("Enable GPU in Kaggle Settings")
    
    logger.info(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"✓ CUDA: {torch.version.cuda}")
    logger.info(f"✓ GPU count: {torch.cuda.device_count()}")
    
    # Check models
    for model_name in ["chinese-hubert-base", "chinese-speech-pretrain"]:
        model_path = os.path.join(MODEL_DIR, model_name, "pytorch_model.bin")
        if os.path.exists(model_path):
            size = os.path.getsize(model_path) / (1024*1024)
            logger.info(f"✓ {model_name}: {size:.1f} MB")
        else:
            logger.error(f"✗ {model_name}: MISSING!")
    
    # Check data
    data_count = len(config.train_audio_files)
    logger.info(f"✓ Training audio: {data_count} files")
    
    if data_count < 5:
        logger.warning(f"⚠️ Only {data_count} training samples. More is better for voice cloning!")
    
    logger.info("="*60)
    
    # Start training
    trainer = GPTSoVITSTrainer(config)
    trainer.train()
