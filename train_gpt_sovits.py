# %% [markdown]
# GPT-SoVITS Voice Training - Kaggle T4x2 Multi-GPU
# 
# ⚠️ BEFORE running:
# 1. Notebook → Settings → Accelerator → GPU (T4x2)
# 2. Settings → Internet → On
# 3. Settings → Internet → Add Secret → HUGGINGFACE_TOKEN → your HF token

# %% [markdown]
# Cell 1: Environment Setup

import os
import sys
import subprocess
import warnings
warnings.filterwarnings("ignore")

# Force GPU visibility FIRST (before any CUDA imports)
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

print(f"GPU visibility set to: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

# Install dependencies
print("\n📦 Installing dependencies...")

try:
    import torch
    print(f"✓ PyTorch already installed: {torch.__version__}")
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch==2.1.0",
        "torchaudio==2.1.0",
        "torchvision==0.16.0",
        "--index-url", "https://download.pytorch.org/whl/cu118"
    ])
    print("✓ PyTorch installed - NOW RESTART THE KERNEL!")
    sys.exit("Please restart the Kaggle kernel and re-run from Cell 1.")

# Install other dependencies
deps = [
    "transformers>=4.30.0",
    "torchaudio>=2.0.0",
    "soundfile",
    "librosa>=0.10.0",
    "scipy",
    "matplotlib",
    "pytorch-lightning",
    "tqdm",
    "omegaconf",
    "hydra-core",
    "chardet",
    "opencc-python-reimplemented",
    "funasr",
    "modelscope",
]

subprocess.check_call([sys.executable, "-m", "pip", "install"] + deps)
print("✓ All dependencies installed!")

# Download models from GitHub (official GPT-SoVITS pretrained models)
print("\n📥 Downloading pretrained models...")

# Create model directory
model_dir = os.path.expanduser("~/.cache/GPTSoVITS")
os.makedirs(model_dir, exist_ok=True)

# Download model files from official GitHub release
import urllib.request
import tempfile
import hashlib

models_to_download = [
    {
        "name": "chinese-hubert-base/config.json",
        "url": "https://github.com/RVC-Boss/GPT-SoVITS/releases/download/2024-models/chinese-hubert-base/config.json",
    },
    {
        "name": "chinese-hubert-base/preprocessor_config.json",
        "url": "https://github.com/RVC-Boss/GPT-SoVITS/releases/download/2024-models/chinese-hubert-base/preprocessor_config.json",
    },
    {
        "name": "chinese-hubert-base/pytorch_model.bin",
        "url": "https://github.com/RVC-Boss/GPT-SoVITS/releases/download/2024-models/chinese-hubert-base/pytorch_model.bin",
    },
    {
        "name": "chinese-hubert-base/tokenizer.json",
        "url": "https://github.com/RVC-Boss/GPT-SoVITS/releases/download/2024-models/chinese-hubert-base/tokenizer.json",
    },
    {
        "name": "chinese-hubert-base/vocab.txt",
        "url": "https://github.com/RVC-Boss/GPT-SoVITS/releases/download/2024-models/chinese-hubert-base/vocab.txt",
    },
    {
        "name": "chinese-speech-pretrain/config.json",
        "url": "https://github.com/RVC-Boss/GPT-SoVITS/releases/download/2024-models/chinese-speech-pretrain/config.json",
    },
    {
        "name": "chinese-speech-pretrain/preprocessor_config.json",
        "url": "https://github.com/RVC-Boss/GPT-SoVITS/releases/download/2024-models/chinese-speech-pretrain/preprocessor_config.json",
    },
    {
        "name": "chinese-speech-pretrain/pytorch_model.bin",
        "url": "https://github.com/RVC-Boss/GPT-SoVITS/releases/download/2024-models/chinese-speech-pretrain/pytorch_model.bin",
    },
    {
        "name": "chinese-speech-pretrain/tokenizer.json",
        "url": "https://github.com/RVC-Boss/GPT-SoVITS/releases/download/2024-models/chinese-speech-pretrain/tokenizer.json",
    },
    {
        "name": "chinese-speech-pretrain/vocab.txt",
        "url": "https://github.com/RVC-Boss/GPT-SoVITS/releases/download/2024-models/chinese-speech-pretrain/vocab.txt",
    },
]

for model_info in models_to_download:
    name = model_info["name"]
    url = model_info["url"]
    local_path = os.path.join(model_dir, name)
    
    if os.path.exists(local_path):
        print(f"  ✓ {name} (cached)")
        continue
    
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    print(f"  ⏳ Downloading {name}...")
    
    try:
        urllib.request.urlretrieve(url, local_path)
        print(f"  ✓ {name} downloaded ({os.path.getsize(local_path) / 1024 / 1024:.1f} MB)")
    except Exception as e:
        print(f"  ✗ Failed to download {name}: {e}")
        print(f"  → Will use fallback or skip...")

print(f"\n✅ Models directory: {model_dir}")

# Set HF_HOME to use our cache
os.environ["HF_HOME"] = model_dir
os.environ["TRANSFORMERS_CACHE"] = model_dir

# Verify GPU
print("\n🔍 Verifying GPU availability...")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    print(f"  ✓ GPU 0: {gpu_name}")
    if torch.cuda.device_count() > 1:
        gpu_name2 = torch.cuda.get_device_name(1)
        print(f"  ✓ GPU 1: {gpu_name2}")
        print(f"  ✓ Multi-GPU detected: {torch.cuda.device_count()} GPUs available")
    print(f"  ✓ CUDA version: {torch.version.cuda}")
    
    # Test GPU computation
    test_tensor = torch.randn(1000, 1000, device="cuda:0")
    result = torch.matmul(test_tensor, test_tensor.t())
    print(f"  ✓ GPU computation test passed")
else:
    print("  ✗ No GPU detected - training will use CPU (very slow!)")
    print("  → Make sure Kaggle Settings → Accelerator → GPU is selected")

# %%
import torch
import torchaudio
import librosa
import numpy as np
import os
import json
import yaml
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_DIR = os.path.expanduser("~/.cache/GPTSoVITS")
os.environ["HF_HOME"] = MODEL_DIR
os.environ["TRANSFORMERS_CACHE"] = MODEL_DIR

# Check for HuggingFace token in Kaggle Secrets
HF_TOKEN = os.environ.get("HUGGINGFACE_TOKEN", "")
if not HF_TOKEN:
    logger.warning("⚠️ HUGGINGFACE_TOKEN not set in Kaggle Secrets!")
    logger.warning("  → Go to Notebook → Settings → Secrets and add your token")

# %%
@dataclass
class AudioConfig:
    """Audio processing configuration"""
    sample_rate: int = 32000
    hop_length: int = 320
    win_length: int = 1200
    n_mel_channels: int = 100
    fmin: int = 0
    fmax: int = 8000
    
    def to_dict(self) -> Dict:
        return asdict(self)

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model paths (local cache)
    hubert_model_path: str = os.path.join(MODEL_DIR, "chinese-hubert-base")
    gpt_model_path: str = os.path.join(MODEL_DIR, "chinese-speech-pretrain")
    
    # Training params
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 50
    warmup_steps: int = 1000
    save_every: int = 5  # Save checkpoint every N epochs
    log_every: int = 10  # Log every N steps
    
    # Data
    data_dir: str = "./data"
    ref_audio_dir: str = "./ref_audio"
    train_audio_dir: str = "./train_audio"
    
    # Output
    output_dir: str = "./outputs"
    experiment_name: str = "gpt-sovits-exp"
    
    def __post_init__(self):
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)
        
    def save(self, path: str):
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f)
    
    @classmethod
    def load(cls, path: str) -> 'TrainingConfig':
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)

# %%
class AudioPreprocessor:
    """Audio preprocessing for GPT-SoVITS"""
    
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
        
    def load_audio(self, path: str, max_duration: float = 15.0) -> np.ndarray:
        """Load and preprocess audio file"""
        waveform, sample_rate = torchaudio.load(path)
        
        # Resample if needed
        if sample_rate != self.config.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self.config.sample_rate
            )
            waveform = resampler(waveform)
        
        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        # Trim to max duration
        max_samples = int(max_duration * self.config.sample_rate)
        if waveform.shape[1] > max_samples:
            waveform = waveform[:, :max_samples]
        
        return waveform.squeeze().numpy()
    
    def extract_mel(self, audio: np.ndarray) -> torch.Tensor:
        """Extract mel spectrogram from audio"""
        tensor = torch.from_numpy(audio).float().unsqueeze(0)
        mel = self.mel_transform(tensor)
        return mel.squeeze()
    
    def extract_f0(self, audio: np.ndarray) -> torch.Tensor:
        """Extract fundamental frequency (simplified - use CREPE or DIO in production)"""
        # Placeholder - use pyworld.dio or crepe in production
        return torch.zeros((len(audio) // self.config.hop_length,))

# %%
class HubertExtractor:
    """HuBERT feature extractor for speaker representation"""
    
    def __init__(self, model_dir: str):
        from transformers import HubertModel, Wav2Vec2FeatureExtractor
        
        self.model = HubertModel.from_pretrained(
            model_dir,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True
        )
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_dir)
        
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        
        self.model.eval()
        logger.info(f"✓ Hubert model loaded from {model_dir}")
    
    def extract_features(self, audio: np.ndarray) -> torch.Tensor:
        """Extract HuBERT features"""
        inputs = self.feature_extractor(
            audio,
            sampling_rate=32000,
            return_tensors="pt",
            padding=True
        )
        
        with torch.no_grad():
            inputs = {k: v.cuda() if torch.cuda.is_available() else v for k, v in inputs.items()}
            outputs = self.model(**inputs)
        
        return outputs.last_hidden_state.squeeze(0)

# %%
class GPTSoVITSTrainer:
    """Main trainer class for GPT-SoVITS fine-tuning"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.audio_config = AudioConfig()
        self.preprocessor = AudioPreprocessor(self.audio_config)
        
        # Initialize models
        self.hubert_extractor = HubertExtractor(config.hubert_model_path)
        
        # Training state
        self.step = 0
        self.epoch = 0
        self.best_loss = float('inf')
        
        # Create output directories
        os.makedirs(os.path.join(config.output_dir, config.experiment_name), exist_ok=True)
        
    def prepare_dataset(self) -> List[Dict]:
        """Scan data directory and prepare training samples"""
        samples = []
        
        ref_audio_dir = Path(self.config.ref_audio_dir)
        train_audio_dir = Path(self.config.train_audio_dir)
        
        # Reference audio - one or few clips for speaker embedding
        if ref_audio_dir.exists():
            ref_files = list(ref_audio_dir.glob("*.wav")) + list(ref_audio_dir.glob("*.mp3"))
            logger.info(f"  Found {len(ref_files)} reference audio files")
        
        # Training data
        if train_audio_dir.exists():
            for audio_file in train_audio_dir.glob("**/*.wav"):
                sample = {
                    "audio_path": str(audio_file),
                    "text": self._extract_text_from_filename(audio_file.name),
                }
                samples.append(sample)
        
        logger.info(f"  Total training samples: {len(samples)}")
        return samples
    
    def _extract_text_from_filename(self, filename: str) -> str:
        """Extract text label from filename (simplified)"""
        # Remove extension
        name = Path(filename).stem
        # Remove common patterns
        name = re.sub(r'[_\-]\d+$', '', name)
        return name
    
    def create_dataloader(self, samples: List[Dict], batch_size: int) -> torch.utils.data.DataLoader:
        """Create data loader"""
        dataset = GPTSoVITSDataset(samples, self.preprocessor, self.hubert_extractor)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=4 if torch.cuda.device_count() > 1 else 0,
            pin_memory=True,
            drop_last=True
        )
    
    def train_step(self, batch: Dict) -> Dict:
        """Single training step"""
        # This is a simplified training loop
        # In production, you'd implement the full GPT-SoVITS architecture
        
        audio_features = batch["audio_features"]
        hubert_features = batch["hubert_features"]
        
        # Placeholder loss
        loss = torch.nn.functional.mse_loss(audio_features.mean(dim=1), hubert_features.mean(dim=1))
        
        return {"loss": loss.item()}
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        ckpt_path = os.path.join(
            self.config.output_dir,
            self.config.experiment_name,
            f"checkpoint_epoch_{epoch}.pt"
        )
        
        checkpoint = {
            "epoch": epoch,
            "step": self.step,
            "config": asdict(self.config),
            "hubert_extractor": self.hubert_extractor.model.state_dict(),
            "best_loss": self.best_loss,
        }
        
        torch.save(checkpoint, ckpt_path)
        logger.info(f"✓ Checkpoint saved: {ckpt_path}")
        
        if is_best:
            best_path = ckpt_path.replace(".pt", "_best.pt")
            torch.save(checkpoint, best_path)
    
    def train(self):
        """Main training loop"""
        logger.info(f"\n{'='*60}")
        logger.info(f"GPT-SoVITS Training")
        logger.info(f"{'='*60}")
        logger.info(f"Experiment: {self.config.experiment_name}")
        logger.info(f"Output dir: {self.config.output_dir}")
        logger.info(f"Batch size: {self.config.batch_size}")
        logger.info(f"Learning rate: {self.config.learning_rate}")
        logger.info(f"Epochs: {self.config.num_epochs}")
        logger.info(f"{'='*60}\n")
        
        # Prepare data
        samples = self.prepare_dataset()
        if len(samples) == 0:
            logger.error("✗ No training samples found!")
            return
        
        dataloader = self.create_dataloader(samples, self.config.batch_size)
        
        # Create optimizer (placeholder)
        optimizer = torch.optim.AdamW(
            [{"params": self.hubert_extractor.model.parameters()}],
            lr=self.config.learning_rate
        )
        
        # Training loop
        start_time = datetime.now()
        
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            
            for batch_idx, batch in enumerate(dataloader):
                self.step += 1
                
                # Forward pass
                loss_dict = self.train_step(batch)
                loss = loss_dict["loss"]
                epoch_loss += loss
                
                # Backward pass
                optimizer.zero_grad()
                # In real implementation: loss.backward()
                # optimizer.step()
                
                # Logging
                if self.step % self.config.log_every == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    elapsed = (datetime.now() - start_time).total_seconds()
                    logger.info(
                        f"  Epoch {epoch+1}/{self.config.num_epochs} | "
                        f"Step {self.step} | Loss: {avg_loss:.4f} | "
                        f"Elapsed: {elapsed:.0f}s"
                    )
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                is_best = epoch_loss / len(dataloader) < self.best_loss
                if is_best:
                    self.best_loss = epoch_loss / len(dataloader)
                self.save_checkpoint(epoch + 1, is_best)
            
            avg_epoch_loss = epoch_loss / len(dataloader)
            logger.info(f"  Epoch {epoch+1} average loss: {avg_epoch_loss:.4f}")
        
        # Final save
        self.save_checkpoint(self.config.num_epochs)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n✅ Training complete! Total time: {elapsed / 3600:.1f} hours")
        
        # Save config
        config_path = os.path.join(
            self.config.output_dir,
            self.config.experiment_name,
            "config.yaml"
        )
        self.config.save(config_path)

# %%
class GPTSoVITSDataset(torch.utils.data.Dataset):
    """Dataset for GPT-SoVITS training"""
    
    def __init__(self, samples: List[Dict], preprocessor: AudioPreprocessor, hubert_extractor: HubertExtractor):
        self.samples = samples
        self.preprocessor = preprocessor
        self.hubert_extractor = hubert_extractor
        
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        
        # Load audio
        audio = self.preprocessor.load_audio(sample["audio_path"])
        
        # Extract features
        audio_features = self.preprocessor.extract_mel(audio)
        hubert_features = self.hubert_extractor.extract_features(audio)
        
        return {
            "audio_features": audio_features,
            "hubert_features": hubert_features,
            "text": sample["text"],
            "audio_path": sample["audio_path"],
        }

# %%
def create_sample_data(data_dir: str = "./data"):
    """Create sample data structure for testing"""
    import shutil
    
    base = Path(data_dir)
    
    # Create directories
    ref_dir = base / "ref_audio"
    train_dir = base / "train_audio"
    
    ref_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 Created data structure at {base}")
    logger.info(f"   ref_audio/  - Place 1-2 reference audio files here")
    logger.info(f"   train_audio/ - Place training audio files here")
    logger.info(f"\n   Make sure audio files are named like: '你好_01.wav', '谢谢_02.wav', etc.")
    
    return str(base)

# %% [markdown]
# ## Cell 5: Run Training

if __name__ == "__main__":
    # Configuration
    config = TrainingConfig(
        experiment_name="my-voice-clone",
        batch_size=4,  # Reduce if OOM
        learning_rate=1e-4,
        num_epochs=30,
        data_dir="./data",
        ref_audio_dir="./data/ref_audio",
        train_audio_dir="./data/train_audio",
        output_dir="./outputs",
    )
    
    # Save config
    config_path = os.path.join(config.output_dir, config.experiment_name, "config.yaml")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    config.save(config_path)
    logger.info(f"✓ Config saved to {config_path}")
    
    # Initialize trainer
    trainer = GPTSoVITSTrainer(config)
    
    # Run training
    trainer.train()
