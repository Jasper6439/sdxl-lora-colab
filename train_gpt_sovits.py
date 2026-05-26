#!/usr/bin/env python3
"""GPT-SoVITS Voice Training - T4x2 Multi-GPU (Fixed)"""
import subprocess, sys, os, json, shutil, zipfile, re, time, torch

def run_with_progress(cmd, step_name="", env=None):
    """Run a command and show real-time progress."""
    print(f"\n{'='*60}")
    print(f"{step_name}")
    print(f"{'='*60}")
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env
    )
    
    epoch_pattern = re.compile(r'epoch[:\s]+(\d+)/(\d+)', re.IGNORECASE)
    loss_pattern = re.compile(r'loss[:\s]+([\d.]+)', re.IGNORECASE)
    
    current_epoch = 0
    total_epochs = 0
    last_loss = 0.0
    
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
            
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            total_epochs = int(epoch_match.group(2))
        
        loss_match = loss_pattern.search(line)
        if loss_match:
            last_loss = float(loss_match.group(1))
        
        if total_epochs > 0:
            pct = (current_epoch / total_epochs) * 100
            bar_len = 30
            filled = int(bar_len * current_epoch / total_epochs)
            bar = '█' * filled + '░' * (bar_len - filled)
            print(f'\r  [{bar}] {pct:.1f}% | Epoch {current_epoch}/{total_epochs} | Loss: {last_loss:.4f}', end='', flush=True)
        
        if any(kw in line.lower() for kw in ['error', 'fail', 'exception']):
            print(f'\n  ⚠️  {line}')
        elif 'saving' in line.lower() or 'checkpoint' in line.lower():
            print(f'\n  💾 {line}')
    
    print()
    proc.wait()
    return proc.returncode


print("🚀 GPT-SoVITS Voice Training - T4x2 Multi-GPU")
print(f"{'='*60}")

# ============================================================
# Step 1: Verify GPU
# ============================================================
print("\n📊 STEP 1/6: Verify GPU")
n_gpus = torch.cuda.device_count()
print(f"  PyTorch: {torch.__version__}")
print(f"  CUDA available: {torch.cuda.is_available()}")
print(f"  GPU count: {n_gpus}")
for i in range(n_gpus):
    print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

if not torch.cuda.is_available():
    print("\n❌ CUDA not available! Training will use CPU (very slow).")
    print("   Fix: Restart Kaggle kernel after installing PyTorch cu124.")
    sys.exit(1)

# Test GPU
test_tensor = torch.randn(10, 10).cuda()
print(f"  ✅ Test tensor on: {test_tensor.device}")

# ============================================================
# Step 2: Clone GPT-SoVITS
# ============================================================
print("\n📦 STEP 2/6: Clone GPT-SoVITS")
os.chdir('/kaggle/working')
if not os.path.exists('GPT-SoVITS'):
    subprocess.run(['git', 'clone', '--depth=1',
        'https://github.com/RVC-Boss/GPT-SoVITS.git'], check=True)
    print("  ✅ Cloned")
else:
    print("  ✅ Already exists")

# ============================================================
# Step 3: Install dependencies
# ============================================================
print("\n📚 STEP 3/6: Install dependencies")
os.chdir('/kaggle/working/GPT-SoVITS')
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-input',
    '-r', 'requirements.txt'],
    capture_output=False)

print(f"  torch: {torch.__version__}")
print(f"  CUDA: {torch.cuda.is_available()}")

# ============================================================
# Step 4: Download pretrained models
# ============================================================
print("\n📥 STEP 4/6: Download pretrained models")

# Read HF_TOKEN from Kaggle Secrets (secure) or env variable
hf_token = os.environ.get('HF_TOKEN', '')
if not hf_token:
    try:
        from kaggle_secrets import UserSecretsClient
        user_secrets = UserSecretsClient()
        hf_token = user_secrets.get_secret("HF_TOKEN")
        os.environ['HF_TOKEN'] = hf_token
        os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token
        print("  ✅ Loaded HF_TOKEN from Kaggle Secrets")
    except Exception as e:
        print(f"  ⚠️  No HF_TOKEN found (Kaggle Secrets or env var)")
        print(f"     Add HF_TOKEN in Kaggle: Settings → Secrets → Add")
else:
    os.environ['HUGGING_FACE_HUB_TOKEN'] = hf_token
    print("  ✅ Loaded HF_TOKEN from environment")

# Install huggingface_hub with CLI support
print("  Installing huggingface_hub[cli]...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-input',
    'huggingface_hub[cli]'], check=True, capture_output=False)

# Use Python module to avoid PATH issues
print("  Downloading pretrained models (this may take 5-10 min)...")
subprocess.run([
    sys.executable, '-m', 'huggingface_hub.commands.huggingface_cli',
    'download', 'RVC-Boss/GPT-SoVITS',
    '--local-dir', '/kaggle/working/GPT-SoVITS'
], check=True, capture_output=False)

gpt_path = '/kaggle/working/GPT-SoVITS/pretrained_models/gpt.ckpt'
sovits_path = '/kaggle/working/GPT-SoVITS/pretrained_models/sovits.pth'
print(f"  ✅ GPT model: {gpt_path}")
print(f"  ✅ SoVITS model: {sovits_path}")

# ============================================================
# Step 5: Prepare dataset
# ============================================================
print("\n📁 STEP 5/6: Prepare dataset")
# CHANGE THIS to your Kaggle Dataset path
INPUT_DIR = '/kaggle/input/chajoowan-wavs'

DATASET_DIR = '/kaggle/working/datasets/4-cjw'
os.makedirs(DATASET_DIR, exist_ok=True)

# Unzip if needed
for f in os.listdir(INPUT_DIR):
    if f.endswith('.zip'):
        with zipfile.ZipFile(os.path.join(INPUT_DIR, f), 'r') as z:
            z.extractall(INPUT_DIR)
        print(f"  ✅ Unzipped: {f}")

# Collect wav files
input_files = []
for root, dirs, files in os.walk(INPUT_DIR):
    for f in files:
        if f.endswith('.wav'):
            input_files.append(os.path.join(root, f))

print(f"  Found {len(input_files)} WAV files")

# Symlink to dataset dir (save space)
for i, wav_path in enumerate(input_files):
    dst = os.path.join(DATASET_DIR, f'{i:05d}.wav')
    if not os.path.exists(dst):
        os.symlink(wav_path, dst)

print(f"  ✅ Dataset ready: {DATASET_DIR}")

# Annotations
ann_file = os.path.join(INPUT_DIR, 'annotations.json')
if os.path.exists(ann_file):
    with open(ann_file) as f:
        annotations = json.load(f)
    for i, ann in enumerate(annotations):
        txt_path = os.path.join(DATASET_DIR, f'{i:05d}.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(ann.get('text', ''))
    print(f"  ✅ Annotations: {len(annotations)} entries")

# ============================================================
# Step 6: Train (SoVITS + GPT)
# ============================================================
os.chdir('/kaggle/working/GPT-SoVITS')
OUTPUT_DIR = '/kaggle/working/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Force GPU via environment (CRITICAL: set BEFORE subprocess)
env = os.environ.copy()
env['CUDA_VISIBLE_DEVICES'] = '0' if n_gpus < 2 else '0,1'
env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

print(f"\n  Using GPUs: {env['CUDA_VISIBLE_DEVICES']}")
print(f"  CUDA available: {torch.cuda.is_available()}")

# Train SoVITS
rc = run_with_progress([
    sys.executable, 'train.py',
    '-s', DATASET_DIR, '-m', OUTPUT_DIR, '-n', 'cjw',
    '--train_type', 'sovits',
    '--pretrained_sovits', sovits_path,
    '--batch_size', '8', '--epochs', '8',
    '--lr', '4e-4', '--save_every_epoch', '4',
], "STEP 6a/6: Train SoVITS", env=env)

# Train GPT
rc = run_with_progress([
    sys.executable, 'train.py',
    '-s', DATASET_DIR, '-m', OUTPUT_DIR, '-n', 'cjw',
    '--train_type', 'gpt',
    '--pretrained_gpt', gpt_path,
    '--batch_size', '8', '--epochs', '15',
    '--lr', '1e-4', '--save_every_epoch', '5',
], "STEP 6b/6: Train GPT", env=env)

# Summary
print(f"\n{'='*60}")
print("📊 OUTPUT FILES")
print(f"{'='*60}")
for f in sorted(os.listdir(OUTPUT_DIR)):
    fpath = os.path.join(OUTPUT_DIR, f)
    if os.path.isfile(fpath):
        size_mb = os.path.getsize(fpath) / 1024 / 1024
        print(f"  {f} ({size_mb:.1f} MB)")

print(f"\n{'='*60}")
print("✅ TRAINING COMPLETE!")
print(f"{'='*60}")
