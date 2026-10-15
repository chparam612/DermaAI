import time
import torch
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.panderm import PanDermClassifier
from src.data.loaders import create_dataloaders
from src.data.datasets import load_label_mapping
from src.train.checkpointing import CheckpointManager

print("Initializing PanDermClassifier on CPU...")
t0 = time.time()
label_map = load_label_mapping("configs/label_mapping.json")
num_classes = len(label_map)

# 1. Initialize architecture
model = PanDermClassifier(
    architecture="panderm_base",
    num_classes=num_classes,
    checkpoint_path=None,  # Do not remap as raw pretrained weights
)

# 2. Load fine-tuned state_dict
ckpt_path = "checkpoints/exp2_panderm_v2/best_model.pt"
CheckpointManager.load(ckpt_path, model)
model.eval()
print(f"Model initialized and loaded in {time.time() - t0:.2f}s")

# 3. Create dataloader
print("Creating validation dataloader (batch_size=32, num_workers=0)...")
_, val_loader, _ = create_dataloaders(
    metadata_path="data/splits/lesion_level_splits.csv",
    image_dir="data/images",
    pipeline_type="panderm",
    batch_size=32,
    num_workers=0,
    modality="dermoscopic",
    image_size=224,
    label_mapping=label_map,
)

# 4. Measure forward pass on 1 batch
batch = next(iter(val_loader))
images, labels, _ = batch

print("Running 1 batch (32 images) on CPU...")
t1 = time.time()
with torch.no_grad():
    logits = model(images)
elapsed = time.time() - t1
print(f"1 batch (32 images) took: {elapsed:.2f}s ({elapsed / 32:.3f}s per image)")
est_total = (786 * 2) * (elapsed / 32)
print(f"Estimated time for 1572 images (val + test): {est_total:.1f}s ({est_total / 60:.1f} minutes)")
