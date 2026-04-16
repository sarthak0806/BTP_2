import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import open_clip

# Paths
BASE_DIR = Path("dataset")
TRAIN_CSV = BASE_DIR / "train.csv"
ATTR_PARQUET = BASE_DIR / "category_attributes.parquet"
TRAIN_IMG_DIR = BASE_DIR / "train_images"
OUT_DIR = Path("outputs_convnext")

MODEL_NAME = "convnext_xxlarge"
PRETRAIN = "laion2b_s34b_b82k_augreg_soup"

EPOCHS = 6
BATCH_SIZE = 16
LR_HEAD = 1e-4
LR_CLIP = 1e-6
HIDDEN_DIM = 1024
DROPOUT = 0.2
NUM_WORKERS = 4
UNFREEZE_STAGES = 2
GRAD_CLIP_NORM = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_key(category, attr):
    return category.replace(" ", "_") + "__" + attr.replace(" ", "_")


# Dataset
class CategoryDataset(Dataset):
    def __init__(self, df, img_dir, preprocess, category, attr_map, encoders):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.preprocess = preprocess
        self.category = category
        self.attr_map = attr_map
        self.encoders = encoders

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = str(row["id"]).zfill(6)

        img_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            p = self.img_dir / f"{img_id}{ext}"
            if p.exists():
                img_path = p
                break

        try:
            img = Image.open(img_path).convert("RGB") if img_path else Image.new("RGB", (256, 256))
        except:
            img = Image.new("RGB", (256, 256))

        img = self.preprocess(img)

        targets = {}
        for attr, col in self.attr_map.items():
            key = safe_key(self.category, attr)
            val = row[col]

            if pd.isna(val) or str(val) not in self.encoders[attr]:
                targets[key] = -1
            else:
                targets[key] = self.encoders[attr][str(val)]

        return img, targets


def collate_fn(batch):
    images = torch.stack([b[0] for b in batch])
    targets = {}

    for key in batch[0][1]:
        targets[key] = torch.tensor([b[1][key] for b in batch], dtype=torch.long)

    return images, targets


# Model
class CategoryMLP(nn.Module):
    def __init__(self, clip_dim, attribute_dims, hidden=1024, dropout=0.2):
        super().__init__()
        self.heads = nn.ModuleDict()

        for key, ncls in attribute_dims.items():
            self.heads[key] = nn.Sequential(
                nn.Linear(clip_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden // 2),
                nn.LayerNorm(hidden // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden // 2, ncls),
            )

    def forward(self, feats):
        return {key: head(feats) for key, head in self.heads.items()}


def unfreeze_last_stages(clip_model, n_stages=2):
    for p in clip_model.parameters():
        p.requires_grad = False

    visual = clip_model.visual

    try:
        stages = visual.trunk.stages
        total = len(stages)

        for i, stage in enumerate(stages):
            if i >= total - n_stages:
                for p in stage.parameters():
                    p.requires_grad = True

        for name in ["norm_pre", "head"]:
            module = getattr(visual.trunk, name, None)
            if module is not None:
                for p in module.parameters():
                    p.requires_grad = True

    except:
        all_params = list(visual.parameters())
        cutoff = int(len(all_params) * 0.75)
        for p in all_params[cutoff:]:
            p.requires_grad = True

    trainable = sum(p.numel() for p in clip_model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable/1e6:.1f}M")


def build_attribute_info(df_full_cat, category, df_attr):
    row = df_attr[df_attr["Category"] == category]
    attribute_list = [str(a) for a in row["Attribute_list"].iloc[0]]

    attribute_mapping = {}
    attribute_encoders = {}
    attribute_dims = {}

    for i, attr in enumerate(attribute_list):
        col = f"attr_{i+1}"
        attribute_mapping[attr] = col

        vals = sorted(df_full_cat[col].dropna().astype(str).unique().tolist())
        attribute_encoders[attr] = {v: idx for idx, v in enumerate(vals)}
        attribute_dims[safe_key(category, attr)] = len(vals)

    return attribute_list, attribute_mapping, attribute_encoders, attribute_dims


def train_one_epoch(clip_model, model_head, loader, optimizer, scaler, criterion):
    clip_model.train()
    model_head.train()

    total_loss = 0
    steps = 0

    for imgs, targs in tqdm(loader):
        imgs = imgs.to(DEVICE)

        with torch.autocast(device_type=DEVICE.type, dtype=torch.float16):
            feats = clip_model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            logits = model_head(feats.float())

            losses = []
            for key, logit in logits.items():
                tgt = targs[key].to(DEVICE)
                mask = tgt != -1

                if mask.sum() > 0:
                    losses.append(criterion(logit[mask], tgt[mask]).mean())

        if not losses:
            continue

        loss = torch.stack(losses).mean()

        optimizer.zero_grad()
        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(clip_model.parameters()) + list(model_head.parameters()),
            GRAD_CLIP_NORM
        )

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        steps += 1

    return total_loss / max(steps, 1)


@torch.no_grad()
def eval_one_epoch(clip_model, model_head, loader):
    clip_model.eval()
    model_head.eval()

    correct = 0
    total = 0

    for imgs, targs in loader:
        imgs = imgs.to(DEVICE)

        feats = clip_model.encode_image(imgs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        logits = model_head(feats.float())

        for key, logit in logits.items():
            tgt = targs[key].to(DEVICE)
            mask = tgt != -1

            if mask.sum() > 0:
                preds = logit.argmax(1)
                correct += (preds[mask] == tgt[mask]).sum().item()
                total += mask.sum().item()

    return correct / max(total, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    args = parser.parse_args()

    df_train = pd.read_csv(TRAIN_CSV, dtype={"id": str})
    df_attr = pd.read_parquet(ATTR_PARQUET)
    df_train["Category_clean"] = df_train["Category"].astype(str).str.strip()

    if args.mode == "smoke":
        print("Running smoke test...")
        df_train = df_train.sample(min(64, len(df_train)), random_state=42)
        epochs_local = 2
    else:
        epochs_local = EPOCHS

    clip_model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAIN
    )
    clip_model = clip_model.to(DEVICE)

    with torch.no_grad():
        dummy = preprocess_train(Image.new("RGB", (256, 256))).unsqueeze(0).to(DEVICE)
        clip_dim = clip_model.encode_image(dummy).shape[-1]

    categories = sorted(set(df_train["Category_clean"].unique()))

    for category in categories:
        print("\nTraining:", category)

        df_cat = df_train[df_train["Category_clean"] == category].copy()
        df_cat["id"] = df_cat["id"].astype(str).str.zfill(6)

        attr_list, attr_map, attr_encoders, attr_dims = build_attribute_info(
            df_cat, category, df_attr
        )

        train_df, val_df = train_test_split(df_cat, test_size=0.2, random_state=42)

        train_ds = CategoryDataset(train_df, TRAIN_IMG_DIR, preprocess_train, category, attr_map, attr_encoders)
        val_ds = CategoryDataset(val_df, TRAIN_IMG_DIR, preprocess_val, category, attr_map, attr_encoders)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        unfreeze_last_stages(clip_model, UNFREEZE_STAGES)

        model_head = CategoryMLP(clip_dim, attr_dims).to(DEVICE)

        optimizer = torch.optim.AdamW([
            {"params": [p for p in clip_model.parameters() if p.requires_grad], "lr": LR_CLIP},
            {"params": model_head.parameters(), "lr": LR_HEAD},
        ])

        scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_local)
        criterion = nn.CrossEntropyLoss(reduction="none")

        for epoch in range(epochs_local):
            loss = train_one_epoch(clip_model, model_head, train_loader, optimizer, scaler, criterion)
            acc = eval_one_epoch(clip_model, model_head, val_loader)
            scheduler.step()

            print(f"Epoch {epoch+1}: loss={loss:.4f}, val_acc={acc:.4f}")


if __name__ == "__main__":
    main()
