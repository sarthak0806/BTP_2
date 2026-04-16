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
LR = 1e-4
HIDDEN_DIM = 1024
DROPOUT = 0.2

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


# Model Head
class CategoryMLP(nn.Module):
    def __init__(self, feature_dim, attribute_dims):
        super().__init__()
        self.heads = nn.ModuleDict()

        for key, ncls in attribute_dims.items():
            self.heads[key] = nn.Sequential(
                nn.Linear(feature_dim, HIDDEN_DIM),
                nn.LayerNorm(HIDDEN_DIM),
                nn.GELU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
                nn.LayerNorm(HIDDEN_DIM // 2),
                nn.GELU(),
                nn.Dropout(DROPOUT),
                nn.Linear(HIDDEN_DIM // 2, ncls),
            )

    def forward(self, feats):
        return {k: head(feats) for k, head in self.heads.items()}


def build_attribute_info(df_full_cat, category, df_attr):
    row = df_attr[df_attr["Category"] == category]
    attr_list = [str(a) for a in row["Attribute_list"].iloc[0]]

    attr_map, encoders, dims = {}, {}, {}

    for i, attr in enumerate(attr_list):
        col = f"attr_{i+1}"
        attr_map[attr] = col

        vals = sorted(df_full_cat[col].dropna().astype(str).unique().tolist())
        encoders[attr] = {v: idx for idx, v in enumerate(vals)}
        dims[safe_key(category, attr)] = len(vals)

    return attr_map, encoders, dims


def train_one_epoch(model, head, loader, optimizer, criterion):
    model.eval()
    head.train()

    total_loss = 0

    for imgs, targs in tqdm(loader):
        imgs = imgs.to(DEVICE)

        with torch.no_grad():
            feats = model.encode_image(imgs)
            feats = feats / feats.norm(dim=-1, keepdim=True)

        logits = head(feats.float())

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
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss


@torch.no_grad()
def eval_one_epoch(model, head, loader):
    model.eval()
    head.eval()

    correct, total = 0, 0

    for imgs, targs in loader:
        imgs = imgs.to(DEVICE)

        feats = model.encode_image(imgs)
        feats = feats / feats.norm(dim=-1, keepdim=True)

        logits = head(feats.float())

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

    df = pd.read_csv(TRAIN_CSV, dtype={"id": str})
    df_attr = pd.read_parquet(ATTR_PARQUET)
    df["Category_clean"] = df["Category"].astype(str).str.strip()

    if args.mode == "smoke":
        print("Smoke mode")
        df = df.sample(min(64, len(df)))
        epochs = 2
    else:
        epochs = EPOCHS

    # Load model (ConvNext via OpenCLIP)
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAIN
    )
    model = model.to(DEVICE)

    # Freeze backbone
    for p in model.parameters():
        p.requires_grad = False

    model.eval()

    # Feature dimension
    with torch.no_grad():
        dummy = preprocess_train(Image.new("RGB", (256, 256))).unsqueeze(0).to(DEVICE)
        feat_dim = model.encode_image(dummy).shape[-1]

    categories = sorted(df["Category_clean"].unique())

    for category in categories:
        print(f"\nTraining: {category}")

        df_cat = df[df["Category_clean"] == category].copy()
        df_cat["id"] = df_cat["id"].astype(str).str.zfill(6)

        attr_map, encoders, dims = build_attribute_info(df_cat, category, df_attr)

        train_df, val_df = train_test_split(df_cat, test_size=0.2, random_state=42)

        train_ds = CategoryDataset(train_df, TRAIN_IMG_DIR, preprocess_train, category, attr_map, encoders)
        val_ds = CategoryDataset(val_df, TRAIN_IMG_DIR, preprocess_val, category, attr_map, encoders)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

        head = CategoryMLP(feat_dim, dims).to(DEVICE)

        optimizer = torch.optim.AdamW(head.parameters(), lr=LR)
        criterion = nn.CrossEntropyLoss(reduction="none")

        for epoch in range(epochs):
            loss = train_one_epoch(model, head, train_loader, optimizer, criterion)
            acc = eval_one_epoch(model, head, val_loader)

            print(f"Epoch {epoch+1}: loss={loss:.4f}, val_acc={acc:.4f}")


if __name__ == "__main__":
    main()
