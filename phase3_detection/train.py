import argparse
import csv
import os
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import ShapeDataset
from model import Detector


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, optimizer, device, train):
    if train:
        model.train()
    else:
        model.eval()

    bce = nn.BCEWithLogitsLoss()
    smooth = nn.SmoothL1Loss()

    total_loss = 0.0
    total_obj = 0.0
    total_box = 0.0
    seen = 0

    with torch.set_grad_enabled(train):
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)

            pred_obj, pred_box = model(images)

            true_obj = targets[:, 0]
            true_box = targets[:, 1:]

            obj_loss = bce(pred_obj, true_obj)

            positive = true_obj > 0.5
            if positive.any():
                box_loss = smooth(pred_box[positive], true_box[positive])
            else:
                box_loss = torch.zeros((), device=device)

            loss = obj_loss + 5.0 * box_loss

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch_size = images.size(0)
            total_loss += loss.item() * batch_size
            total_obj += obj_loss.item() * batch_size
            total_box += box_loss.item() * batch_size
            seen += batch_size

    return total_loss / seen, total_obj / seen, total_box / seen


def save_history(history, path):
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "train_obj", "val_obj"])
        for row in history:
            writer.writerow(row)


def plot_history(history, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed, skipping plot")
        return

    epochs = [row[0] for row in history]
    train_loss = [row[1] for row in history]
    val_loss = [row[2] for row in history]

    plt.figure(figsize=(6, 4))
    plt.plot(epochs, train_loss, label="train")
    plt.plot(epochs, val_loss, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="checkpoints")
    args = parser.parse_args()

    set_seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device("cpu")

    train_set = ShapeDataset(os.path.join(args.data, "train"), train=True)
    val_set = ShapeDataset(os.path.join(args.data, "val"), train=False)

    train_loader = DataLoader(train_set, batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch, shuffle=False, num_workers=0)

    model = Detector().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.5)

    os.makedirs(args.out, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    best_loss = 999999.0
    history = []

    print("train images:", len(train_set))
    print("val images:", len(val_set))

    for epoch in range(1, args.epochs + 1):
        train_loss, train_obj, train_box = run_epoch(
            model, train_loader, optimizer, device, True
        )
        val_loss, val_obj, val_box = run_epoch(
            model, val_loader, None, device, False
        )
        scheduler.step()

        history.append([epoch, train_loss, val_loss, train_obj, val_obj])

        saved = ""
        if val_loss < best_loss:
            best_loss = val_loss
            checkpoint = {
                "model_state": model.state_dict(),
                "epoch": epoch,
                "val_loss": val_loss,
            }
            torch.save(checkpoint, os.path.join(args.out, "best.pt"))
            saved = " saved"

        print(
            "epoch", epoch, "/", args.epochs,
            "train", round(train_loss, 4),
            "val", round(val_loss, 4),
            "obj", round(val_obj, 4),
            "box", round(val_box, 4),
            saved,
        )

    save_history(history, os.path.join("results", "training_history.csv"))
    plot_history(history, os.path.join("results", "training_curve.png"))
    print("best val loss:", round(best_loss, 4))
    print("checkpoint:", os.path.join(args.out, "best.pt"))


if __name__ == "__main__":
    main()
