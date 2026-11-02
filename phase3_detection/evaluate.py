import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader

from dataset import ShapeDataset
from iou import match_one
from model import Detector


def to_pixels(box, size):
    cx, cy, w, h = box
    x1 = (cx - w / 2.0) * size
    y1 = (cy - h / 2.0) * size
    return x1, y1, w * size, h * size


def draw_predictions(images, targets, pred_obj, pred_boxes, names, path, size):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("matplotlib not installed, skipping prediction image")
        return

    count = min(8, images.size(0))
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()

    for i in range(count):
        img = images[i].permute(1, 2, 0).numpy() * 0.5 + 0.5
        img = img.clip(0, 1)

        ax = axes[i]
        ax.imshow(img)
        ax.axis("off")

        true_obj = targets[i, 0].item()
        true_box = targets[i, 1:].tolist()
        score = pred_obj[i].item()
        pred_box = pred_boxes[i].tolist()

        if true_obj > 0.5:
            x, y, w, h = to_pixels(true_box, size)
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="lime", linewidth=2))

        if score >= 0.5:
            x, y, w, h = to_pixels(pred_box, size)
            ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="red", linewidth=2))

        if true_obj > 0.5 and score >= 0.5:
            iou = match_one(score, pred_box, true_obj, true_box)[3]
            ax.set_title("score " + str(round(score, 2)) + "  iou " + str(round(iou, 2)))
        else:
            ax.set_title("score " + str(round(score, 2)))

    for i in range(count, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/val")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--out", default="results")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--score", type=float, default=0.5)
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    torch.set_num_threads(2)
    os.makedirs(args.out, exist_ok=True)

    dataset = ShapeDataset(args.data, size=args.size, train=False)
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

    model = Detector()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    tp = 0
    fp = 0
    fn = 0
    correct_obj = 0
    total = 0
    iou_sum = 0.0
    iou_count = 0

    rows = []
    first_images = None
    first_targets = None
    first_scores = None
    first_boxes = None
    first_names = None

    with torch.no_grad():
        for images, targets in loader:
            pred_obj, pred_box = model(images)
            # the model returns logits, so squash them to 0..1 for scoring
            pred_obj = torch.sigmoid(pred_obj)

            if first_images is None:
                first_images = images
                first_targets = targets
                first_scores = pred_obj
                first_boxes = pred_box
                first_names = [row["filename"] for row in dataset.rows[:images.size(0)]]

            for i in range(images.size(0)):
                true_obj = targets[i, 0].item()
                true_box = targets[i, 1:].tolist()
                score = pred_obj[i].item()
                box = pred_box[i].tolist()

                t, f, n, iou = match_one(
                    score, box, true_obj, true_box,
                    score_thresh=args.score,
                    iou_thresh=args.iou,
                )
                tp += t
                fp += f
                fn += n

                predicted_positive = score >= args.score
                true_positive = true_obj > 0.5
                if predicted_positive == true_positive:
                    correct_obj += 1

                if t == 1:
                    iou_sum += iou
                    iou_count += 1

                rows.append([
                    dataset.rows[total]["filename"],
                    round(true_obj, 3),
                    round(score, 3),
                    round(iou, 3),
                    t,
                    f,
                    n,
                ])
                total += 1

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    mean_iou = iou_sum / iou_count if iou_count > 0 else 0.0
    obj_accuracy = correct_obj / total if total > 0 else 0.0

    lines = []
    lines.append("phase 3 object detector - validation results")
    lines.append("")
    lines.append("images: " + str(total))
    lines.append("score threshold: " + str(args.score))
    lines.append("iou threshold: " + str(args.iou))
    lines.append("")
    lines.append("true positives: " + str(tp))
    lines.append("false positives: " + str(fp))
    lines.append("false negatives: " + str(fn))
    lines.append("precision: " + str(round(precision, 3)))
    lines.append("recall: " + str(round(recall, 3)))
    lines.append("f1: " + str(round(f1, 3)))
    lines.append("objectness accuracy: " + str(round(obj_accuracy, 3)))
    lines.append("mean iou (true positives): " + str(round(mean_iou, 3)))

    with open(os.path.join(args.out, "metrics.txt"), "w") as f:
        f.write("\n".join(lines) + "\n")

    with open(os.path.join(args.out, "predictions.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "true_obj", "pred_obj", "iou", "tp", "fp", "fn"])
        writer.writerows(rows)

    if first_images is not None:
        draw_predictions(
            first_images, first_targets, first_scores, first_boxes,
            first_names, os.path.join(args.out, "predictions.png"), args.size,
        )

    print("\n".join(lines))
    print("")
    print("saved", os.path.join(args.out, "metrics.txt"))
    print("saved", os.path.join(args.out, "predictions.png"))


if __name__ == "__main__":
    main()
