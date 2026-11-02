import argparse
import csv
import os

import numpy as np
from PIL import Image, ImageDraw


def make_background(size, rng):
    # a noisy solid colour. the noise stops the model from just
    # learning one exact background colour.
    base = rng.integers(20, 235, size=3).astype(np.int16)
    noise = rng.normal(0, 10, (size, size, 3)).astype(np.int16)
    arr = np.clip(base + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, "RGB")


def pick_shape_color(bg_color, rng):
    # keep the shape clearly different from the background
    for _ in range(100):
        color = tuple(int(v) for v in rng.integers(0, 256, size=3))
        diff = sum(abs(color[i] - bg_color[i]) for i in range(3))
        if diff > 220:
            return color
    return (255, 255, 255)


def make_image(size, rng, empty_chance):
    img = make_background(size, rng)
    bg_color = img.getpixel((0, 0))

    if rng.random() < empty_chance:
        return img, "none", (0.0, 0.0, 0.0, 0.0)

    shape = str(rng.choice(["circle", "rect", "triangle"]))
    color = pick_shape_color(bg_color, rng)
    draw = ImageDraw.Draw(img)

    if shape == "circle":
        radius = int(rng.integers(12, min(32, size // 2 - 4)))
        w = 2 * radius
        h = 2 * radius
    elif shape == "rect":
        w = int(rng.integers(24, 64))
        h = int(rng.integers(24, 64))
    else:
        w = int(rng.integers(28, 64))
        h = int(rng.integers(28, 64))

    w = min(w, size - 8)
    h = min(h, size - 8)

    # pick a centre that keeps the whole box inside the image
    cx = int(rng.integers(w // 2 + 2, size - w // 2 - 2))
    cy = int(rng.integers(h // 2 + 2, size - h // 2 - 2))

    x1 = cx - w // 2
    x2 = cx + w // 2
    y1 = cy - h // 2
    y2 = cy + h // 2

    if shape == "circle":
        draw.ellipse([x1, y1, x2, y2], fill=color)
    elif shape == "rect":
        draw.rectangle([x1, y1, x2, y2], fill=color)
    else:
        # a slightly wonky triangle so every image is not identical
        top_x = cx + int(rng.integers(-w // 4, w // 4 + 1))
        points = [(top_x, y1), (x1, y2), (x2, y2)]
        draw.polygon(points, fill=color)

    box = (cx / size, cy / size, w / size, h / size)
    return img, shape, box


def write_split(out_dir, count, size, empty_chance, rng, prefix):
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    rows = []
    for i in range(count):
        img, shape, box = make_image(size, rng, empty_chance)
        filename = prefix + "_" + str(i).zfill(4) + ".png"
        img.save(os.path.join(img_dir, filename))
        rows.append([
            filename,
            shape,
            round(box[0], 6),
            round(box[1], 6),
            round(box[2], 6),
            round(box[3], 6),
        ])

    labels_path = os.path.join(out_dir, "labels.csv")
    with open(labels_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "shape", "cx", "cy", "w", "h"])
        writer.writerows(rows)

    print("wrote", count, "images to", out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data")
    parser.add_argument("--train", type=int, default=480)
    parser.add_argument("--val", type=int, default=120)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--empty", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    train_dir = os.path.join(args.out, "train")
    val_dir = os.path.join(args.out, "val")

    write_split(train_dir, args.train, args.size, args.empty, rng, "train")
    write_split(val_dir, args.val, args.size, args.empty, rng, "val")

    print("done. next: python train.py")


if __name__ == "__main__":
    main()
