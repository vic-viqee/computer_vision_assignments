# computer vision assignments

My coursework for the computer vision class, split by the phase it belongs to.

| phase | folder | what it is |
|-------|--------|------------|
| 1 | [`phase1_filter_engine/`](phase1_filter_engine/) | image filters from scratch: 2d coordinate grid transforms, RGB channel splitting, brightness/contrast, RGB->HSV and Gaussian blurring in pure numpy, plus a generic templated C++ image filter engine |
| 2 | `phase2_panorama/` | a panorama stitcher from scratch in C++: image gradients (Harris), feature matching, perspective (homography) coordinate transforms, warping and blending |
| 3 | [`phase3_detection/`](phase3_detection/) | a small PyTorch object detector trained on a custom synthetic dataset: preprocessing, residual CNN, IoU, precision/recall and prediction visualisation |

## phase 1

Pure numpy image processing plus a templated C++ filter engine. See
[`phase1_filter_engine/README.md`](phase1_filter_engine/README.md).

## phase 2: panorama stitcher

An image stitching engine written from scratch in a single C++ file (no
OpenCV). It finds corners, builds feature descriptors, matches them, estimates
a 3x3 perspective matrix with RANSAC, then warps and blends the photos onto one
canvas.

Build and run:

```bash
cd phase2_panorama
make
./stitcher images/boat1.ppm images/boat2.ppm images/boat3.ppm out.ppm
```

See [`phase2_panorama/README.md`](phase2_panorama/README.md) for the full
pipeline, how to get the test photos, and the honest limits.

## phase 3: object detection

A small PyTorch detector trained from scratch on a generated dataset of shapes.
It uses a residual CNN, predicts an objectness score and a bounding box, and is
evaluated with hand-written IoU, precision and recall.

```bash
cd phase3_detection
python3 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python make_data.py
python train.py
python evaluate.py
```

See [`phase3_detection/README.md`](phase3_detection/README.md) for the dataset,
model, results and limitations.
