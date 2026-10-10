# Phase 2: Panorama Stitcher (from scratch, C++)

An image stitching engine written from scratch in a single file while learning
computer vision. No OpenCV — the gradients, the feature detector, the
descriptors, the matcher, the perspective (homography) transform and the
blending are all hand-written C++.

The assignment was image gradients, feature matching and coordinate
transformations, so the whole pipeline is built from those pieces.

## Pipeline (all inside `stitcher.cpp`)

1. **Image gradients / keypoints** — Sobel gradients and a Harris corner
   response on the greyscale image.
2. **Feature descriptors (generic matchers)** — a normalised 17×17 intensity
   patch descriptor per corner; matching is a generic nearest-neighbour search,
   so the descriptor/match step is kept separate from the geometry.
3. **Matching** — brute-force descriptor distance + Lowe's ratio test +
   mutual-best check to throw away bad matches.
4. **Coordinate transformation (perspective matrix)** — RANSAC + direct linear
   transform to fit a 3×3 homography, with Hartley normalisation so the linear
   solve stays well-conditioned.
5. **Warping** — each photo's homography is chained into the first photo's
   frame; every destination pixel is inverse-mapped through the matrix and
   sampled bilinearly.
6. **Blending** — overlapping pixels are averaged onto one unified canvas.

## Build

```
make
```

Needs `g++` with C++17. Nothing else (no OpenCV, no extra libraries).

## Getting test photos

The engine reads the simple **PPM** format (P6) so it needs no image library.
Any set of overlapping photos works. To use the classic OpenCV "boat" panorama
set:

```bash
mkdir -p images && cd images
for i in 1 2 3 4 5; do
  curl -LO https://raw.githubusercontent.com/opencv/opencv_extra/4.x/testdata/stitching/boat$i.jpg
done
cd ..
```

Then turn the JPEGs into shrunken PPM files with the little helper (needs
Python + OpenCV):

```bash
python tools/convert_to_ppm.py
```

## Run

```
./stitcher images/boat1.ppm images/boat2.ppm images/boat3.ppm images/boat4.ppm images/boat5.ppm out.ppm
```

`out.ppm` is the panorama. To view it as a PNG (any image tool, or OpenCV):

```
python -c "import cv2; cv2.imwrite('out.png', cv2.imread('out.ppm'))"
```

You can pass any subset of the photos, e.g. just three:

```
./stitcher images/boat3.ppm images/boat4.ppm images/boat5.ppm out.ppm
```

## Honest limits

- Harris corners + patch descriptors are **not rotation or scale invariant**,
  so only mildly changing photos match well.
- The photos are joined with one flat perspective matrix per pair, chained
  together. The boat set is a **rotation** pan (the camera turns ~60° between
  shots), and chaining flat perspective matrices cannot model a turn that
  large: the error adds up until the "warped" canvas stretches off to
  infinity. Real stitchers first estimate the camera focal length and warp onto
  a cylinder/sphere instead.
- Because of that, the program **stops before a photo that would blow up the
  canvas** and keeps the widest panorama it has so far. Running it on all five
  boat photos therefore gives the valid `boat1 + boat2 + boat3` result, while
  `boat3 + boat4 + boat5` (fewer, smaller turns) stitches fine on its own.
- Blending is a plain average, so overlaps can show a soft seam or ghosting.

These are expected for a learning version, and are the things a real stitcher
(like OpenCV's `Stitcher`) does much better.
