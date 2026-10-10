"""
convert_to_ppm.py

Small helper script. The C++ stitcher only understands the simple PPM format,
but the boat photos we downloaded are JPEG. This script just turns every
images/*.jpg into images/*.ppm.

It also shrinks the photos a bit (max side = 700) so the C++ program does not
take forever, because my stitching code uses plain loops over every pixel.

Run it once (needs Python + OpenCV):
    python tools/convert_to_ppm.py
"""

import glob
import os

import cv2

MAX_SIDE = 700
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # phase2_panorama/, the images/ folder lives here

jpg_files = sorted(glob.glob(os.path.join(ROOT, "images", "*.jpg")))
if len(jpg_files) == 0:
    print("No .jpg files found in images/")
else:
    for path in jpg_files:
        img = cv2.imread(path)
        if img is None:
            print("could not read", path)
            continue

        h, w = img.shape[:2]
        scale = MAX_SIDE / float(max(h, w))
        if scale < 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)))

        out = os.path.splitext(path)[0] + ".ppm"
        cv2.imwrite(out, img)
        print("wrote", out, img.shape[1], "x", img.shape[0])

print("done")
