# computer vision assignment

This is my project for the computer vision class. we had to build image filters
from scratch using numpy (python) and templates (c++), without using high level
library functions like cv2.blur or scipy convolution.

## files

- `spatial_processor.py` - the python version. it can:
  - build a 2d coordinate grid and rotate it around a center point
  - split an rgb image into separate red / green / blue channels
  - change brightness and contrast
  - convert rgb to hsv
  - blur an image with a gaussian kernel using a manual convolution
- `filter_engine.cpp` - the c++ version. a template matrix class that works
  with any numeric pixel type (tested with unsigned char and float) and runs a
  spatial filter over an image kernel.

## how to run

python:

```bash
python spatial_processor.py
```

c++ (needs g++ with c++17):

```bash
g++ -std=c++17 -O2 filter_engine.cpp -o filter_engine
./filter_engine
```

## notes

- i wrote the convolution by hand with loops on purpose, that was part of the
  assignment.
- padding is zero padding, so edges get a bit dark after blurring because the
  kernel picks up zeros there.
- still learning, so sorry if some parts are messy.

assignment: build custom 2d grid transformations, color channel splitters, and
spatial blurring in pure numpy without high level helpers, plus a generic c++
image filter engine using templates that accepts arbitrary numeric pixel types
(uchar, float).