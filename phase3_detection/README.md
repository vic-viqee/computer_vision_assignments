# Phase 3: object detection with PyTorch

This is my phase 3 assignment. The brief was to train a custom object detector
in PyTorch, write the data preprocessing, build a CNN with residual skip
connections, calculate IoU, and evaluate the detector on a custom dataset.

I did not use torchvision or any pretrained model. The dataset, the model, the
loss, the IoU code and the evaluation are all in this folder.

## what it detects

There is one object per image (or no object at all). The object is a random
circle, rectangle or triangle drawn on a noisy background. The model predicts:

- a score for "is there an object here?"
- a box around the object: `cx, cy, w, h` (all normalized to 0..1)

It does not predict which shape it is. I kept it to one object because that was
the part I could train and debug on my CPU.

## files

- `make_data.py` - makes the synthetic dataset and the `labels.csv` files
- `dataset.py` - loads the images, normalizes them, and does simple augmentation
- `model.py` - the CNN with residual blocks and the two prediction heads
- `iou.py` - my own IoU, matching and precision/recall code
- `train.py` - training loop, validation, checkpoint and loss curve
- `evaluate.py` - loads the checkpoint and reports detection metrics
- `requirements.txt`
- `samples/` - a few images from the validation set
- `results/` - metrics, training curve and a grid of predictions

## dataset

`make_data.py` draws 128x128 RGB images. Each image has one shape or is empty.
The shape size, position, colour and background are random. About 15% of the
images are empty so the model also learns when there is nothing to detect.

The labels are saved as normalized `cx, cy, w, h`. I split the data into
`data/train` and `data/val` so the evaluation is on images the model did not
train on.

Generate it with:

```bash
python make_data.py
```

That creates 480 training images and 120 validation images by default.

## model

The model is a small CNN:

1. a convolution stem that downsamples the image
2. four residual blocks (conv -> batch norm -> relu, then add the input back)
3. flatten the last 8x8 feature map (I did not pool it because the box head
   needs to know where the object is)
4. a small fully connected layer
5. one output for objectness and four outputs for the box

The box outputs go through a sigmoid so they stay between 0 and 1. The loss is
binary cross entropy for objectness plus smooth L1 for the box, and the box loss
is only used on images that actually have an object.

## how to run

I used Python 3.14, which is the version already on my laptop. The code should
also work on 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate

# CPU-only torch keeps the download smaller
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python make_data.py
python train.py
python evaluate.py
```

`train.py` saves the best checkpoint to `checkpoints/best.pt`. `evaluate.py`
writes `results/metrics.txt` and `results/predictions.png`.

## results

RESULTS_PLACEHOLDER

The green boxes are the real labels and the red boxes are the model predictions.
The score in the title is the objectness score. A prediction counts as correct
when the score is at least 0.5 and the IoU with the real box is at least 0.5.

## things I would improve

- The dataset is synthetic and very clean, so the numbers here are much better
  than they would be on real photos.
- The model only handles one object per image.
- It does not classify the shape, it only finds a box.
- A real detector would use anchor boxes or a grid of predictions instead of one
  box from the whole image.
- I only trained on CPU, so the model is small and the images are 128x128.
