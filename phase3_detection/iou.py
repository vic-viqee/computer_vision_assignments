def corners(box):
    # box is cx, cy, w, h. turn it into x1, y1, x2, y2.
    cx, cy, w, h = box
    return cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0


def box_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = corners(box_a)
    bx1, by1, bx2, by2 = corners(box_b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 0.0:
        return 0.0
    return inter / union


def match_one(pred_obj, pred_box, true_obj, true_box, score_thresh=0.5, iou_thresh=0.5):
    # returns true positive, false positive, false negative, iou
    if pred_obj >= score_thresh:
        if true_obj > 0.5:
            iou = box_iou(pred_box, true_box)
            if iou >= iou_thresh:
                return 1, 0, 0, iou
            return 0, 1, 1, iou
        return 0, 1, 0, 0.0

    if true_obj > 0.5:
        return 0, 0, 1, 0.0
    return 0, 0, 0, 0.0


def precision_recall(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return precision, recall, f1
