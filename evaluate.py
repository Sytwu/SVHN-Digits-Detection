# evaluate.py

import torch
from torch.amp import autocast
from torchvision.ops.boxes import box_iou
from tqdm import tqdm


def evaluate(model, data_loader, device):
    # Evaluate the model using mean Average Precision (mAP)
    model.eval()
    iou_threshold = 0.5
    APs = []

    with torch.no_grad():
        bar = tqdm(data_loader, desc="Evaluating", leave=False)
        for imgs, tars in bar:
            images = list(img.to(device) for img in imgs)
            targets = [{k: v.to(device) for k, v in t.items()} for t in tars]

            # Perform inference with automatic mixed precision
            with autocast('cuda'):
                outputs = model(images)

            for target, output in zip(targets, outputs):

                # Extract ground truth and predicted boxes and labels
                gt_boxes = target["boxes"]
                gt_labels = target["labels"]
                pred_boxes = output["boxes"]
                pred_scores = output["scores"]
                pred_labels = output["labels"]

                if len(pred_boxes) == 0:
                    APs.append(0)
                    continue

                # Sort predictions by confidence score
                idx = torch.argsort(pred_scores, descending=True)
                pred_boxes, pred_labels = pred_boxes[idx], pred_labels[idx]

                # Compute IoU between predicted and ground truth boxes ans
                ious = box_iou(pred_boxes, gt_boxes)

                # Initialize tensors to track matches, TP, and FP
                t_bool = torch.bool
                t_f32 = torch.float32
                matched = torch.zeros(len(gt_boxes), dtype=t_bool).to(device)
                tp = torch.zeros(len(pred_boxes), dtype=t_f32).to(device)
                fp = torch.zeros(len(pred_boxes), dtype=t_f32).to(device)

                # Determine true positives and false positives
                for i, pred_box in enumerate(pred_boxes):
                    max_iou, max_idx = torch.max(ious[i], dim=0)

                    over_thres = max_iou >= iou_threshold
                    box_unmatched = not matched[max_idx]
                    label_matched = pred_labels[i] == gt_labels[max_idx]

                    if over_thres and box_unmatched and label_matched:
                        tp[i] = 1
                        matched[max_idx] = True
                    else:
                        fp[i] = 1

                # Compute cumulative true positives and false positives
                tp_cumsum = torch.cumsum(tp, dim=0)
                fp_cumsum = torch.cumsum(fp, dim=0)

                # Calculate precision and recall
                precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
                recalls = tp_cumsum / (len(gt_boxes) + 1e-6)

                # Compute Average Precision (AP) using the trapezoidal rule
                AP = torch.trapz(precisions, recalls).item()
                APs.append(AP)

    # Calculate and return mean Average Precision (mAP)
    mean_AP = sum(APs) / len(APs)
    return mean_AP
