import torch
import torch.nn as nn


class SegmentationMetrics:

    def __init__(self, class_names, thresholds=None):
        self.class_names = class_names
        if thresholds is None:
            thresholds = [0.5] * len(class_names)
        self.thresholds = thresholds
        self.criterion = nn.BCEWithLogitsLoss()
        self.criterion_each = nn.BCEWithLogitsLoss(reduction="none")

    @torch.no_grad()
    def loss(self, logits, masks):
        return self.criterion(logits, masks,).item()

    @torch.no_grad()
    def class_loss(self, logits, masks):
        pixel_loss = self.criterion_each( logits, masks)

        result = {}
        for idx, cls in enumerate(self.class_names):
            class_mask = masks[:, idx].bool()
            if class_mask.any():
                result[cls] = pixel_loss[:, idx][class_mask].mean().item()
            else:
                result[cls] = 0.0
        return result

    @torch.no_grad()
    def iou(self, logits, masks):
        thresholds = torch.tensor(
            self.thresholds, device=logits.device, dtype=logits.dtype
            ).view(
                1, len(self.class_names), 1, 1
                )

        pred = (torch.sigmoid(logits) > thresholds).float()
        intersection = (pred * masks).sum(dim=(0, 2, 3))
        union = (pred.sum(dim=(0, 2, 3)) + masks.sum(dim=(0, 2, 3)) - intersection)
        class_iou = (intersection + 1e-6) / (union + 1e-6)

        result = {}
        for idx, cls in enumerate(self.class_names):
            result[cls] = class_iou[idx].item()
        result["mean"] = class_iou.mean().item()

        return result