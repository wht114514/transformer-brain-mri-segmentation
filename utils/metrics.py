import torch


def dice_coefficient(pred, target, num_classes=3):
    pred = torch.argmax(pred, dim=1)
    dice = 0.0
    count = 0
    for c in range(1, num_classes):  # 跳过背景
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum().float()
        union = p.sum().float() + t.sum().float()
        if union > 0:
            dice += (2 * inter / union).item()
            count += 1
    return dice / max(count, 1)


def mean_iou(pred, target, num_classes=3):
    pred = torch.argmax(pred, dim=1)
    iou = 0.0
    count = 0
    for c in range(1, num_classes):
        p = (pred == c)
        t = (target == c)
        inter = (p & t).sum().float()
        union = (p | t).sum().float()
        if union > 0:
            iou += (inter / union).item()
            count += 1
    return iou / max(count, 1)


def dice_loss(pred, target, num_classes=3, smooth=1.0):
    pred = torch.softmax(pred, dim=1)
    target_onehot = torch.zeros_like(pred)
    target_onehot.scatter_(1, target.unsqueeze(1), 1)

    total = 0.0
    for c in range(1, num_classes):
        p = pred[:, c]
        t = target_onehot[:, c]
        inter = (p * t).sum()
        union = p.sum() + t.sum()
        total += 1 - (2 * inter + smooth) / (union + smooth)
    return total / (num_classes - 1)
