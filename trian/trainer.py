import os
import time
import torch
import torch.nn as nn

from ..utils.metrics import dice_loss, dice_coefficient, mean_iou


class Trainer:
    def __init__(self, model, device, num_classes=3, lr=1e-4, optimizer="sgd"):
        self.model = model.to(device)
        self.device = device
        self.num_classes = num_classes

        if optimizer == "sgd":
            self.optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        else:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=True)

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0
        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            pred = self.model(images)
            loss = dice_loss(pred, labels, self.num_classes)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        total_dice = 0.0
        total_iou = 0.0
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                pred = self.model(images)
                total_dice += dice_coefficient(pred, labels, self.num_classes)
                total_iou += mean_iou(pred, labels, self.num_classes)

        n = len(dataloader)
        return total_dice / n, total_iou / n

    def run(self, train_loader, val_loader, epochs, save_dir="checkpoints"):
        os.makedirs(save_dir, exist_ok=True)
        best_dice = 0.0

        for epoch in range(epochs):
            start = time.time()
            loss = self.train_one_epoch(train_loader)
            dice, iou = self.evaluate(val_loader)

            print(f"epoch {epoch+1}/{epochs}  loss={loss:.4f}  "
                  f"dice={dice:.4f}  miou={iou:.4f}  time={time.time()-start:.1f}s")

            if dice > best_dice:
                best_dice = dice
                torch.save(self.model.state_dict(),
                           os.path.join(save_dir, "best_model.pth"))

        print(f"训练完成，最佳 Dice = {best_dice:.4f}")
