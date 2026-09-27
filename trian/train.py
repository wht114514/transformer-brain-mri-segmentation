import argparse
import torch
from torch.utils.data import DataLoader

from models import LinearNet, USwinTransformer, TransBTS
from data.dataset import Brats2DDataset, Brats3DDataset
from data.transforms import build_transform
from train.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="uswin",
                        choices=["linear", "uswin", "transbts"])
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    return parser.parse_args()


def build_model(name, device):
    if name == "linear":
        model = LinearNet(in_channels=1, num_classes=3)
    elif name == "uswin":
        model = USwinTransformer(in_channels=1, num_classes=3)
    elif name == "transbts":
        model = TransBTS(in_channels=4, num_classes=3)
    else:
        raise ValueError(f"未知模型 {name}")
    return model.to(device)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(args.model, device)

    if args.model == "transbts":
        train_set = Brats3DDataset(args.data_dir, split="train", fold=args.fold)
        val_set = Brats3DDataset(args.data_dir, split="val", fold=args.fold)
    else:
        train_set = Brats2DDataset(args.data_dir, split="train", fold=args.fold,
                                   transform=build_transform(train=True))
        val_set = Brats2DDataset(args.data_dir, split="val", fold=args.fold)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    optimizer = "sgd" if args.model != "transbts" else "adam"
    trainer = Trainer(model, device, lr=args.lr, optimizer=optimizer)
    trainer.run(train_loader, val_loader, args.epochs, args.save_dir)


if __name__ == "__main__":
    main()
