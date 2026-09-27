import argparse
import torch
import numpy as np
from PIL import Image

from models import USwinTransformer


def predict(image_path, model_path, save_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = USwinTransformer(in_channels=1, num_classes=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    if image_path.endswith(".npy"):
        img = np.load(image_path).astype(np.float32)
    else:
        img = np.array(Image.open(image_path).convert("L")).astype(np.float32) / 255.0

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(tensor)
    pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

    if save_path:
        color_map = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0), 4: (0, 0, 255)}
        h, w = pred.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for v, color in color_map.items():
            colored[pred == v] = color
        Image.fromarray(colored).save(save_path)
        print(f"结果保存到 {save_path}")

    return pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--model", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    predict(args.image, args.model, args.save)
