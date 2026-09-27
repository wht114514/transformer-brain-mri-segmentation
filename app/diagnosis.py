import os
import torch
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

from models import USwinTransformer


class DiagnosisApp:
    # 基于 U-ST 的脑肿瘤 MRI 辅助诊疗系统
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = USwinTransformer(in_channels=1, num_classes=3).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        self.root = tk.Tk()
        self.root.title("脑肿瘤 MRI 辅助诊疗系统")
        self.root.geometry("800x600")
        self.show_menu()

    def show_menu(self):
        for w in self.root.winfo_children():
            w.destroy()

        tk.Label(self.root, text="脑肿瘤 MRI 辅助诊疗系统",
                 font=("Arial", 24)).pack(pady=60)

        tk.Button(self.root, text="推理", width=20, height=2,
                  font=("Arial", 16), command=self.show_predict).pack(pady=20)
        tk.Button(self.root, text="退出", width=20, height=2,
                  font=("Arial", 16), command=self.root.quit).pack(pady=20)

    def show_predict(self):
        for w in self.root.winfo_children():
            w.destroy()

        tk.Button(self.root, text="选择图片", width=20, height=2,
                  font=("Arial", 14), command=self.select_image).pack(pady=20)
        tk.Button(self.root, text="返回", width=20, height=2,
                  font=("Arial", 14), command=self.show_menu).pack(pady=20)

        self.result_label = tk.Label(self.root, text="请选择 MRI 图片")
        self.result_label.pack(pady=20)

    def select_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("图片", "*.png *.jpg *.jpeg *.npy")])
        if not path:
            return

        pred = self.predict(path)
        self.show_result(path, pred)

    def predict(self, path):
        if path.endswith(".npy"):
            img = np.load(path).astype(np.float32)
        else:
            img = np.array(Image.open(path).convert("L")).astype(np.float32) / 255.0

        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            pred = self.model(tensor)
        return torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()

    def show_result(self, path, pred):
        color_map = {0: (0, 0, 0), 1: (255, 0, 0), 2: (0, 255, 0), 4: (0, 0, 255)}
        h, w = pred.shape
        colored = np.zeros((h, w, 3), dtype=np.uint8)
        for v, color in color_map.items():
            colored[pred == v] = color

        image = Image.fromarray(colored).resize((400, 400))
        photo = ImageTk.PhotoImage(image)

        self.result_label.config(text="推理结果", image=photo)
        self.result_label.image = photo

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = DiagnosisApp("checkpoints/best_model.pth")
    app.run()
