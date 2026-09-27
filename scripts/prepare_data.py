import os
import argparse
import numpy as np

from data.preprocess import load_brats_case, slice_2d, normalize


def prepare_2d(raw_dir, out_dir, threshold=0.05):
    # 3D 数据切 2D 切片，保存为 npy
    os.makedirs(out_dir, exist_ok=True)
    all_slices = []
    all_labels = []

    cases = sorted(os.listdir(raw_dir))
    for case in cases:
        case_dir = os.path.join(raw_dir, case)
        if not os.path.isdir(case_dir):
            continue

        image, seg = load_brats_case(case_dir, modalities=("t1",), has_seg=True)
        img = normalize(image[0])
        slices, labels = slice_2d(img, seg, threshold=threshold)

        for s, l in zip(slices, labels):
            all_slices.append(s)
            all_labels.append(l)

    np.save(os.path.join(out_dir, "slices.npy"), np.array(all_slices))
    np.save(os.path.join(out_dir, "labels.npy"), np.array(all_labels))
    print(f"生成 {len(all_slices)} 张切片，保存到 {out_dir}")


def prepare_3d(raw_dir, out_dir):
    # 每个病例的四种模态打包成 pkl
    import pickle
    os.makedirs(out_dir, exist_ok=True)

    for case in sorted(os.listdir(raw_dir)):
        case_dir = os.path.join(raw_dir, case)
        if not os.path.isdir(case_dir):
            continue

        image, seg = load_brats_case(case_dir, has_seg=True)
        image = normalize(image)
        sample = {"image": image, "seg": seg}

        with open(os.path.join(out_dir, f"{case}.pkl"), "wb") as f:
            pickle.dump(sample, f)

    print(f"打包完成，保存到 {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["2d", "3d"], required=True)
    parser.add_argument("--raw_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    if args.mode == "2d":
        prepare_2d(args.raw_dir, args.out_dir)
    else:
        prepare_3d(args.raw_dir, args.out_dir)
