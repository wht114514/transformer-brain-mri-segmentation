import os
import numpy as np
import nibabel as nib


def load_nii(path):
    return nib.load(path).get_fdata().astype(np.float32)


def load_brats_case(case_dir, modalities=("t1", "t2", "t1ce", "flair"), has_seg=True):
    images = []
    for mod in modalities:
        name = [f for f in os.listdir(case_dir) if mod in f and f.endswith(".nii.gz")]
        if not name:
            raise FileNotFoundError(f"{case_dir} 缺少 {mod} 模态")
        images.append(load_nii(os.path.join(case_dir, name[0])))

    image = np.stack(images, axis=0)

    seg = None
    if has_seg:
        seg_name = [f for f in os.listdir(case_dir) if "seg" in f and f.endswith(".nii.gz")]
        if seg_name:
            seg = load_nii(os.path.join(case_dir, seg_name[0]))

    return image, seg


def slice_2d(image, seg=None, threshold=0.05, axis=2):
    # 按阈值 0.05 切 2D 切片，只保留含有效组织的切片
    slices = []
    seg_slices = []

    n_slices = image.shape[axis]
    for i in range(n_slices):
        img_slice = np.take(image, i, axis=axis)
        if seg is not None:
            seg_slice = np.take(seg, i, axis=axis)
            ratio = (seg_slice > 0).sum() / seg_slice.size
            if ratio < threshold:
                continue
            seg_slices.append(seg_slice)
        slices.append(img_slice)

    if seg is not None:
        return np.array(slices), np.array(seg_slices)
    return np.array(slices)


def normalize(image):
    img = image.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img
