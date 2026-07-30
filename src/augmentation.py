"""
时序增强 + 坐标抖动 + 随机 mask。仅训练时使用。
"""
import numpy as np


def temporal_scale(keypoints, scale_range=(0.8, 1.2)):
    """
    keypoints: (T, D) float32
    线性插值拉伸/压缩序列长度，模拟不同打手语速度。
    """
    T = keypoints.shape[0]
    if T < 2:
        return keypoints
    scale = np.random.uniform(*scale_range)
    new_T = max(2, int(T * scale))
    old_idx = np.linspace(0, T - 1, T)
    new_idx = np.linspace(0, T - 1, new_T)
    D = keypoints.shape[1]
    augmented = np.zeros((new_T, D), dtype=np.float32)
    for d in range(D):
        augmented[:, d] = np.interp(new_idx, old_idx, keypoints[:, d])
    return augmented


def coordinate_jitter(keypoints, sigma=0.005):
    """
    keypoints: (T, D) float32
    对非零帧加微小高斯噪声，模拟 MediaPipe 检测误差。零帧（手部丢失）保持不变。
    """
    noise = np.random.randn(*keypoints.shape).astype(np.float32) * sigma
    frame_norm = np.abs(keypoints).sum(axis=1, keepdims=True)
    mask = (frame_norm > 1e-6).astype(np.float32)
    return keypoints + noise * mask


def random_mask_keypoints(keypoints, mask_ratio=0.05):
    """
    keypoints: (T, 84) float32
    随机将部分关键点坐标置零，模拟遮挡/检测失败。仅作用于有效帧。
    """
    augmented = keypoints.copy()
    T, D = keypoints.shape
    n_mask = max(1, int(D * mask_ratio))
    for t in range(T):
        if np.abs(augmented[t]).sum() > 1e-6:
            indices = np.random.choice(D, n_mask, replace=False)
            augmented[t, indices] = 0.0
    return augmented


def apply_augmentation(keypoints, p_temporal=0.5, p_jitter=0.8, p_mask=0.3):
    kp = keypoints.astype(np.float32)
    if np.random.random() < p_temporal:
        kp = temporal_scale(kp)
    if np.random.random() < p_jitter:
        kp = coordinate_jitter(kp)
    if np.random.random() < p_mask:
        kp = random_mask_keypoints(kp)
    return kp
