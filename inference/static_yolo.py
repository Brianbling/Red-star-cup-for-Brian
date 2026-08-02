"""YOLO 静态手势旁路推理封装。

L1290 权重是 yolov8s 检测模型（35 类手势检测框），非分类模型。
本模块对每次推理只关心"整帧最高置信度检测框"，取其 class_id 映射为类名，
再做同类别连续 N 帧防抖，得到稳定的静态词输出。
"""

from __future__ import annotations

import time
from pathlib import Path

from ultralytics import YOLO
import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = ROOT / "YOLOv8/runs/l1290/weights/best.pt"
# l1290_data.yaml 的 path 字段指向数据集，与训练时一致；本文件仅用于读 names
DEFAULT_DATA_YAML = ROOT / "YOLOv8/l1290_data.yaml"
# yolov8s 检测模型原生 conf 阈值 0.25 太低，web 摄像头场景抬到 0.5 防误报
CONF_THRESHOLD = 0.5
DEFAULT_CONFIRM_FRAMES = 15  # 10fps 下约 1.5s 连续帧才确认一个静态词


def _load_class_names(data_yaml: str | Path | None) -> list[str]:
    """从 data.yaml 读 names。返回按类别 id 索引的列表。"""
    if data_yaml:
        with open(data_yaml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        names = cfg.get("names")
        if names:
            if isinstance(names, list):
                return list(names)
            if isinstance(names, dict):
                return [names[i] for i in sorted(int(k) for k in names)]
    raise ValueError(f"无法从 data.yaml 读取类名: {data_yaml}")


class StaticYOLO:
    """加载检测模型，提供带防抖的静态手势推理。线程安全：被单个后台推理线程独占调用。"""

    def __init__(
        self,
        weights: str | Path = DEFAULT_WEIGHTS,
        data_yaml: str | Path = DEFAULT_DATA_YAML,
        conf_threshold: float = CONF_THRESHOLD,
        confirm_frames: int = DEFAULT_CONFIRM_FRAMES,
    ):
        self.weights = str(weights)
        self.conf_threshold = conf_threshold
        self.confirm_frames = confirm_frames
        self.names = _load_class_names(data_yaml)
        self.model = YOLO(self.weights)
        if self.model.task != "detect":
            raise ValueError(f"模型任务应为 detect，实际为 {self.model.task}")
        self._current_cls: int | None = None
        self._count = 0

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def infer(self, frame_bgr) -> tuple[str | None, float | None]:
        """输入一帧 BGR 图，返回 (class_name, confidence)。

        同类别连续 >=confirm_frames 帧且置信度 >=conf_threshold 才输出词；
        未达标/类别切换/低于阈值时返回 (None, None)。
        """
        result = self.model.predict(frame_bgr, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            self._current_cls = None
            self._count = 0
            return None, None

        cls_id = int(result.boxes.cls[0].item())
        conf = float(result.boxes.conf[0].item())
        if cls_id != self._current_cls:
            self._current_cls = cls_id
            self._count = 0
        if conf < self.conf_threshold:
            self._count = 0
            return None, None
        self._count += 1
        if self._count >= self.confirm_frames:
            return self.names[cls_id], conf
        return None, None

    def reset(self) -> None:
        self._current_cls = None
        self._count = 0


if __name__ == "__main__":
    import cv2
    import sys

    # 冒烟自检：随机帧 + 一张真实数据集图
    yolo = StaticYOLO()
    print("model loaded, classes:", len(yolo.names))
    noisy = (__import__("numpy").random.rand(480, 640, 3) * 255).astype("uint8")
    print("noise frame ->", yolo.infer(noisy))
    yolo.reset()
    img = cv2.imread(sys.argv[1]) if len(sys.argv) > 1 else None
    if img is not None:
        print("real image ->", yolo.infer(img))
