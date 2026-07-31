"""L1290 C382 手势手语数据集 YOLOv8s 检测训练。

用法：
    python -u train_l1290.py > train_l1290.log 2>&1

说明：
- 用 yolov8s.yaml 结构 + COCO 预训练权重（pretrained=True）
- imgsz=640（原图 640x480）
- batch=16（RTX 5060 Laptop 8.5GB，若 OOM 降到 8）
- workers=0（Windows 兼容）
- 静态手势分类用途：只关心每张图里的手势检测框类别，验证看 mAP
"""
import os

from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    model = YOLO(os.path.join(HERE, "yolov8s.yaml"))
    model.train(
        data=os.path.join(HERE, "l1290_data.yaml"),
        epochs=100,
        batch=16,
        imgsz=640,
        pretrained=True,
        workers=0,
        project=os.path.join(HERE, "runs"),
        name="l1290",
    )


if __name__ == "__main__":
    main()
