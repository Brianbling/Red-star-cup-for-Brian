from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolov8s.yaml")
    model.train(
        data="mnist_data.yaml",
        epochs=50,
        batch=16,
        imgsz=64,  # MNIST原图28x28，缩小imgsz提升速度
        pretrained=True,
        workers=0
    )