from ultralytics import YOLO
import sys

# train
if __name__ == "__main__":
    for name in ["CBAM", "SE", "GC", "SK"]:
        model = YOLO(f"ultralytics/models/v8/yolov8s_{name}.yaml")
        model.train(**{"cfg": f"ultralytics/yolo/cfg/train.yaml", "name": f"train_e150_iou3_b16_valriou_{name}"})

    # for name in ["CBAM", "SE", "GC", "SK", "noatt"]:
    #     model = YOLO(f"ultralytics/models/v8/yolov8s_{name}.yaml")
    #     model.train(**{"cfg": f"ultralytics/yolo/cfg/train.yaml", "name": f"train_e300_iou3_b16_i640_valriou_{name}"})

    # model = YOLO("runs/detect/train4/weights/best.pt")
    # model.val(imgsz=640, save_json=False, iou=0.7)

    # model = YOLO("runs/detect/train9/weights/last.pt")
    # model.predict(source=r"E:\data\aiprj\yolov8043obb\mydata\test.txt", conf=0.25, iou=0.5, save_txt=True, save=True)
