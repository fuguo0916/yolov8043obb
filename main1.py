from ultralytics import YOLO
import sys

# train
if __name__ == "__main__":
    model = YOLO("ultralytics/models/v8/yolov8s.yaml")
    model.train(**{"cfg": f"ultralytics/yolo/cfg/{sys.argv[1]}"})

    # model = YOLO("runs/detect/train4/weights/best.pt")
    # model.val(imgsz=640, save_json=False, iou=0.7)

    # model = YOLO("runs/detect/train9/weights/last.pt")
    # model.predict(source=r"E:\data\aiprj\yolov8043obb\mydata\test.txt", conf=0.25, iou=0.5, save_txt=True, save=True)
