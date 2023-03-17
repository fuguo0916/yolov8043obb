from ultralytics import YOLO
import sys

# train
if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] == "train":
        model = YOLO("ultralytics/models/v8/yolov8s_att.yaml")
        model.train(**{"cfg": "ultralytics/yolo/cfg/train.yaml"})
    elif sys.argv[1] == "val":
        model = YOLO("runs/detect/train29/weights/best.pt")
        model.val(imgsz=640, save_json=False, iou=0.4, batch=4)
    elif sys.argv[1] == "test":
        model = YOLO("runs/detect/train9/weights/last.pt")
        model.predict(source="E:\\data\\aiprj\\yolov8043obb\\mydata\\test.txt", conf=0.25, iou=0.5, save_txt=True, save=True)
    else:
        assert False, f"Wrong argument to main.py: {sys.argv[1]}"
