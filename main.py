from ultralytics import YOLO
import sys

# train
if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] == "train":
        model = YOLO("ultralytics/models/v8/yolov8s_noatt.yaml")
        model.train(**{"cfg": "ultralytics/yolo/cfg/train.yaml"})
    elif sys.argv[1] == "val":
        model = YOLO("runs/train_02_epochs_batch_imgsz/train_e500_iou3_b4_i1280_valriou_noatt_close20/weights/last.pt")
        model.val(imgsz=1280, save_json=False, iou=0.3, batch=4)
    elif sys.argv[1] == "test":
        model = YOLO("runs/detect/train9/weights/last.pt")
        model.predict(source="E:\\data\\aiprj\\yolov8043obb\\mydata\\test.txt", conf=0.25, iou=0.5, save_txt=True, save=True)
    else:
        assert False, f"Wrong argument to main.py"
