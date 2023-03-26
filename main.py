from ultralytics import YOLO
import sys

# train
if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] == "train":
        model = YOLO("ultralytics/models/v8/yolov8s_noatt.yaml")
        model.train(**{"cfg": "ultralytics/yolo/cfg/train.yaml"})
    elif sys.argv[1] == "val":
        model = YOLO("runs/train_02_epochs_batch_imgsz/train_e500_iou3_b4_i1280_valriou_noatt_close20/weights/last.pt")
        model.val(imgsz=1280, save_json=True, iou=0.3, batch=4)
    elif sys.argv[1] == "test":
        model = YOLO("runs/train_08_miou/train_e1000_iou3_b16_i640_miou_valriou_noatt_close200/weights/last.pt")
        model.predict(source="/root/autodl-tmp/yolov8043obb/mydata/test.mp4", conf=0.1, iou=0.3, save_txt=True, save=True)
    else:
        assert False, f"Wrong argument to main.py"
