from ultralytics import YOLO
import sys

# train
if __name__ == "__main__":
    if len(sys.argv) == 1 or sys.argv[1] == "train":
        model = YOLO("ultralytics/models/v8/yolov8s_noatt.yaml")
        model.train(**{"cfg": "ultralytics/yolo/cfg/train.yaml"})
    elif sys.argv[1] == "val":
        model = YOLO("runs/train_11_clip/t11_e1000_iou3_b16_i640_miou_valriou_noatt_close200/weights/last.pt")
        for iou in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
            print(f"\nValidation at iou threshold {iou}.\n")
            model.val(imgsz=640, save_json=True, iou=iou, batch=4, workers=1)
    elif sys.argv[1] == "test":
        model = YOLO("runs/train_08_miou/trainl_e1000_iou3_b16_i640_miou_valriou_noatt_close200/weights/best.pt")
        model.predict(source="/root/autodl-tmp/yolov8043obb/mydata/test.mp4", conf=0.1, iou=0.3, save_txt=True, save=True)
    elif sys.argv[1] == "pretrain":
        model = YOLO("ultralytics/models/v8/yolov8s_noatt.yaml")
        model.train(**{"cfg": "ultralytics/yolo/cfg/train.yaml",
                       "data": "airbus.yaml",
                       "imgsz": 768,
                       "batch": 16,
                       "label_smoothing": 0.0,
                       "mosaic": 0.0,
                       "epoch": 100,
                       "project": "runs/pretrain",
                       "nc": 1,
                       })
    else:
        assert False, f"Wrong argument to main.py"
