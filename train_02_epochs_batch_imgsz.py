from ultralytics import YOLO
import sys

# train
if __name__ == "__main__":
    model = YOLO(f"ultralytics/models/v8/yolov8s_noatt.yaml")
    model.train(**{"cfg": f"ultralytics/yolo/cfg/train.yaml",
                   "project": f"runs/train_02_epochs_batch_imgsz",
                   "name": f"train_e300_iou3_b16_i640_valriou_noatt",
                   "epochs": 300,
                   "batch": 16,
                   "imgsz": 640})

    model = YOLO(f"ultralytics/models/v8/yolov8s_noatt.yaml")
    model.train(**{"cfg": f"ultralytics/yolo/cfg/train.yaml",
                    "project": f"runs/train_02_epochs_batch_imgsz",
                   "name": f"train_e300_iou3_b4_i1280_valriou_noatt",
                   "epochs": 300,
                   "batch": 4,
                   "imgsz": 1280})

    model = YOLO(f"ultralytics/models/v8/yolov8s_noatt.yaml")
    model.train(**{"cfg": f"ultralytics/yolo/cfg/train.yaml",
                    "project": f"runs/train_02_epochs_batch_imgsz",
                   "name": f"train_e400_iou3_b4_i1280_valriou_noatt_close20",
                   "epochs": 400,
                   "batch": 4,
                   "imgsz": 1280,
                   "close_mosaic": 20})

    model = YOLO(f"ultralytics/models/v8/yolov8s_noatt.yaml")
    model.train(**{"cfg": f"ultralytics/yolo/cfg/train.yaml",
                    "project": f"runs/train_02_epochs_batch_imgsz",
                   "name": f"train_e500_iou3_b4_i1280_valriou_noatt_close20",
                   "epochs": 500,
                   "batch": 4,
                   "imgsz": 1280,
                   "close_mosaic": 20})


    model = YOLO(f"ultralytics/models/v8/yolov8s_noatt.yaml")
    model.train(**{"cfg": f"ultralytics/yolo/cfg/train.yaml",
                    "project": f"runs/train_02_epochs_batch_imgsz",
                   "name": f"train_e1000_iou3_b8_i1280_valriou_noatt_close50",
                   "epochs": 1000,
                   "batch": 8,
                   "imgsz": 1280,
                   "close_mosaic": 50})

    model = YOLO(f"ultralytics/models/v8/yolov8s_noatt.yaml")
    model.train(**{"cfg": f"ultralytics/yolo/cfg/train.yaml",
                    "project": f"runs/train_02_epochs_batch_imgsz",
                   "name": f"train_e1000_iou3_b4_i1280_valriou_noatt_close50",
                   "epochs": 1000,
                   "batch": 4,
                   "imgsz": 1280,
                   "close_mosaic": 50,
                   "patience": 200})

