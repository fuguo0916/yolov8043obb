from ultralytics import YOLO
import sys
import os

for imgsz, batch in zip([320, 640, 960, 1280], [16, 16, 16, 8]):
        if imgsz == 640:
            continue
        os.system(f"echo train_e1000_iou3_b{batch}_i{imgsz}_kfiou_valriou_noatt_close200 >> whisper.txt")
        model = YOLO(f"ultralytics/models/v8/yolov8s_noatt.yaml")
        model.train(**{"cfg": f"ultralytics/yolo/cfg/train.yaml",
                    "project": f"runs/train_07_imgsz",
                    "name": f"train_e1000_iou3_b{batch}_i{imgsz}_kfiou_valriou_noatt_close200",
                    "epochs": 1000,
                    "batch": batch,
                    "imgsz": imgsz,
                    "close_mosaic": 200,
                    "patience": 1000})
