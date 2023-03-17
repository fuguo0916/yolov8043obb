from ultralytics import YOLO
import sys
import os

for close_mosaic in [10, 50, 100, 200, 400]:
        if close_mosaic == 200:
            continue
        os.system(f"echo train_e1000_iou3_b16_i640_kfiou_valriou_noatt_close{close_mosaic} >> whisper.txt")
        model = YOLO(f"ultralytics/models/v8/yolov8s_noatt.yaml")
        model.train(**{"cfg": f"ultralytics/yolo/cfg/train.yaml",
                    "project": f"runs/train_05_close",
                    "name": f"train_e1000_iou3_b16_i640_kfiou_valriou_noatt_close{close_mosaic}",
                    "epochs": 1000,
                    "batch": 16,
                    "imgsz": 640,
                    "close_mosaic": close_mosaic,
                    "patience": 1200})
