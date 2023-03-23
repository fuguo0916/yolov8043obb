from ultralytics import YOLO
import sys
import os

for epochs, close_mosaic in zip([200, 500, 1000, 1500], [40, 100, 200, 300]):
        if epochs == 1000:
            continue
        os.system(f"echo train_e{epochs}_iou3_b16_i640_kfiou_valriou_noatt_close{close_mosaic} >> whisper.txt")
        model = YOLO(f"ultralytics/models/v8/yolov8s_noatt.yaml")
        model.train(**{"cfg": f"ultralytics/yolo/cfg/train.yaml",
                    "project": f"runs/train_06_epochs",
                    "name": f"train_e{epochs}_iou3_b16_i640_kfiou_valriou_noatt_close{close_mosaic}",
                    "epochs": epochs,
                    "batch": 16,
                    "imgsz": 640,
                    "close_mosaic": close_mosaic,
                    "patience": 1000})
