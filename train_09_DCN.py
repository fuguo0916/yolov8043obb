from ultralytics import YOLO
import sys
import os

os.system(f"echo train_e1000_iou3_b16_i640_miou_valriou_noatt_close200 >> whisper.txt")
model = YOLO(f"ultralytics/models/v8/yolov8s_DCN.yaml")
model.train(**{"cfg": f"ultralytics/yolo/cfg/train.yaml",
            "project": f"runs/train_09_DCN",
            "name": f"trainl_e1000_iou3_b16_i640_miou_valriou_noatt_dcn_close200",
            "epochs": 1000,
            "batch": 16,
            "imgsz": 640,
            "close_mosaic": 200,
            "patience": 1000})