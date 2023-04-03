from ultralytics import YOLO
import sys
import os

args2 = [
    {
        "model": f"ultralytics/models/v8/yolov8s_noatt.yaml",
        "weight": "yolov8s.pt",
    },
    {
        "cfg": f"ultralytics/yolo/cfg/train.yaml",
        "project": f"runs/train_11_clip",
        "epochs": 800,
        "iou": 0.3,
        "batch": 16,
        "imgsz": 640,
        "close_mosaic": 100,
        "patience": 1000,
        "name": "t11_e800_iou3_b16_i640_miou_valriou_noatt_close100",
    },
]

args = args2

model = YOLO(**args[0])
model.train(**args[1])