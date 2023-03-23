from ultralytics import YOLO
import sys

for name in ["noatt", "CBAM", "SE", "GC", "SK"]:
        model = YOLO(f"ultralytics/models/v8/yolov8s_{name}.yaml")
        model.train(**{"cfg": f"ultralytics/yolo/cfg/train.yaml",
                    "project": f"runs/train_04_att",
                    "name": f"train_e1000_iou3_b16_i640_kfiou_valriou_{name}_close200",
                    "epochs": 1000,
                    "batch": 16,
                    "imgsz": 640,
                    "close_mosaic": 200,
                    "patience": 200})
