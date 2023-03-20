from ultralytics import YOLO
import sys

# train
if __name__ == "__main__":
    # for name in ["noatt", "CBAM", "SE", "GC", "SK"]:
    #     model = YOLO(f"ultralytics/models/v8/yolov8s_{name}.yaml")
    #     model.train(**{"cfg": f"ultralytics/yolo/cfg/train.yaml",
    #                 "project": f"runs/train_03_att",
    #                 "name": f"train_e1000_iou3_b8_i960_valriou_{name}_close50",
    #                 "epochs": 1000,
    #                 "batch": 8,
    #                 "imgsz": 960,
    #                 "close_mosaic": 50,
    #                 "patience": 200})

    for name in ["noatt", "CBAM", "SE", "GC", "SK"]:
        model = YOLO(f"ultralytics/models/v8/yolov8s_{name}.yaml")
        model.train(**{"cfg": f"ultralytics/yolo/cfg/train.yaml",
                    "project": f"runs/train_03_att",
                    "name": f"train_e1000_iou3_b16_i960_valriou_{name}_close100",
                    "epochs": 1000,
                    "batch": 16,
                    "imgsz": 960,
                    "close_mosaic": 50,
                    "patience": 200})
