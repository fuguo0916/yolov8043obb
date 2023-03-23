from ultralytics import YOLO
import sys

# train
if __name__ == "__main__":
    for name in ["noatt", "CBAM", "SE", "GC", "SK"]:
        model = YOLO(f"ultralytics/models/v8/yolov8s_{name}.yaml")
        model.train(**{"cfg": f"ultralytics/yolo/cfg/train.yaml",
                    "project": f"runs/train_03_att",
                    "name": f"train_e1000_iou3_b8_i960_valriou_{name}_close50",
                    "epochs": 1000,
                    "batch": 8,
                    "imgsz": 960,
                    "close_mosaic": 50,
                    "patience": 200})

    for name in ["noatt", "CBAM", "SE", "GC", "SK"]:
        model = YOLO(f"ultralytics/models/v8/yolov8s_{name}.yaml")
        model.train(**{"cfg": f"ultralytics/yolo/cfg/train.yaml",
                    "project": f"runs/train_03_att",
                    "name": f"train_e1000_iou3_b16_i960_valriou_{name}_close50",
                    "epochs": 1000,
                    "batch": 16,
                    "imgsz": 960,
                    "close_mosaic": 50,
                    "patience": 200})

    for name in ["noatt"]:
        model = YOLO(f"ultralytics/models/v8/yolov8s_{name}.yaml")
        model.train(**{"cfg": f"ultralytics/yolo/cfg/train.yaml",
                    "project": f"runs/train_03_att",
                    "name": f"train_e200_iou3_b16_i960_valriou_{name}_close20",
                    "epochs": 200,
                    "batch": 16,
                    "imgsz": 960,
                    "close_mosaic": 20,
                    "patience": 200})

    for name in ["noatt"]:
        model = YOLO(f"ultralytics/models/v8/yolov8s_{name}.yaml")
        model.train(**{"cfg": f"ultralytics/yolo/cfg/train.yaml",
                    "project": f"runs/train_03_att",
                    "name": f"train_e200_iou3_b16_i640_valriou_{name}_close20",
                    "epochs": 200,
                    "batch": 16,
                    "imgsz": 640,
                    "close_mosaic": 20,
                    "patience": 200})

    for name in ["noatt"]:
        model = YOLO(f"ultralytics/models/v8/yolov8s_{name}.yaml")
        model.train(**{"cfg": f"ultralytics/yolo/cfg/train.yaml",
                    "project": f"runs/train_03_att",
                    "name": f"train_e1000_iou3_b16_i640_valriou_{name}_close50",
                    "epochs": 1000,
                    "batch": 16,
                    "imgsz": 640,
                    "close_mosaic": 50,
                    "patience": 200})

    for name in ["noatt"]:
        model = YOLO(f"ultralytics/models/v8/yolov8s_{name}.yaml")
        model.train(**{"cfg": f"ultralytics/yolo/cfg/train.yaml",
                    "project": f"runs/train_03_att",
                    "name": f"train_e1000_iou3_b16_i640_valriou_{name}_close100",
                    "epochs": 1000,
                    "batch": 16,
                    "imgsz": 640,
                    "close_mosaic": 100,
                    "patience": 200})

    for name in ["noatt"]:
        model = YOLO(f"ultralytics/models/v8/yolov8s_{name}.yaml")
        model.train(**{"cfg": f"ultralytics/yolo/cfg/train.yaml",
                    "project": f"runs/train_03_att",
                    "name": f"train_e1000_iou3_b16_i960_valriou_{name}_close100",
                    "epochs": 1000,
                    "batch": 16,
                    "imgsz": 960,
                    "close_mosaic": 100,
                    "patience": 200})

    for name in ["noatt"]:
        model = YOLO(f"ultralytics/models/v8/yolov8s_{name}.yaml")
        model.train(**{"cfg": f"ultralytics/yolo/cfg/train.yaml",
                    "project": f"runs/train_03_att",
                    "name": f"train_e1200_iou3_b16_i960_valriou_{name}_close300",
                    "epochs": 1200,
                    "batch": 16,
                    "imgsz": 960,
                    "close_mosaic": 300,
                    "patience": 200})

    for name in ["noatt"]:
        model = YOLO(f"ultralytics/models/v8/yolov8s_{name}.yaml")
        model.train(**{"cfg": f"ultralytics/yolo/cfg/train.yaml",
                    "project": f"runs/train_03_att",
                    "name": f"train_e1500_iou3_b16_i640_kfiou_valriou_{name}_close500",
                    "epochs": 1500,
                    "batch": 16,
                    "imgsz": 640,
                    "close_mosaic": 500,
                    "patience": 200})

    for name in ["noatt"]:
        model = YOLO(f"ultralytics/models/v8/yolov8s_{name}.yaml")
        model.train(**{"cfg": f"ultralytics/yolo/cfg/train.yaml",
                    "project": f"runs/train_03_att",
                    "name": f"train_e1500_iou3_b16_i960_kfiou_valriou_{name}_close500",
                    "epochs": 1500,
                    "batch": 16,
                    "imgsz": 960,
                    "close_mosaic": 500,
                    "patience": 200})
