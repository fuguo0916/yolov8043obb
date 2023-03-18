from ultralytics import YOLO
import sys

# train
if __name__ == "__main__":
    model = YOLO(f"ultralytics/models/v8/yolov8s_noatt.yaml")
    model.train(**{"cfg": f"ultralytics/yolo/cfg/train.yaml",
                   "project": f"train_02_epochs_batch_imgsz",
                   "name": f"train_e300_iou3_b16_i640_valriou_noatt",
                   "epochs": 300,
                   "batch": 16,
                   "imgsz": 640})

    model2 = YOLO(f"ultralytics/models/v8/yolov8s_noatt.yaml")
    model2.train(**{"cfg": f"ultralytics/yolo/cfg/train.yaml",
                    "project": f"train_02_epochs_batch_imgsz",
                   "name": f"train_e300_iou3_b4_i1280_valriou_noatt",
                   "epochs": 300,
                   "batch": 4,
                   "imgsz": 1280})

    # model3 = YOLO(f"ultralytics/models/v8/yolov8s_noatt.yaml")
    # model3.train(**{"cfg": f"ultralytics/yolo/cfg/train.yaml",
    #                 "project": f"train_02_epochs_batch_imgsz",
    #                "name": f"train_e500_iou3_b16_i640_valriou_noatt",
    #                "epochs": 500,
    #                "batch": 16,
    #                "imgsz": 640})

    # model4 = YOLO(f"ultralytics/models/v8/yolov8s_noatt.yaml")
    # model4.train(**{"cfg": f"ultralytics/yolo/cfg/train.yaml",
    #                 "project": f"train_02_epochs_batch_imgsz",
    #                "name": f"train_e500_iou3_b4_i1280_valriou_noatt",
    #                "epochs": 500,
    #                "batch": 4,
    #                "imgsz": 1280})

