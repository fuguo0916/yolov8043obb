import matplotlib.pyplot as plt
import pandas as pd

csvs2labels = {
    # "runs/train_03_att/train_e1000_iou3_b16_i960_valriou_noatt_close100/results.csv":       "i960(1000e, close100, MIoU, noatt)",
    # "runs/train_03_att/train_e1000_iou3_b16_i640_valriou_noatt_close100/results.csv":       "i640(1000e, close100, MIoU, noatt)",
    # "runs/train_03_att/train_e1200_iou3_b16_i960_valriou_noatt_close300/results.csv":       "i960(1200e, close300, MIoU, noatt)",
    # "runs/train_03_att/train_e1500_iou3_b16_i640_kfiou_valriou_noatt_close500/results.csv": "i640(1500e, close500, KFIoU, Early Stopped)",
    # "runs/train_03_att/train_e1500_iou3_b16_i960_kfiou_valriou_noatt_close500/results.csv": "i960(1500e, close500, KFIoU)",
    "runs/train_04_att/train_e1000_iou3_b16_i640_kfiou_valriou_noatt_close200/results.csv":     "i640(1000e, close200, KFIoU)",
    # "runs/train_04_att/train_e1000_iou3_b16_i640_kfiou_valriou_CBAM_close200/results.csv":      "i640(1000e, close200, KFIoU, CBAM)",
    # "runs/train_04_att/train_e1000_iou3_b16_i640_kfiou_valriou_SE_close200/results.csv":        "i640(1000e, close200, KFIoU, SE)",
    # "runs/train_04_att/train_e1000_iou3_b16_i640_kfiou_valriou_GC_close200/results.csv":        "i640(1000e, close200, KFIoU, GC)",
    # "runs/train_04_att/train_e1000_iou3_b16_i640_kfiou_valriou_SK_close200/results.csv":        "i640(1000e, close200, KFIoU, SK)",
    # "runs/train_06_epochs/train_e200_iou3_b16_i640_kfiou_valriou_noatt_close40/results.csv":    "i640(200e, close40, KFIoU)",
    # "runs/train_06_epochs/train_e500_iou3_b16_i640_kfiou_valriou_noatt_close100/results.csv":   "i640(500e, close100, KFIoU)",
    # "runs/train_06_epochs/train_e1500_iou3_b16_i640_kfiou_valriou_noatt_close300_amend/results.csv":  "i640(1500e, close300, KFIoU)",
    # "runs/train_05_close/train_e1000_iou3_b16_i640_kfiou_valriou_noatt_close10/results.csv":    "i640(1000e, close10, KFIoU)",
    # "runs/train_05_close/train_e1000_iou3_b16_i640_kfiou_valriou_noatt_close50/results.csv":    "i640(1000e, close50, KFIoU)",
    # "runs/train_05_close/train_e1000_iou3_b16_i640_kfiou_valriou_noatt_close100/results.csv":    "i640(1000e, close100, KFIoU)",
    # "runs/train_05_close/train_e1000_iou3_b16_i640_kfiou_valriou_noatt_close400/results.csv":    "i640(1000e, close400, KFIoU)",
    # "runs/train_07_imgsz/train_e1000_iou3_b16_i320_kfiou_valriou_noatt_close200/results.csv":   "i320(1000e, close200, KFIoU)",
    # "runs/train_07_imgsz/train_e1000_iou3_b16_i960_kfiou_valriou_noatt_close200_amend/results.csv":   "i960(1000e, close200, KFIoU)",
    # "runs/train_07_imgsz/train_e1000_iou3_b8_i1280_kfiou_valriou_noatt_close200/results.csv":  "i1280(1000e, close200, KFIoU)",
    "runs/train_08_miou/train_e1000_iou3_b16_i640_miou_valriou_noatt_close200/results.csv":     "i640(1000e, close200, MIoU)"
}

# img_caption = "train_06_epochs"
# img_path = "runs/train_06_epochs/plot.png"
# img_caption = "train_05_close"
# img_path = "runs/train_05_close/plot.png"
# img_caption = "train_07_imgsz"
# img_path = "runs/train_07_imgsz/plot.png"
img_caption = "train_08_miou"
img_path = "runs/train_08_miou/plot.png"

map50_95_lists = [[] for _ in range(len(csvs2labels))]

for map50_95_list, csv in zip(map50_95_lists, csvs2labels.keys()):
    f = open(csv, mode="r", encoding="utf8")
    for line in f.readlines():
        line.replace("\n", "")
        line.replace(" ", "")
        try:
            map50_95 = float(line.split(",")[7])
            map50_95_list.append(map50_95)
        except Exception as e:
            pass
    f.close()

for map50_95_list, label in zip(map50_95_lists, csvs2labels.values()):
    plt.plot(list(range(1, 1 + len(map50_95_list))), map50_95_list, label=label)

plt.xlabel("Epoch")
plt.ylabel("mAP50-95")
plt.title(img_caption)
plt.legend()
plt.savefig(img_path, format="png")
plt.savefig(img_path.replace(".png", ".svg"), format="svg")
