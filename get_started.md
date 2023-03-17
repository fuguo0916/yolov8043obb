```shell
# for calculating rotated_iou for metrics
pip install -U openmim
mim install mmcv

# for nms
cd ultralytics/yolo/utils/nms_rotated/
python setup.py develop

# for drawing labels
pip install --upgrade pillow
```