import sys
import os

# ??????(? fsod-dc ???)??? Python ???
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

# ??????????
from src.modeling.rcnn import FsodRCNN
from src.utils import setup
from detectron2.engine import default_argument_parser

import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2

# ????? args ??
args = default_argument_parser().parse_args(args=[])
args.config_file = "/users/acr23hk/paper/fsod-dc/configs/voc/fsod.yaml"
args.opts = []
args.eval_only = False
args.resume = False

# ?????????????
cfg = setup(args)

# ??????
cfg.defrost()

# ???????????
cfg.MODEL.WEIGHTS = "/users/acr23hk/paper/fsod-dc/checkpoints/voc/1726531430/fsod1/2shot/seed1/model_final.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cpu"
cfg.DATASETS.TEST = ("voc_2007_test",)

# ????,??????????
# cfg.freeze()

# ?????(??????)
from detectron2.data.datasets import register_pascal_voc

'''register_pascal_voc(
    name="voc_2007_test",
    dirname="/mnt/parscratch/users/acr23hk/dataset/VOC2007",
    split="test",
    year="2007",
)'''

# ??????
MetadataCatalog.get("voc_2007_test").thing_classes = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
]

# ??????
predictor = DefaultPredictor(cfg)

# ?????????
img = cv2.imread("/mnt/parscratch/users/acr23hk/dataset/VOC2007/JPEGImages/000032.jpg")
outputs = predictor(img)

# ???????
v = Visualizer(img[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
out = v.draw_instance_predictions(outputs["instances"].to("cpu"))

# ??????
output_path = '/users/acr23hk/paper/fsod-dc/picture/1.jpg'
cv2.imwrite(output_path, out.get_image()[:, :, ::-1])

