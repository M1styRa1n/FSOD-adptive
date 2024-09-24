import logging
import os

from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer

from src.evaluation import COCOEvaluator, PascalVOCDetectionEvaluator
from src.solver import build_optimizer

__all__ = ["Trainer"]


def build_evaluator(cfg, dataset_name, output_folder=None):
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type == "coco":
        return COCOEvaluator(dataset_name, distributed=True, output_dir=output_folder)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    else:
        raise NotImplemented(
            f"no Evaluator for the dataset {dataset_name} with the type {evaluator_type}"
        )


class Trainer(DefaultTrainer):
    @classmethod
    def build_model(cls, cfg):
        logging.getLogger("detectron2.checkpoint").setLevel(logging.CRITICAL)
        model = build_model(cfg)  # ??????
        
        # ???????????
        teacher_model = build_model(cfg)
        DetectionCheckpointer(teacher_model).resume_or_load(cfg.MODEL.ROI_HEADS.TEACHER_WEIGHTS)
        teacher_model.eval()
        for param in teacher_model.parameters():
            param.requires_grad = False
        
        # ???????????????
        model.roi_heads.set_teacher_model(teacher_model)
        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def build_optimizer(cls, cfg, model):
        return build_optimizer(cfg, model)
