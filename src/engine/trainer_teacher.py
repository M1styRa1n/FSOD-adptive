import logging
import os
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.modeling import build_model
from src.evaluation import COCOEvaluator, PascalVOCDetectionEvaluator
from src.solver import build_optimizer

__all__ = ["Trainer_teacher"]

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


class Trainer_teacher(DefaultTrainer):
    def __init__(self, cfg, teacher_model=None):
        super().__init__(cfg)
        self.teacher_model = teacher_model  # ??????
        
        # ?????????
        if self.teacher_model is not None:
            for param in self.teacher_model.parameters():
                param.requires_grad = False  # ??????,????

        # ????????????
        self._data_loader_iter = iter(self.train_loader)

    @classmethod
    def build_model(cls, cfg):
        # ????????
        logging.getLogger("detectron2.checkpoint").setLevel(logging.CRITICAL)
        model = build_model(cfg)
        return model

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def build_optimizer(cls, cfg, model):
        return build_optimizer(cfg, model)

    def run_step(self):
        """
        ?? DefaultTrainer ? run_step ??,?????????
        """
        assert self.model.training, "[Trainer_teacher] ?????????!"

        # ?????????
        data = next(self._data_loader_iter)

        # ???????,?????????
        teacher_features = None
        if self.teacher_model is not None:
            with torch.no_grad():  # ???????????
                teacher_features = self.teacher_model(data)

        # ?????????
        loss_dict = self.model(data, teacher_features=teacher_features)
        
        # ?????
        losses = sum(loss_dict.values())

        # ???????
        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

