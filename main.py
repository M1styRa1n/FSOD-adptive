from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser, launch
from detectron2.evaluation import verify_results
from detectron2.utils import comm
import inspect
from packaging.version import Version
import distutils.version

import src.data
import src.modeling
from src.engine import Trainer
from src.utils import setup


def main(args):
    cfg = setup(args)
    
    # Build the student model (which will be trained)
    model = Trainer.build_model(cfg)

    # Always use a teacher model for self-distillation
    teacher_model = Trainer.build_model(cfg)
    DetectionCheckpointer(teacher_model).resume_or_load(cfg.MODEL.ROI_HEADS.TEACHER_WEIGHTS)
    
    # Freeze the teacher model (no gradient updates)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False


    # Set the teacher model inside the student model for distillation
    model.set_teacher_model(teacher_model)

    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


if __name__ == "__main__":
    print(inspect.getfile(distutils.version.LooseVersion))

    args = default_argument_parser().parse_args()
    launch(
        main,
        
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
