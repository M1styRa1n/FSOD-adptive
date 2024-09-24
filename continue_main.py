from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import default_argument_parser, launch
from detectron2.evaluation import verify_results
from detectron2.utils import comm

import src.data
import src.modeling
from src.engine import Trainer_teacher
from src.utils import setup

import inspect
import distutils.version


def main(args):
    cfg = setup(args)
    
    # ??????
    teacher_model = Trainer_teacher.build_model(cfg)
    DetectionCheckpointer(teacher_model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        "/users/acr23hk/paper/fsod-dc/checkpoints/voc/1726573001/fsod1/5shot/seed1/model_final.pth", resume=False
    )

    # ????????
    for param in teacher_model.parameters():
        param.requires_grad = False  # ??????,????

    if args.eval_only:
        model = Trainer_teacher.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer_teacher.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer_teacher(cfg, teacher_model=teacher_model)  # ???????? Trainer
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