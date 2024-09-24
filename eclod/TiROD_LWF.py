import argparse
import os
import warnings
import yaml
import sys
import shutil
sys.path.append('/home/pasti/PycharmProjects/Robot_CLOD/')

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.accelerators import find_usable_cuda_devices
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from nanodet.data.collate import naive_collate
from nanodet.data.dataset import build_dataset
from nanodet.evaluator import build_evaluator
from nanodet.trainer.task import TrainingTask
from nanodet.trainer.LwF_task import DistTrainingTask as LwFTask
from torchvision.transforms import ToTensor, ToPILImage
from nanodet.util import (
    NanoDetLightningLogger,
    cfg,
    convert_old_model,
    env_utils,
    load_config,
    load_model_weight,
    mkdir,
)

class CustomConcatDataset(ConcatDataset):
    def __init__(self, datasets):
        super().__init__(datasets)
        # Ensure all datasets have the same attributes
        self.coco_api = datasets[0].coco_api
        self.cat_ids = datasets[0].cat_ids
        self.all_cat_ids = datasets[0].cat_ids
        self.class_names = datasets[0].class_names

#Function to create the task configuration file required for training
def create_exp_cfg(yml_path, task, tirod_task):
    #Load the YAML file
    with open(yml_path, 'r') as file:
        temp_cfg = yaml.safe_load(file)
    #Save dir of the model
    temp_cfg['save_dir'] = 'models/' + tirod_task + 'task' + str(task)
    
    if task == 2:
        temp_cfg['data']['train']['img_path'] = '/home/pasti/Dataset/TiROD/Domain1/Low/images/train'
        temp_cfg['data']['val']['img_path'] = '/home/pasti/Dataset/TiROD/Domain1/Low/images/test'
        temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/TiROD/Domain1/Low/annotations/train.json'
        temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/TiROD/Domain1/Low/annotations/test.json'
    elif task == 3:
        temp_cfg['data']['train']['img_path'] = '/home/pasti/Dataset/TiROD/Domain2/High/images/train'
        temp_cfg['data']['val']['img_path'] = '/home/pasti/Dataset/TiROD/Domain2/High/images/test'
        temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/TiROD/Domain2/High/annotations/train.json'
        temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/TiROD/Domain2/High/annotations/test.json'        
    elif task == 4:
        temp_cfg['data']['train']['img_path'] = '/home/pasti/Dataset/TiROD/Domain2/Low/images/train'
        temp_cfg['data']['val']['img_path'] = '/home/pasti/Dataset/TiROD/Domain2/Low/images/test'
        temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/TiROD/Domain2/Low/annotations/train.json'
        temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/TiROD/Domain2/Low/annotations/test.json'
    elif task == 5:
        temp_cfg['data']['train']['img_path'] = '/home/pasti/Dataset/TiROD/Domain3/High/images/train'
        temp_cfg['data']['val']['img_path'] = '/home/pasti/Dataset/TiROD/Domain3/High/images/test'
        temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/TiROD/Domain3/High/annotations/train.json'
        temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/TiROD/Domain3/High/annotations/test.json'
    elif task == 6:
        temp_cfg['data']['train']['img_path'] = '/home/pasti/Dataset/TiROD/Domain3/Low/images/train'
        temp_cfg['data']['val']['img_path'] = '/home/pasti/Dataset/TiROD/Domain3/Low/images/test'
        temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/TiROD/Domain3/Low/annotations/train.json'
        temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/TiROD/Domain3/Low/annotations/test.json'
    elif task == 7:
        temp_cfg['data']['train']['img_path'] = '/home/pasti/Dataset/TiROD/Domain4/High/images/train'
        temp_cfg['data']['val']['img_path'] = '/home/pasti/Dataset/TiROD/Domain4/High/images/test'
        temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/TiROD/Domain4/High/annotations/train.json'
        temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/TiROD/Domain4/High/annotations/test.json'
    elif task == 8:
        temp_cfg['data']['train']['img_path'] = '/home/pasti/Dataset/TiROD/Domain4/Low/images/train'
        temp_cfg['data']['val']['img_path'] = '/home/pasti/Dataset/TiROD/Domain4/Low/images/test'
        temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/TiROD/Domain4/Low/annotations/train.json'
        temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/TiROD/Domain4/Low/annotations/test.json'
    elif task == 9:
        temp_cfg['data']['train']['img_path'] = '/home/pasti/Dataset/TiROD/Domain5/High/images/train'
        temp_cfg['data']['val']['img_path'] = '/home/pasti/Dataset/TiROD/Domain5/High/images/test'
        temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/TiROD/Domain5/High/annotations/train.json'
        temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/TiROD/Domain5/High/annotations/test.json'
    elif task == 10:
        temp_cfg['data']['train']['img_path'] = '/home/pasti/Dataset/TiROD/Domain5/Low/images/train'
        temp_cfg['data']['val']['img_path'] = '/home/pasti/Dataset/TiROD/Domain5/Low/images/test'
        temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/TiROD/Domain5/Low/annotations/train.json'
        temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/TiROD/Domain5/Low/annotations/test.json'
    
    if task > 1:
        temp_cfg['schedule']['load_model'] = 'models/' + tirod_task + 'task' + str(task-1) + '/model_last.ckpt'
        
    temp_cfg_name = 'cfg/' + tirod_task +'task' + str(task) + '.yml'
    #Save the new configuration file
    with open(temp_cfg_name, 'w') as file:
        yaml.safe_dump(temp_cfg, file)

#Set logger and seed
logger = NanoDetLightningLogger('test')
pl.seed_everything(1234)

#Define the parser to set the CL experience
parser = argparse.ArgumentParser(description="Parser for training task")
parser.add_argument('--TiROD_task', type=str, help='Type of TiROD analysis (domain)', required=True)
parser.add_argument('--cfg', type=str, help='Path to the configuration file', required=True)
parser.add_argument('--gpu' , type=int, help='GPU to use', required=True)
args = parser.parse_args()


val_datasets = []

#Start CL experience
for task in range (1, 11):
    
    #Create the task configuration file based on the task number and load the configuration
    create_exp_cfg(args.cfg, task, args.TiROD_task)
    load_config(cfg, 'cfg/'+ args.TiROD_task +'task' + str(task) + '.yml')
    
    logger = NanoDetLightningLogger('run_logs/task'+str(task))
    logger.info("TiROD analysis: " + args.TiROD_task)
    logger.info(cfg['data']['train']['img_path'])
    logger.info("Starting task" + str(task))
    logger.info("Setting up data...")

    #Data set up
    train_dataset = build_dataset(cfg.data.train, "train")
    val_dataset = build_dataset(cfg.data.val, "val")
    val_datasets.append(val_dataset)

    evaluator = build_evaluator(cfg.evaluator, val_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=False,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=False,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=False,
    )

    #Create the model based on the task configuration file
    logger.info("Creating model...")
    if task == 1:
        TrainTask = TrainingTask(cfg, evaluator)
    else:
        TrainTask = LwFTask(cfg, 1.0, evaluator)
        ckpt = torch.load(cfg.schedule.load_model if task > 2 else 'baseModels/CUMULtask1/model_last.ckpt')
        load_model_weight(TrainTask.model, ckpt, logger)
        load_model_weight(TrainTask.teacher, ckpt, logger)
        logger.info("Loaded model weight!")
        logger.info("Loaded also teacher!")

    model_resume_path = (
        os.path.join(cfg.save_dir, "model_last.ckpt")
        if "resume" in cfg.schedule
        else None
    )
    #Set the device to GPU if available
    if cfg.device.gpu_ids == -1:
        logger.info("Using CPU training")
        accelerator, devices, strategy, precision = (
            "cpu",
            None,
            None,
            cfg.device.precision,
        )
    else:
        accelerator, devices, strategy, precision = (
            "gpu",
            cfg.device.gpu_ids,
            None,
            cfg.device.precision,
        )

    if devices and len(devices) > 1:
        strategy = "ddp"
        env_utils.set_multi_processing(distributed=True)
    if task > 1:
        trainer = pl.Trainer(
            default_root_dir=cfg.save_dir,
            max_epochs=cfg.schedule.total_epochs,
            check_val_every_n_epoch=cfg.schedule.val_intervals,
            accelerator=accelerator,
            devices=[args.gpu],
            log_every_n_steps=cfg.log.interval,
            num_sanity_val_steps=0,
            callbacks=[TQDMProgressBar(refresh_rate=0)],
            logger=logger,
            benchmark=cfg.get("cudnn_benchmark", True),
            gradient_clip_val=cfg.get("grad_clip", 0.0),
            strategy=strategy,
            precision=precision,
        )
        trainer.fit(TrainTask, train_dataloader, val_dataloader, ckpt_path=model_resume_path)
        'Save only the student model'
        state_dict = TrainTask.model.state_dict()
        new_state_dict = {"model." + k: v for k, v in state_dict.items() if "teacher" not in k}

        torch.save({'state_dict': new_state_dict}, 'models/' + args.TiROD_task + 'task' + str(task) + '/model_last.ckpt')
    if task == 1:
        src = '/home/pasti/PycharmProjects/Robot_CLOD/eclod/baseModels/CUMULtask1'
        dst = '/home/pasti/PycharmProjects/Robot_CLOD/eclod/models/' + args.TiROD_task + 'task1'
        shutil.copytree(src, dst)
    ####END OF TRAINING
    
    ###TESTING ON ALL TASK DATASETS
    if task > 1:
        i = 1
        logger.log("Finished task" + str(task) + " training")
        logger.log("-----STARTING TESTING ON ALL TASK DATASETS-----")
        for test_dataset in val_datasets:
            load_config(cfg, 'cfg/'+ args.TiROD_task +'task' + str(i) + '.yml')
            cfg.defrost()
            cfg.update({"test_mode": 'val'})
            logger.log("Testing on task: " + str(i))
            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=cfg.device.batchsize_per_gpu,
                shuffle=False,
                num_workers=cfg.device.workers_per_gpu,
                pin_memory=True,
                collate_fn=naive_collate,
                drop_last=False,
            )
            evaluator = build_evaluator(cfg.evaluator, test_dataset)

            logger.info("Creating model...")
            TestTask = TrainingTask(cfg, evaluator)
            ckpt = torch.load('models/' + args.TiROD_task + 'task' + str(task) + '/model_last.ckpt')
            TestTask.load_state_dict(ckpt["state_dict"])

            if cfg.device.gpu_ids == -1:
                logger.info("Using CPU")
                accelerator, devices = "cpu", None
            else:
                accelerator, devices = "gpu", cfg.device.gpu_ids

            trainer = pl.Trainer(
                default_root_dir=cfg.save_dir,
                accelerator=accelerator,
                devices=[args.gpu],
                log_every_n_steps=1,
                num_sanity_val_steps=0,
                logger=logger,
            )
            results = trainer.test(TestTask, test_dataloader)
            i += 1
