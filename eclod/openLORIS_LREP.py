import argparse
import os
import warnings
import yaml
import sys
sys.path.append('/home/pasti/PycharmProjects/Robot_CLOD/')

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.accelerators import find_usable_cuda_devices
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader, Subset
from nanodet.data.collate import naive_collate
from nanodet.data.dataset import build_dataset
from nanodet.evaluator import build_evaluator
from nanodet.trainer.task import TrainingTask
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
import random

class ReplayDataLoader:
    def __init__(self, dataset1, dataset2, batch_size, shuffle):
        """
        This class is used to create a dataloader that creates batches
        using the task specific dataset and the replay buffer dataset.
        Batches are created using 50% of the task specific dataset and 50% of the replay buffer dataset.
        The iterators are reset when the task specific dataset is exhausted.
        If the replay buffer dataset is exhausted before the task specific dataset, its iterator is reset.
        
        Args:
            dataset1: task specific dataset
            dataset2: replay buffer dataset
            batch_size: Batch size
            shuffle: Whether to shuffle the data
        """
        
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.dataset1_loader = torch.utils.data.DataLoader(
            self.dataset1,
            batch_size=self.batch_size-2,
            shuffle=self.shuffle,
            num_workers=8,
            pin_memory=True,
            collate_fn=naive_collate,
            drop_last=True,
        )
        self.dataset2_loader = torch.utils.data.DataLoader(
            self.dataset2,
            batch_size=2,
            shuffle=self.shuffle,
            num_workers=8,
            pin_memory=True,
            collate_fn=naive_collate,
            drop_last=True,
        )
    
    def __iter__(self):
        self.dataset1_iter = iter(self.dataset1_loader)
        self.dataset2_iter = iter(self.dataset2_loader)
        return self
    
    def __next__(self):
        try:
            batch1 = next(self.dataset1_iter)
        except StopIteration:
            raise StopIteration
        try:
            batch2 = next(self.dataset2_iter)
        except StopIteration:
            self.dataset2_iter = iter(self.dataset2_loader)
            batch2 = next(self.dataset2_iter)
            
        merged_batch = {}
        for key in batch1.keys():
            if key == 'img':
                merged_batch[key] = batch1[key] + batch2[key]
            elif key == 'img_info':
                merged_batch[key] = {k: batch1[key][k] + batch2[key][k] for k in batch1[key]}
            elif key in ['gt_bboxes', 'gt_labels', 'gt_bboxes_ignore', 'warp_matrix']:
                #merged_batch[key] = [torch.cat((torch.tensor(b1), torch.tensor(b2))) for b1, b2 in zip(batch1[key], batch2[key])]
                merged_batch[key] = batch1[key] + batch2[key]
            else:
                raise ValueError(f"Key not recognized")
        
        return merged_batch
    
    def __len__(self):
        return len(self.dataset1)
        
class StandardBufferDataset(Dataset):
    """
    This class is used to create a replay buffer dataset that stores random samples of the task specific dataset.
    At init it takes random samples of the first task dataset to fill the buffer.
    Then from task n to task n+1 it, where n>0, it updates 50% of the buffer with the new task dataset.
    
    Args:
        dataset_n: task specific dataset
        buffer_size: size of the replay buffer
    """
    def __init__(self, dataset_n, buffer_size=250):
        self.buffer_size = buffer_size
        #At initialization, take random samples of the task 0 dataset to fill the buffer
        self.buffer_indices = random.sample(range(len(dataset_n)), self.buffer_size)
        self.buffer_dataset = Subset(dataset_n, self.buffer_indices)

    def __getitem__(self, buff_index):
        #Just return a buffer item at the index
        return self.buffer_dataset[buff_index]

    def __len__(self):
        #Return the buffer size
        return self.buffer_size

    def update_buffer(self, dataset_np1):
        #Take a random subset of the old buffer
        if len(dataset_np1) < int(self.buffer_size/2):
            update_buffer_indices = random.sample(range(self.buffer_size), self.buffer_size - len(dataset_np1))
            subset_n = Subset(self.buffer_dataset, update_buffer_indices)
            
            self.buffer_dataset = torch.utils.data.ConcatDataset([subset_n, dataset_np1])        
        else:
            update_buffer_indices = random.sample(range(self.buffer_size), int(self.buffer_size/2))
            subset_n = Subset(self.buffer_dataset, update_buffer_indices)
    
            #Take a random subset of the new task dataset
            new_rand_indices = random.sample(range(len(dataset_np1)), int(self.buffer_size/2))
            subset_np1 = Subset(dataset_np1, new_rand_indices)

            #Concate the two subsets to form the new buffer
            self.buffer_dataset = torch.utils.data.ConcatDataset([subset_n, subset_np1])
    def smart_update_buffer(self, dataset_np1, task):
            #Keep task-1/task of the buffer
            #replace 1/task of the buffer with the new task dataset
            keep_samples = int(int(self.buffer_size*(task-1)/task))
            update_buffer_indices = random.sample(range(self.buffer_size), keep_samples)
            subset_n = Subset(self.buffer_dataset, update_buffer_indices)
    
            #Take a random subset of the new task dataset
            new_rand_indices = random.sample(range(len(dataset_np1)), self.buffer_size - keep_samples)
            subset_np1 = Subset(dataset_np1, new_rand_indices)

            #Concate the two subsets to form the new buffer
            self.buffer_dataset = torch.utils.data.ConcatDataset([subset_n, subset_np1])  


#Function to create the task configuration file required for training
def create_exp_cfg(yml_path, task, Loris_task, factor=None):
    #Load the YAML file
    with open(yml_path, 'r') as file:
        temp_cfg = yaml.safe_load(file)
    #Save dir of the model
    temp_cfg['save_dir'] = 'models/' + Loris_task + factor + 'task' + str(task)
    
    #Define parameters based on the CL task
    if Loris_task == 'sequential':
        
        if task < 4:
            temp_cfg['data']['train']['img_path'] = '/home/pasti/Dataset/openLORIS/images1/train/illumination'
            temp_cfg['data']['val']['img_path'] = '/home/pasti/Dataset/openLORIS/images1/test/illumination'
            if task == 1:
                temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/illumination/Strong'
                temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/illumination/Strong'
            elif task == 2:
                temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/illumination/Normal'
                temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/illumination/Normal'
            elif task == 3:
                temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/illumination/Weak'
                temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/illumination/Weak'
        
        elif task >= 4 & task < 7:
            temp_cfg['data']['train']['img_path'] = '/home/pasti/Dataset/openLORIS/images1/train/occlusion'
            temp_cfg['data']['val']['img_path'] = '/home/pasti/Dataset/openLORIS/images1/test/occlusion'
            if task == 4:
                temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/occlusion/0%'
                temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/occlusion/0%'
            elif task == 5:
                temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/occlusion/25%'
                temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/occlusion/25%'
            elif task == 6:
                temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/occlusion/50%'
                temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/occlusion/50%'
        
        elif task >= 7 & task < 10:
            temp_cfg['data']['train']['img_path'] = '/home/pasti/Dataset/openLORIS/images1/train/pixel'
            temp_cfg['data']['val']['img_path'] = '/home/pasti/Dataset/openLORIS/images1/test/pixel'
            if task == 7:
                temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/pixel/200'
                temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/pixel/200'
            elif task == 8:
                temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/pixel/30-200'
                temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/pixel/30-200'
            elif task == 9:
                temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/pixel/30'
                temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/pixel/30'

        elif task >= 10 & task < 13:
            temp_cfg['data']['train']['img_path'] = '/home/pasti/Dataset/openLORIS/images1/train/clutter'
            temp_cfg['data']['val']['img_path'] = '/home/pasti/Dataset/openLORIS/images1/test/clutter'
            if task == 10:
                temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/clutter/Low'
                temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/clutter/Low'
            elif task == 11:
                temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/clutter/Normal'
                temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/clutter/Normal'
            elif task == 12:
                temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/clutter/High'
                temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/clutter/High'

    if Loris_task == 'single':
        temp_cfg['data']['train']['img_path'] = '/home/pasti/Dataset/openLORIS/images1/train/' + factor + '/segment' + str(task)
        temp_cfg['data']['val']['img_path'] = '/home/pasti/Dataset/openLORIS/images1/test/' + factor + '/segment' + str(task)
        if task < 4:
            if factor == 'illumination':
                temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/illumination/Strong'
                temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/illumination/Strong'
            elif factor == 'occlusion':
                temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/occlusion/0%'
                temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/occlusion/0%'
            elif factor == 'pixel':
                temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/pixel/200'
                temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/pixel/200'
            elif factor == 'clutter':
                temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/clutter/Low'
                temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/clutter/Low'
        elif task >= 4 & task < 7:
            if factor == 'illumination':
                temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/illumination/Normal'
                temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/illumination/Normal'
            elif factor == 'occlusion':
                temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/occlusion/25%'
                temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/occlusion/25%'
            elif factor == 'pixel':
                temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/pixel/30-200'
                temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/pixel/30-200'
            elif factor == 'clutter':
                temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/clutter/Normal'
                temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/clutter/Normal'           
        elif task >= 7 & task < 10:
            if factor == 'illumination':
                temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/illumination/Weak'
                temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/illumination/Weak'
            elif factor == 'occlusion':
                temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/occlusion/50%'
                temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/occlusion/50%'
            elif factor == 'pixel':
                temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/pixel/30'
                temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/pixel/30'
            elif factor == 'clutter':
                temp_cfg['data']['train']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/clutter/High'
                temp_cfg['data']['val']['ann_path'] = '/home/pasti/Dataset/openLORIS/annotations/mask_and_bbox/scence#1/clutter/High'  
    if task > 1:
        temp_cfg['schedule']['load_model'] = 'models/' + Loris_task + factor + 'task' + str(task-1) + '/model_last.ckpt'
        
    temp_cfg_name = 'cfg/' + factor +'task' + str(task) + '.yml'
    #Save the new configuration file
    with open(temp_cfg_name, 'w') as file:
        yaml.safe_dump(temp_cfg, file)

#Set logger and seed
logger = NanoDetLightningLogger('test')
pl.seed_everything(1234)

#Define the parser to set the CL experience
#e.g. python openLORIS.py --Loris_task single --cfg '../cfg/LORIS.yml' --factor illumination
parser = argparse.ArgumentParser(description="Parser for training task")
parser.add_argument('--Loris_task', type=str, help='Type of Loris analysis (sequential or single)', required=True)
parser.add_argument('--cfg', type=str, help='Path to the configuration file', required=True)
parser.add_argument('--factor', type=str, help='Factor to inspect for single factor analysis', required=False)
parser.add_argument('--gpu' , type=int, help='GPU to use', required=True)
args = parser.parse_args()

if args.Loris_task == 'sequential':
    total_tasks = 12
elif args.Loris_task == 'single':
    total_tasks = 9
else:
    raise ValueError('Invalid Loris task')

val_datasets = []

#Start CL experience
for task in range (1, total_tasks+1):
    
    #Create the task configuration file based on the task number and load the configuration
    create_exp_cfg(args.cfg, task, args.Loris_task, args.factor)
    load_config(cfg, 'cfg/'+ args.factor +'task' + str(task) + '.yml')
    
    logger = NanoDetLightningLogger('run_logs/task'+str(task))
    logger.info("LORIS analysis: " + args.Loris_task)
    if args.Loris_task == 'single':
        logger.info("Factor : " + args.factor)
    logger.info(cfg['data']['train']['img_path'])
    logger.info("Starting task" + str(task))
    logger.info("Setting up data...")

    #Data set up
    train_dataset = build_dataset(cfg.data.train, "train")
    val_dataset = build_dataset(cfg.data.val, "val")
    val_datasets.append(val_dataset)

    evaluator = build_evaluator(cfg.evaluator, val_dataset)
    if task == 1:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.device.batchsize_per_gpu,
            shuffle=False,
            num_workers=cfg.device.workers_per_gpu,
            pin_memory=True,
            collate_fn=naive_collate,
            drop_last=True,
        )
    else:
        train_dataloader = ReplayDataLoader(
            train_dataset, 
            buffer_dataset, 
            cfg.device.batchsize_per_gpu,
            shuffle = True
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
    logger.info("Creating model")
    
    TrainTask = TrainingTask(cfg, evaluator)
    # FOR LATENT REPLAY
    if task > 1:
        for param in TrainTask.model.backbone.parameters():
            param.requires_grad = False
    if task > 1:
        ckpt = torch.load(cfg.schedule.load_model)
        load_model_weight(TrainTask.model, ckpt, logger)
        logger.info("Loaded model weight from {}".format(cfg.schedule.load_model))
    
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
    
    #Replay code
    #If task is 1, initialize the replay buffer with the task 0 dataset 
    #if task > 1 update the buffer with the new task dataset    
    if task == 1:
        print("Creating buffer dataset")
        buffer_dataset = StandardBufferDataset(train_dataset)
    else:
        print("Updating buffer dataset")
        buffer_dataset.smart_update_buffer(train_dataset, task)
    ####END OF TRAINING
    
    ###TESTING ON ALL TASK DATASETS
    if task > 0:
        i = 1
        logger.log("Finished task" + str(task) + " training")
        logger.log("-----STARTING TESTING ON ALL TASK DATASETS-----")
        for test_dataset in val_datasets:
            load_config(cfg, 'cfg/'+ args.factor +'task' + str(i) + '.yml')
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
            ckpt = torch.load('models/' + args.Loris_task + args.factor + 'task' + str(task) + '/model_last.ckpt')
            TestTask.load_state_dict(ckpt["state_dict"])

            if cfg.device.gpu_ids == -1:
                logger.info("Using CPU training")
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
