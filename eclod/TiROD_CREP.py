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
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from nanodet.data.collate import naive_collate
from nanodet.data.dataset import build_dataset
from nanodet.evaluator import build_evaluator
from nanodet.trainer.task import TrainingTask
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

#SIMCLR imports
from torchvision import transforms
from simclr.models.encoder import EncodeProject
import simclr.models as models
from argparse import Namespace
import PIL
import torch.nn.functional as F
from simclr.utils.datautils import Clip, GaussianBlur

from sklearn.cluster import KMeans


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
            batch_size= self.batch_size//2,#self.batch_size-2
            shuffle=self.shuffle,
            num_workers=8,
            pin_memory=True,
            collate_fn=naive_collate,
            drop_last=True,
        )
        self.dataset2_loader = torch.utils.data.DataLoader(
            self.dataset2,
            batch_size=self.batch_size//2,#self.batch_size-2
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
        
class ContrastiveBufferDataset(Dataset):

    def __init__(self, dataset_n, buffer_size=300, device='cuda:1', model_path = '/home/pasti/PycharmProjects/Robot_CLOD/simclr/pth/resnet50_imagenet_bs2k_epochs600.pth.tar'):
        self.buffer_size = buffer_size
        self.device = device
        # Load encoder model on gpu if available
        self.preprocess= transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])
        device = torch.device(device if torch.cuda.is_available() else 'cpu')
        namespace = Namespace(**{"data" : "imagenet", "arch" : "ResNet50"})
        self.encoder = EncodeProject(namespace)
        model_dict = torch.load(model_path)
        self.encoder.load_state_dict(model_dict['state_dict'])
        self.encoder.eval()
        self.encoder.to(device)
        
        #dataset_n = [item for item in train_dataset if item['gt_labels'].size != 0]
        indices = self.get_contr_buffer(dataset_n, self.buffer_size)
        self.buffer_dataset = torch.utils.data.Subset(dataset_n, indices)

    def __getitem__(self, index):
        return self.buffer_dataset[index]

    def __len__(self):
        return self.buffer_size
    
    def get_contr_buffer_v0(self, dataset_n, update_size):
        # 1. Get outputs from encord for all samples in dataset_n
        #    output size -> (samples, torch.Size([1, 128]))
        print('Executing step 1')
        samples = len(dataset_n)
        encoder_outs = []
        for item in dataset_n:
            path = cfg.data.train.img_path + '/' + item['img_info']['file_name']
            img = PIL.Image.open(path)
            # Compute outs
            inp = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():  # Disable gradient calculation
                encoder_out = self.encoder(inp)[1].detach().cpu()  # Detach and move to CPU
            encoder_outs.append(encoder_out)
        print('Executing step 2')
        # 2. Compute cosine similarity between all pairs of samples
        #    output size -> (torch.size([samples, samples])
        cosine_sim_matrix = torch.zeros((samples, samples))
        for i in range(samples):
            for j in range(samples):
                cosine_sim_matrix[i, j] = F.cosine_similarity(encoder_outs[i], encoder_outs[j])

        for i in range(samples):
            cosine_sim_matrix[i, i] = -1

        selected_indices = []
        for i in range(update_size):
            # 3. Sum matrix rows to get items that are most dissimilar to all the others samples
            #    output size -> (samples)
            if i==0:
                row_max = torch.max(cosine_sim_matrix, dim=1)[0]
                smallest_value, smallest_index = torch.topk(row_max, 1, largest=False)
                selected_indices.append(smallest_index.item())
                cosine_sim_matrix[smallest_index, :] = 1
            else:
                row_max = torch.max(cosine_sim_matrix[:,selected_indices], dim=1)[0]
                smallest_value, smallest_index = torch.topk(row_max, 1, largest=False)
                selected_indices.append(smallest_index.item())
                cosine_sim_matrix[smallest_index, :] = 1                
        return selected_indices
    
    def get_contr_buffer(self, dataset_n, update_size):
        # 1. Get outputs from encord for all samples in dataset_n
        #    output size -> (samples, torch.Size([1, 128]))
        print('Executing step 1')
        samples = len(dataset_n)
        encoder_outs = []
        for item in dataset_n:
            path = cfg.data.train.img_path + '/' + item['img_info']['file_name']
            img = PIL.Image.open(path)
            # Compute outs
            inp = self.preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():  # Disable gradient calculation
                encoder_out = self.encoder(inp)[0].detach().cpu()  # Detach and move to CPU
            encoder_outs.append(encoder_out)
        print('encoder outs shape' + str(encoder_outs[0].shape))
        print('Executing step 2')
        print('Clustering k= ' + str(update_size))
        # Step 1: Convert list of tensors to a single tensor
        encoder_outs_tensor = torch.cat(encoder_outs, dim=0)  # Shape: (samples, 128)

        # Step 2: Apply k-means clustering
        n_clusters = update_size # Set the number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(encoder_outs_tensor.numpy())

        # Step 3: Find the tensor closest to each cluster centroid
        centroids = kmeans.cluster_centers_
        selected_indices = []

        for centroid in centroids:
            # Compute the distance between the centroid and all points
            distances = torch.norm(encoder_outs_tensor - torch.tensor(centroid), dim=1)
            # Find the index of the closest point
            closest_index = torch.argmin(distances).item()
            selected_indices.append(closest_index)
        return selected_indices

    def update_buffer_v0(self, dataset_np1, task):
        ### KEEP FIRST IMAGES PER SUBSET
        previous_pertask = int(self.buffer_size / (task - 1))
        keep_samples_per_task = int(self.buffer_size / task)
        update_buffer_indices = []

        for t in range(0, task - 1):
            task_indices = range(t * previous_pertask, (t + 1) * previous_pertask)
            update_buffer_indices.extend(task_indices[:keep_samples_per_task])
            
        subset_n = Subset(self.buffer_dataset, update_buffer_indices)
        
        #Take a subset of the new task dataset
        #Select subset of the new task dataset
        new_indices = self.get_contr_buffer(dataset_np1 , self.buffer_size - len(subset_n))
        subset_np1 = Subset(dataset_np1, new_indices)
        # Concatenate the two subsets to form the new buffer
        self.buffer_dataset = torch.utils.data.ConcatDataset([subset_n, subset_np1])
    
    def update_buffer(self, dataset_np1, task):
        
        #dataset_np1 = [item for item in dataset_np1 if item['gt_labels'].size != 0]
        previous_pertask = int(self.buffer_size / (task - 1))
        keep_samples_per_task = int(self.buffer_size / task)
        update_buffer_indices = []

        for t in range(0, task - 1):
            task_indices = range(t * previous_pertask, (t + 1) * previous_pertask)
            update_buffer_indices.extend(random.sample(task_indices, keep_samples_per_task))
        subset_n = Subset(self.buffer_dataset, update_buffer_indices)
        
        new_indices = self.get_contr_buffer(dataset_np1 , self.buffer_size - len(subset_n))
        subset_np1 = Subset(dataset_np1, new_indices)
        # Concatenate the two subsets to form the new buffer
        self.buffer_dataset = torch.utils.data.ConcatDataset([subset_n, subset_np1])


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
    if task == 1:
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=cfg.device.batchsize_per_gpu,
            shuffle=True,
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

    #Create the model based on the task configuration file
    logger.info("Creating model...")
    Traintask = TrainingTask(cfg, evaluator)
    #Load the model weights if task is not 0
    if "load_model" in cfg.schedule:
        ckpt = torch.load(cfg.schedule.load_model)
        load_model_weight(Traintask.model, ckpt, logger)
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
        trainer.fit(Traintask, train_dataloader, val_dataloader, ckpt_path=model_resume_path)
    if task == 1:
        src = '/home/pasti/PycharmProjects/Robot_CLOD/eclod/baseModels/CUMULtask1'
        dst = '/home/pasti/PycharmProjects/Robot_CLOD/eclod/models/' + args.TiROD_task + 'task1'
        shutil.copytree(src, dst)
    ####END OF TRAINING

    ####CREATING REPLAY BUFFER
    if task == 1:
        print("Creating buffer dataset")
        buffer_dataset = ContrastiveBufferDataset(train_dataset, buffer_size=150)
    else:
        print("Updating buffer dataset")
        buffer_dataset.update_buffer(train_dataset, task)
    
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
