import argparse
import os
import random
import copy
import wandb

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.optims import (
    LinearWarmupCosineLRScheduler,
    LinearWarmupStepLRScheduler,
)
from lavis.common.registry import registry
from lavis.common.utils import now

# imports modules for registration
from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners import *
from lavis.tasks import *

from vl_unlearn.runner_unlearn import *
from vl_unlearn.task_unlearn import *


def parse_args():
    parser = argparse.ArgumentParser(description="Training")

    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--unlearn_method", required=True, type=str, default='ft', help="unlearning method")
    parser.add_argument("--backbone", required=True, type=str, default='albef', help="vl model")
    parser.add_argument("--task", required=True, type=str, default='retrieval_flickr30k', help="vl model")
    parser.add_argument("--df_size", required=True, type=int, default=100, help="number of images to delete")
    parser.add_argument("--alpha", type=float, default=0.5, help="number of images to delete")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )

    args = parser.parse_args()
    # if 'LOCAL_RANK' not in os.environ:
    #     os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def get_runner_class(cfg):
    """
    Get runner class from config. Default to epoch-based runner.
    """
    runner_cls = registry.get_runner_class(cfg.run_cfg.get("runner", "runner_base"))

    return runner_cls


def prepare_dr_data(dataset_train_ori, cfg, data_type):
    with open(f'Df/{data_type}/image-{cfg.run_cfg.seed}.txt', 'r') as f:
        df_ids = f.readlines()
    df_ids = [i.strip() for i in df_ids]
    df_ids = df_ids[:cfg.run_cfg.df_size]
    df_ids_set = set(df_ids)

    dataset = copy.deepcopy(dataset_train_ori)

    if cfg.run_cfg.task == 'retrieval':
        num_image_before_removal = len(set([i['image'] for i in dataset.annotation]))
        dataset.annotation = [i for i in dataset.annotation if i['image'] not in df_ids_set]
        num_image_after_removal = len(set([i['image'] for i in dataset.annotation]))

    elif cfg.run_cfg.task == 'vqa':
        num_image_before_removal = len(set([i['image'] for i in dataset.annotation]))
        dataset.annotation = [i for i in dataset.annotation if i['image'] not in df_ids_set]
        dataset._add_instance_ids()
        num_image_after_removal = len(set([i['image'] for i in dataset.annotation]))

    elif cfg.model_cfg.model_type == 'nlvr':
        num_image_before_removal = len(set([str(tuple(i['images'])) for i in dataset.annotation]))
        dataset.annotation = [i for i in dataset.annotation if str(tuple(i['images'])) not in df_ids_set]
        dataset._add_instance_ids()
        num_image_after_removal = len(set([str(tuple(i['images'])) for i in dataset.annotation]))

    elif cfg.model_cfg.model_type == 've':
        num_image_before_removal = len(set([i['image'] for i in dataset.annotation]))
        dataset.annotation = [i for i in dataset.annotation if i['image'] not in df_ids_set]
        dataset._add_instance_ids()
        num_image_after_removal = len(set([i['image'] for i in dataset.annotation]))

    # assert num_image_before_removal == num_image_after_removal + cfg.run_cfg.df_size

    return dataset

def prepare_df_data(dataset_train_ori, cfg, data_type):
    with open(f'Df/{data_type}/image-{cfg.run_cfg.seed}.txt', 'r') as f:
        df_ids = f.readlines()
    df_ids = [i.strip() for i in df_ids]
    df_ids = df_ids[:cfg.run_cfg.df_size]
    df_ids_set = set(df_ids)

    dataset = copy.deepcopy(dataset_train_ori)
    
    if cfg.run_cfg.task == 'retrieval':
        dataset.annotation = [i for i in dataset.annotation if i['image'] in df_ids_set]
        num_image_after_removal = len(set([i['image'] for i in dataset.annotation]))

    elif cfg.run_cfg.task == 'vqa':
        dataset.annotation = [i for i in dataset.annotation if i['image'] in df_ids_set]
        dataset._add_instance_ids()
        num_image_after_removal = len(set([i['image'] for i in dataset.annotation]))

    elif cfg.model_cfg.model_type == 'nlvr':
        num_image_before_removal = len(set([str(tuple(i['images'])) for i in dataset.annotation]))
        dataset.annotation = [i for i in dataset.annotation if str(tuple(i['images'])) in df_ids_set]
        dataset._add_instance_ids()
        num_image_after_removal = len(set([str(tuple(i['images'])) for i in dataset.annotation]))

    elif cfg.model_cfg.model_type == 've':
        num_image_before_removal = len(set([i['image'] for i in dataset.annotation]))
        dataset.annotation = [i for i in dataset.annotation if i['image'] in df_ids_set]
        dataset._add_instance_ids()
        num_image_after_removal = len(set([i['image'] for i in dataset.annotation]))

    # assert num_image_after_removal == cfg.run_cfg.df_size, f"{num_image_after_removal}, {cfg.run_cfg.df_size}"

    return dataset


def prepare_df_data_for_test(dataset_train_ori, dataset_test_ori, cfg, data_type):
    with open(f'Df/{data_type}/image-{cfg.run_cfg.seed}.txt', 'r') as f:
        df_ids = f.readlines()
    df_ids = [i.strip() for i in df_ids]
    df_ids = df_ids[:cfg.run_cfg.df_size]
    df_ids_set = set(df_ids)


    if cfg.run_cfg.task == 'retrieval':
        # Retrieval train and test data are different. We want to use retrieval test data for Df. So copy the ori test data
        df_for_test = copy.deepcopy(dataset_test_ori)

        annotation = [i for i in dataset_train_ori.annotation if i['image'] in df_ids_set]
        num_image_after_removal = len(set([i['image'] for i in annotation]))

        # Convert to grouped format for init of RetrievalEvalDataset
        test_anno = pd.DataFrame(annotation).sort_values(by='image')
        test_anno = test_anno.groupby(['image'])['caption'].apply(list).reset_index()
        test_anno = test_anno.to_dict(orient='records')
        df_for_test.annotation = test_anno      # For __len__ method

        # init of RetrievalEvalDataset
        text = []
        image = []
        txt2img = {}
        img2txt = {}
        text_processor = df_for_test.text_processor

        txt_id = 0
        for img_id, ann in enumerate(test_anno):
            image.append(ann["image"])
            img2txt[img_id] = []
            for i, caption in enumerate(ann["caption"]):
                text.append(text_processor(caption))
                img2txt[img_id].append(txt_id)
                txt2img[txt_id] = img_id
                txt_id += 1

        df_for_test.text = text
        df_for_test.image = image
        df_for_test.txt2img = txt2img
        df_for_test.img2txt = img2txt

    elif cfg.run_cfg.task == 'vqa':
        # breakpoint()
        # Retrieval train and test data are same. To use VQA test data for Df, copy the ori train data
        df_for_test = copy.deepcopy(dataset_train_ori)

        df_for_test.annotation = [i for i in df_for_test.annotation if i['image'] in df_ids_set]
        df_for_test._add_instance_ids()
        num_image_after_removal = len(set([i['image'] for i in df_for_test.annotation]))
        # breakpoint()

    # elif cfg.run_cfg.task == 'multimodal_classification':
    #     breakpoint()
    #     df_for_test = copy.deepcopy(dataset_test_ori)

    #     df_for_test.annotation = [i for i in dataset_train_ori.annotation if i['image'] in df_ids_set]
    #     df_for_test._add_instance_ids()
    #     num_image_after_removal = len(set([i['image'] for i in df_for_test.annotation]))
    #     breakpoint()

    # NLVR train and test data are different. To use NLVR test data for Df, copy the ori test data
    elif cfg.model_cfg.model_type == 'nlvr':
        df_for_test = copy.deepcopy(dataset_test_ori)
        df_for_test.annotation = copy.deepcopy(dataset_train_ori.annotation)

        df_for_test.annotation = [i for i in df_for_test.annotation if str(tuple(i['images'])) in df_ids_set]
        df_for_test._add_instance_ids()
        num_image_after_removal = len(set([str(tuple(i['images'])) for i in df_for_test.annotation]))

    elif cfg.model_cfg.model_type in ['base', 've']:
        df_for_test = copy.deepcopy(dataset_test_ori)
        df_for_test.annotation = copy.deepcopy(dataset_train_ori.annotation)

        df_for_test.annotation = [i for i in df_for_test.annotation if i['image'] in df_ids_set]
        df_for_test._add_instance_ids()
        num_image_after_removal = len(set([i['image'] for i in df_for_test.annotation]))

    # assert num_image_after_removal == cfg.run_cfg.df_size, f"{num_image_after_removal}, {cfg.run_cfg.df_size}"

    return df_for_test

def prepare_dr_data_for_test(dataset_train_ori, dataset_test_ori, cfg, data_type, sample_size=None):
    with open(f'Df/{data_type}/image-{cfg.run_cfg.seed}.txt', 'r') as f:
        df_ids = f.readlines()
    df_ids = [i.strip() for i in df_ids]
    df_ids = df_ids[:cfg.run_cfg.df_size]
    df_ids_set = set(df_ids)


    if cfg.run_cfg.task == 'retrieval':
        num_image_before_removal = len(set([i['image'] for i in dataset_train_ori.annotation]))

        # Retrieval train and test data are different. We want to use retrieval test data for Df. So copy the ori test data
        dr_for_test = copy.deepcopy(dataset_test_ori)

        annotation = [i for i in dataset_train_ori.annotation if i['image'] not in df_ids_set]
        num_image_after_removal = len(set([i['image'] for i in annotation]))

        # Convert to grouped format for init of RetrievalEvalDataset
        test_anno = pd.DataFrame(annotation).sort_values(by='image')
        test_anno = test_anno.groupby(['image'])['caption'].apply(list).reset_index()
        test_anno = test_anno.to_dict(orient='records')
        dr_for_test.annotation = test_anno      # For __len__ method

        # init of RetrievalEvalDataset
        text = []
        image = []
        txt2img = {}
        img2txt = {}
        text_processor = dr_for_test.text_processor

        txt_id = 0
        for img_id, ann in enumerate(test_anno):
            image.append(ann["image"])
            img2txt[img_id] = []
            for i, caption in enumerate(ann["caption"]):
                text.append(text_processor(caption))
                img2txt[img_id].append(txt_id)
                txt2img[txt_id] = img_id
                txt_id += 1

        dr_for_test.text = text
        dr_for_test.image = image
        dr_for_test.txt2img = txt2img
        dr_for_test.img2txt = img2txt

    elif cfg.run_cfg.task == 'vqa':
        # breakpoint()
        # Retrieval train and test data are same. To use VQA test data for Df, copy the ori train data
        dr_for_test = copy.deepcopy(dataset_train_ori)

        dr_for_test.annotation = [i for i in dr_for_test.annotation if i['image'] not in df_ids_set]
        dr_for_test._add_instance_ids()
        num_image_after_removal = len(set([i['image'] for i in dr_for_test.annotation]))
        # breakpoint()

    # elif cfg.run_cfg.task == 'multimodal_classification':
    #     breakpoint()
    #     dr_for_test = copy.deepcopy(dataset_test_ori)

    #     dr_for_test.annotation = [i for i in dataset_train_ori.annotation if i['image'] in df_ids_set]
    #     dr_for_test._add_instance_ids()
    #     num_image_after_removal = len(set([i['image'] for i in dr_for_test.annotation]))
    #     breakpoint()

    # NLVR train and test data are different. To use NLVR test data for Df, copy the ori test data
    elif cfg.model_cfg.model_type == 'nlvr':
        dr_for_test = copy.deepcopy(dataset_test_ori)
        dr_for_test.annotation = copy.deepcopy(dataset_train_ori.annotation)

        num_image_before_removal = len(set([str(tuple(i['images'])) for i in dr_for_test.annotation]))
        dr_for_test.annotation = [i for i in dr_for_test.annotation if str(tuple(i['images'])) not in df_ids_set]

        if sample_size is not None:
            anno_id = np.arange(len(dr_for_test.annotation))
            indices = np.random.choice(anno_id, sample_size, replace=False)
            dr_for_test.annotation = [dr_for_test.annotation[i] for i in indices]

        dr_for_test._add_instance_ids()
        num_image_after_removal = len(set([str(tuple(i['images'])) for i in dr_for_test.annotation]))

    elif cfg.model_cfg.model_type in ['base', 've']:
        dr_for_test = copy.deepcopy(dataset_test_ori)
        dr_for_test.annotation = copy.deepcopy(dataset_train_ori.annotation)

        num_image_before_removal = len(set([i['image'] for i in dr_for_test.annotation]))
        dr_for_test.annotation = [i for i in dr_for_test.annotation if i['image'] not in df_ids_set]
        dr_for_test._add_instance_ids()
        num_image_after_removal = len(set([i['image'] for i in dr_for_test.annotation]))

    # assert num_image_before_removal == num_image_after_removal + cfg.run_cfg.df_size

    return dr_for_test

def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"

    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()

    args = parse_args()
    cfg = Config(args)
    cfg.run_cfg.df_size = args.df_size
    cfg.run_cfg.unlearn_method = args.unlearn_method
    cfg.run_cfg.output_dir = f'output/unlearn-{args.unlearn_method}/{args.backbone}/{args.task}/{args.df_size}/'

    # if args.unlearn_method in ['ft', 'neggrad', 'dtd']:
    #     cfg.run_cfg.max_epoch = cfg.run_cfg.max_epoch + 1
    if 'ours' in args.unlearn_method:
        cfg.run_cfg.max_epoch = 10

    if args.unlearn_method != 'retrain':
        cfg.run_cfg.resume_ckpt_path = f'output/original/{args.backbone}/{args.task}/checkpoint_best.pth'

    # Wandb
    # project = 'Unlearning - Multimodal'
    # group = args.backbone + '-' + args.task
    # name = args.unlearn_method + '-' + str(args.df_size)
    # run_id = group + '-' + name
    # wandb.init(project=project, group=group, name=name, config=args, id=run_id, resume='allow')


    if 'vlul' in args.unlearn_method:
        cfg.run_cfg.distributed = False
    else:
        init_distributed_mode(cfg.run_cfg)

    setup_seeds(cfg)

    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()

    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    runner = NewRunner(
        cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets
    )


    ## Prepare for Dr and Df
    data_name = list(cfg.datasets_cfg.keys())[0]
    if 'flickr30k' in data_name:
        data_type = 'flickr30k'
    elif 'coco' in data_name:
        data_type = 'coco'
    elif data_name == 'nlvr':
        data_type = 'nlvr'
    elif 'snli_ve' in data_name:
        data_type = 've'


    import copy
    dtrain = datasets[data_name]['train']
    dtest = datasets[data_name]['test']
    dr = prepare_dr_data(dtrain, cfg, data_type)
    df = prepare_df_data(dtrain, cfg, data_type)
    df_for_test = prepare_df_data_for_test(dtrain, dtest, cfg, data_type)
    dr_for_test = prepare_dr_data_for_test(dtrain, dtest, cfg, data_type, len(df_for_test.annotation)*5)
    datasets[data_name]['df'] = df_for_test
    datasets[data_name]['dr'] = dr_for_test


    if args.unlearn_method in ['retrain', 'ft']:
        datasets[data_name]['train'] = dr
        runner.train()

    elif args.unlearn_method in ['neggrad']:
        if args.task == 'retrieval':
            task = NegativeGradientRetrievalTask.setup_task(cfg=cfg)
        elif args.task == 'vqa':
            task = NegativeGradientVQATask.setup_task(cfg=cfg)

        datasets[data_name]['train'] = df
        runner.train()

    elif args.unlearn_method == 'dtd':
        # if args.unlearn_method == 'dtd':
        runner_class = DescentToDelete
        # elif args.unlearn_method == 'fisher':
        #     runner_class = Fisher
        # elif args.unlearn_method == 'ours':
        #     runner_class = MultimodalUnlearn

        runner = runner_class(
            cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets,
        )
        runner.unlearn()

    elif 'vlul' in args.unlearn_method:
        cfg.run_cfg.batch_size_train = cfg.run_cfg.batch_size_train // 2
        datasets[data_name]['train'] = df
        datasets[data_name]['dr_train'] = dr

        if args.task == 'retrieval':
            task = VLUnlearnClassificationTask.setup_task(cfg=cfg)
        elif args.task == 'vqa':
            task = VLUnlearnVQATask.setup_task(cfg=cfg)
        elif args.task in ['nlvr', 've']:
            task = VLUnlearnClassificationTask.setup_task(cfg=cfg)

        runner_class = MultimodalUnlearn
        model_ori = task.build_model(cfg)
        model_ori.eval()
        runner = runner_class(
            cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets, 
        )
        runner.unlearn(args, cfg, model_ori)

    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()
