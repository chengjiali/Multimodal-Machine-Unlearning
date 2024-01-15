import os
import copy
import time
import json
import wandb
import logging
import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from lavis.runners.runner_base import RunnerBase
from tqdm import tqdm, trange

import torch
import torch.distributed as dist
import webdataset as wds
from lavis.common.dist_utils import (
    download_cached_file,
    get_rank,
    get_world_size,
    is_main_process,
    main_process,
)
from lavis.common.registry import registry
from lavis.common.utils import is_url
from lavis.datasets.data_utils import concat_datasets, reorg_datasets_by_split
from lavis.datasets.datasets.dataloader_utils import (
    IterLoader,
    MultiIterLoader,
    PrefetchLoader,
)
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.data.dataset import ChainDataset
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.datasets.data_utils import prepare_sample
from lavis.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from sklearn.metrics import roc_auc_score


class NewRunner(RunnerBase):
    @property
    def valid_splits(self):
        # valid_splits = self.config.run_cfg.get("valid_splits", [])

        # if len(valid_splits) == 0:
        #     logging.info("No validation splits found.")

        # if hasattr(self.config.run_cfg, "unlearn_method") and 'dr' not in valid_splits:
        #     valid_splits.append('dr')
        # if hasattr(self.config.run_cfg, "unlearn_method") and 'df' not in valid_splits:
        #     valid_splits.append('df')

        return ['dr', 'df', 'test']#self.test_splits
        
    @property
    def test_splits(self):
        test_splits = self.config.run_cfg.get("test_splits", [])
        if hasattr(self.config.run_cfg, "unlearn_method") and 'dr' not in test_splits:
            test_splits.append('dr')
        if hasattr(self.config.run_cfg, "unlearn_method") and 'df' not in test_splits:
            test_splits.append('df')

        if self.config.run_cfg.unlearn_method == 'original':
            test_splits = ['dtrain']

        return test_splits

    def setup_output_dir(self):
        output_dir = Path(self.config.run_cfg.output_dir)
        result_dir = output_dir / "result"

        output_dir.mkdir(parents=True, exist_ok=True)
        result_dir.mkdir(parents=True, exist_ok=True)

        if registry.get_path("result_dir") is None:
            registry.register_path("result_dir", str(result_dir))
        if registry.get_path("output_dir") is None:
            registry.register_path("output_dir", str(output_dir))

        self.result_dir = result_dir
        self.output_dir = output_dir

    def train(self):
        start_time = time.time()
        best_agg_metric = 0
        best_epoch = 0

        self.log_config()

        # resume from checkpoint if specified
        if not self.evaluate_only and self.resume_ckpt_path is not None:
            logging.info(f'Use ckpt from: {self.resume_ckpt_path}')
            self._load_checkpoint(self.resume_ckpt_path)
            # raise
            ## Change start
            if self.resume_ckpt_path is not None:
                if hasattr(self.model, "use_distill") and self.model.use_distill:
                    self.unwrap_dist_model(self.model.copy_params())

        # Check if trained model exists
        skip_reload = False
        if self.config.run_cfg.unlearn_method != 'dtd':
            skip_reload = True
        # cur_epoch = 0
        # file1 = os.path.join(self.output_dir, f'checkpoint_{cur_epoch}.pth')
        # file1 = f'checkpoint_{cur_epoch}.pth'
        # file2 = os.path.join(self.output_dir, 'checkpoint_best.pth')
        # os.system(f'ln -sT {file1} {file2}')

        if True: #not os.path.exists(os.path.join(self.output_dir, 'checkpoint_best.pth')) and self.config.run_cfg.unlearn_method != 'dtd':
            if self.config.run_cfg.unlearn_method in ['ft', 'ft-F', 'neggrad', 'neggrad-F', 'dtd', 'dtd-F']:
                epoch_iter = range(1)
            else:
                epoch_iter = range(self.start_epoch, self.max_epoch)
                ## Change end

            for cur_epoch in epoch_iter:
                # training phase
                if not self.evaluate_only:
                    logging.info("Start training")
                    train_stats = self.train_epoch(cur_epoch)
                    self.log_stats(split_name="train", stats=train_stats)

                    self._save_checkpoint(cur_epoch, is_best=False)
                    file1 = f'checkpoint_{cur_epoch}.pth'
                    file2 = os.path.join(self.output_dir, 'checkpoint_best.pth')
                    os.system(f'ln -sfT {file1} {file2}')

                # evaluation phase
                # FT and NegGrad only trains for 1 epoch. Skip valid.
                # and self.config.run_cfg.unlearn_method not in ['ft', 'neggrad']: 
                if len(self.valid_splits) > 0:
                    valid_logs = {}
                    for split_name in self.valid_splits:
                        logging.info("Evaluating on {}.".format(split_name))

                        logs = self.eval_epoch(
                            split_name=split_name, cur_epoch=cur_epoch
                        )

                        if split_name == 'test':
                            if 'acc' in logs:
                                test_logs['dt'] = logs['acc']
                            else:
                                test_logs['txt_r_mean'] = logs['txt_r_mean']
                                test_logs['img_r_mean'] = logs['img_r_mean']

                        if split_name == 'df':
                            if 'txt_r_mean' in logs:
                                image_embed_dr = torch.load(os.path.join(registry.get_path("output_dir"), str(cur_epoch), 'image_embed_dr.pt'))
                                text_embed_dr = torch.load(os.path.join(registry.get_path("output_dir"), str(cur_epoch), 'text_embed_dr.pt'))
                                image_embed_df = torch.load(os.path.join(registry.get_path("output_dir"), str(cur_epoch), 'image_embed_df.pt'))
                                text_embed_df = torch.load(os.path.join(registry.get_path("output_dir"), str(cur_epoch), 'text_embed_df.pt'))
                                dr_pred = (image_embed_dr @ text_embed_dr.t()).diagonal()
                                df_pred = (image_embed_df @ text_embed_df.t()).diagonal()

                                aucs = []
                                for i in range(5):
                                    s = i * df_pred.shape[0]
                                    e = s + df_pred.shape[0]

                                    y = [0] * df_pred.shape[0] + [1] * dr_pred[s:e].shape[0]
                                    p = torch.hstack([df_pred, dr_pred[s:e]]).flatten().sigmoid().numpy()

                                    a = roc_auc_score(y, p)
                                    aucs.append(a)

                            else:
                                dr_pred = torch.load(os.path.join(registry.get_path("output_dir"), str(cur_epoch), 'pred_dr.pt'))
                                df_pred = torch.load(os.path.join(registry.get_path("output_dir"), str(cur_epoch), 'pred_df.pt'))

                                aucs = []
                                for i in range(5):
                                    s = i * df_pred.shape[0]
                                    e = s + df_pred.shape[0]

                                    y = [0] * df_pred.shape[0] + [1] * dr_pred[s:e].shape[0]
                                    p = F.softmax(torch.vstack([df_pred, dr_pred[s:e]]), dim=1).numpy()

                                    if p.shape[1] == 2:
                                        a = roc_auc_score(y, p[:, 1])
                                        aucs.append(a)
                                    elif p.shape[1] == 3:
                                        a = roc_auc_score(y, p[:, 0])
                                        aucs.append(a)
                                        a = roc_auc_score(y, p[:, 1])
                                        aucs.append(a)
                                        a = roc_auc_score(y, p[:, 2])
                                        aucs.append(a)


                            print('aaaaaaaaaaaaaaaaa', split_name, aucs)
                            logs['auc'] = np.mean(aucs)
                            valid_logs['auc'] = np.mean(aucs)

                        # with open(self.output_dir / split_name, 'w') as f:
                        #     json.dump(logs, f)
                            
                        valid_logs[split_name] = logs
                        logging.info(f'{split_name}: {str(logs)}')
                        self.log_stats(logs, split_name)
                        wandb.log({split_name: logs})
                
                    self.log_stats(valid_logs, 'test')
                    logging.info(str(valid_logs))
                    with open(self.output_dir / f'{cur_epoch}/log', 'w') as f:
                        json.dump(valid_logs, f)


                else:
                    # if no validation split is provided, we just save the checkpoint at the end of each epoch.
                    if not self.evaluate_only:
                        self._save_checkpoint(cur_epoch, is_best=False)
                        if is_main_process():
                            file1 = f'checkpoint_{cur_epoch}.pth'
                            file2 = os.path.join(self.output_dir, 'checkpoint_best.pth')
                            os.system(f'ln -sfT {file1} {file2}')

                if self.evaluate_only:
                    break

                if is_dist_avail_and_initialized():
                    dist.barrier()
            # skip_reload = True

        # testing phase
        # test_epoch = "best" if len(self.valid_splits) > 0 else cur_epoch

        # # FT and NegGrad only trains for 1 epoch. Skip reloading best ckpt.
        # if self.config.run_cfg.unlearn_method in ['ft', 'neggrad']:
        #     skip_reload = True
        # self.evaluate(cur_epoch=test_epoch, skip_reload=skip_reload)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

    def get_rep(self):
        for split_name in ['val', 'dtrain'][1:]:
            self.eval_epoch(split_name=split_name, cur_epoch='best', skip_reload=False)

    def evaluate(self, cur_epoch="best", skip_reload=False):
        '''evaluate is the test loop. eval_epoch is the validation loop.'''
        test_logs = dict()

        if len(self.test_splits) > 0:
            for split_name in self.test_splits:

                if os.path.exists(self.output_dir / split_name):
                    logging.info(f"Found cache for {split_name}")
                    with open(self.output_dir / split_name, 'r') as f:
                        logs = json.load(f)

                else:
                    logging.info(f"Test on {split_name}")
                    logs = self.eval_epoch(
                        split_name=split_name, cur_epoch=cur_epoch, skip_reload=skip_reload
                    )

                    if split_name == 'test':
                        if 'acc' in logs:
                            test_logs['dt'] = logs['acc']
                        else:
                            test_logs['txt_r_mean'] = logs['txt_r_mean']
                            test_logs['img_r_mean'] = logs['img_r_mean']

                    if split_name == 'df':
                        if 'txt_r_mean' in logs:
                            image_embed_dr = torch.load(os.path.join(registry.get_path("output_dir"), str(cur_epoch), 'image_embed_dr.pt'))
                            text_embed_dr = torch.load(os.path.join(registry.get_path("output_dir"), str(cur_epoch), 'text_embed_dr.pt'))
                            image_embed_df = torch.load(os.path.join(registry.get_path("output_dir"), str(cur_epoch), 'image_embed_df.pt'))
                            text_embed_df = torch.load(os.path.join(registry.get_path("output_dir"), str(cur_epoch), 'text_embed_df.pt'))
                            dr_pred = (image_embed_dr @ text_embed_dr.t()).diagonal()
                            df_pred = (image_embed_df @ text_embed_df.t()).diagonal()

                            aucs = []
                            for i in range(5):
                                s = i * df_pred.shape[0]
                                e = s + df_pred.shape[0]

                                y = [0] * df_pred.shape[0] + [1] * dr_pred[s:e].shape[0]
                                p = torch.hstack([df_pred, dr_pred[s:e]]).flatten().sigmoid().numpy()

                                a = roc_auc_score(y, p)
                                aucs.append(a)

                        else:
                            dr_pred = torch.load(os.path.join(registry.get_path("output_dir"), str(cur_epoch), 'pred_dr.pt'))
                            df_pred = torch.load(os.path.join(registry.get_path("output_dir"), str(cur_epoch), 'pred_df.pt'))

                            aucs = []
                            for i in range(5):
                                s = i * df_pred.shape[0]
                                e = s + df_pred.shape[0]

                                y = [0] * df_pred.shape[0] + [1] * dr_pred[s:e].shape[0]
                                p = F.softmax(torch.vstack([df_pred, dr_pred[s:e]]), dim=1).numpy()

                                if p.shape[1] == 2:
                                    a = roc_auc_score(y, p[:, 1])
                                    aucs.append(a)
                                elif p.shape[1] == 3:
                                    a = roc_auc_score(y, p[:, 0])
                                    aucs.append(a)
                                    a = roc_auc_score(y, p[:, 1])
                                    aucs.append(a)
                                    a = roc_auc_score(y, p[:, 2])
                                    aucs.append(a)


                        print('aaaaaaaaaaaaaaaaa', split_name, aucs)
                        logs['auc'] = np.mean(aucs)
                        test_logs['auc'] = np.mean(aucs)

                    # with open(self.output_dir / split_name, 'w') as f:
                    #     json.dump(logs, f)
                        
                test_logs[split_name] = logs
                logging.info(f'{split_name}: {str(logs)}')
                wandb.log({split_name: logs})
            
            self.log_stats(test_logs, 'test')
            with open(self.output_dir / 'test', 'w') as f:
                json.dump(test_logs, f)

            return test_logs



class DescentToDelete(NewRunner):

    def compute_sigma(self, num_examples, iterations, lipshitz, smooth, strong, epsilon, delta):
        """Theorem 3.1 https://arxiv.org/pdf/2007.02923.pdf"""

        print('delta', delta)
        gamma = (smooth - strong) / (smooth + strong)
        numerator = 4 * np.sqrt(2) * lipshitz * np.power(gamma, iterations)
        denominator = (strong * num_examples * (1 - np.power(gamma, iterations))) * ((np.sqrt(np.log(1 / delta) + epsilon)) - np.sqrt(np.log(1 / delta)))
        print('sigma', numerator, denominator, numerator / denominator)
    
        return numerator / denominator

    def publish(self, model, sigma):
        """Publishing function which adds Gaussian noise with scale sigma."""

        with torch.no_grad():
            for n, p in model.named_parameters():
                if p.requires_grad:
                    p.copy_(p + torch.empty_like(p).normal_(0, sigma))
    
    def unlearn(self):

        # resume from checkpoint if specified
        logging.info(f'Use ckpt from: {self.resume_ckpt_path}')
        self._load_checkpoint(self.resume_ckpt_path)
        ## Change start
        if hasattr(self.model, "use_distill") and self.model.use_distill:
            self.unwrap_dist_model(self.model.copy_params())
        ## Change end

        # if os.path.exists()
        train_size = len(self.dataloaders['train']._dataloader.dataset)

        cur_epoch = 1
        sigma = self.compute_sigma(
            train_size, 
            cur_epoch, 
            1 + self.config.run_cfg.weight_decay, 
            4 - self.config.run_cfg.weight_decay, 
            self.config.run_cfg.weight_decay, 
            5, 
            1 / train_size / train_size)
        
        self.publish(self.model, sigma)

        # self._save_checkpoint(cur_epoch, is_best=False)
        self.evaluate(cur_epoch=0, skip_reload=True)


class MultimodalUnlearn(NewRunner):
    def unlearn(self, args, cfg):
        best_acc = 0
        # dtrain_image_embed = torch.load(f'output/original/{args.backbone}/{args.task}/image_embed_dtrain.pt')
        # dtrain_hidden_state = torch.load(f'output/original/{args.backbone}/{args.task}/hidden_state_dtrain.pt')

        # Load original model and Initialize unlearned model from original checkpoint
        resume_ckpt_path = f'output/original/{args.backbone}/{args.task}/checkpoint_best.pth'
        self._load_checkpoint(resume_ckpt_path)
        if hasattr(self.model, "use_distill") and self.model.use_distill:
            self.model.copy_params()
        model_ori = copy.deepcopy(self.model)
        model_ori.eval()
        for n, p in model_ori.named_parameters():
            p.requires_grad = False
        logging.info(f'Use ckpt from: {resume_ckpt_path}')


        un = {n: p for n, p in self.model.named_parameters()}
        ori = {n: p for n, p in model_ori.named_parameters()}
        common_keys = un.keys() & ori.keys()

        for n in common_keys:
            print(n, (un[n] == ori[n]).all())

        # distributed training wrapper
        if self.use_distributed:
            model_ori = DDP(model_ori, device_ids=[self.config.run_cfg.gpu])


        start_time = time.time()
        best_agg_metric = 0
        best_epoch = 0

        self.log_config()

        # resume from checkpoint if specified
        # if not self.evaluate_only and self.resume_ckpt_path is not None:
        #     logging.info(f'Use ckpt from: {self.resume_ckpt_path}')
        #     self._load_checkpoint(self.resume_ckpt_path)
        #     # raise
        #     ## Change start
        #     if self.resume_ckpt_path is not None:
        #         if hasattr(self.model, "use_distill") and self.model.use_distill:
        #             self.unwrap_dist_model(self.model).copy_params()

        # Check if trained model exists
        skip_reload = False
        if self.config.run_cfg.unlearn_method != 'dtd':
            skip_reload = True

        self._load_checkpoint(f'output/unlearn-vlul-md-multi-image-F/albef/retrieval_flickr30k/5000/checkpoint_2.pth')
        if True:#os.path.exists(os.path.join(self.output_dir, 'checkpoint_best.pth')):
            for cur_epoch in range(self.max_epoch):#, desc='Epoch'):
                logging.info("Start training")
                logging.info(f'eeeeeeeeeeeeeeeeeeeee: {self.optimizer.param_groups[0]["lr"]}; {cur_epoch}')
                print(f'eeeeeeeeeeeeeeeeeeeee: {self.optimizer.param_groups[0]["lr"]}; {cur_epoch}')
                self.model.train()

                # Basic training loop from base_task.py
                use_amp = False

                iters_per_epoch = len(self.train_loader)
                if not hasattr(self.train_loader, "__next__"):
                    # convert to iterator if not already
                    self.train_loader = iter(self.train_loader)
                if not hasattr(self.dataloaders['dr_train'], "__next__"):
                    # convert to iterator if not already
                    self.dataloaders['dr_train'] = iter(self.dataloaders['dr_train'])


                metric_logger = MetricLogger(delimiter="  ")
                metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.4e}"))
                metric_logger.add_meter("loss", SmoothedValue(window_size=1, fmt="{value:.4e}"))
                metric_logger.add_meter("loss_md", SmoothedValue(window_size=1, fmt="{value:.4e}"))
                metric_logger.add_meter("loss_multi", SmoothedValue(window_size=1, fmt="{value:.4e}"))
                metric_logger.add_meter("loss_uni", SmoothedValue(window_size=1, fmt="{value:.4e}"))

                # if iter-based runner, schedule lr based on inner epoch.
                logging.info(f"Start training epoch {cur_epoch}, {iters_per_epoch} iters per inner epoch.")
                header = "Train: data epoch: [{}]".format(cur_epoch)
                start_iters = None
                if start_iters is None:
                    # epoch-based runner
                    inner_epoch = cur_epoch
                else:
                    # In iter-based runner, we schedule the learning rate based on iterations.
                    inner_epoch = start_iters // iters_per_epoch
                    header = header + "; inner epoch [{}]".format(inner_epoch)

                for i in metric_logger.log_every(range(iters_per_epoch), self.log_freq, header):
                    # if using iter-based runner, we stop after iters_per_epoch iterations.
                    if i >= iters_per_epoch:
                        break

                    samples_df = next(self.train_loader)
                    samples_df = prepare_sample(samples_df, cuda_enabled=self.cuda_enabled)
                    samples_df.update(
                        {
                            "epoch": inner_epoch,
                            "num_iters_per_epoch": iters_per_epoch,
                            "iters": i,
                        }
                    )


                    samples_dr = next(self.dataloaders['dr_train'])
                    samples_dr = prepare_sample(samples_dr, cuda_enabled=self.cuda_enabled)
                    samples_dr.update(
                        {
                            "epoch": inner_epoch,
                            "num_iters_per_epoch": iters_per_epoch,
                            "iters": i,
                        }
                    )


                    self.lr_scheduler.step(cur_epoch=inner_epoch, cur_step=i)

                    with torch.cuda.amp.autocast(enabled=use_amp):
                        out = self.task.train_step(args, cfg, model_ori, self.model, samples_df, samples_dr)

                    loss = out['train_loss']
                    loss_md = out['train_loss_md']
                    loss_multi = out['train_loss_multi']
                    loss_uni = out['train_loss_uni']

                    # after_train_step()
                    loss.backward()

                    # update gradients every accum_grad_iters iterations
                    if (i + 1) % self.accum_grad_iters == 0:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                    metric_logger.update(loss=loss.item())
                    metric_logger.update(loss_md=loss_md.item())
                    metric_logger.update(loss_multi=loss_multi.item())
                    metric_logger.update(loss_uni=loss_uni.item())
                    metric_logger.update(lr=self.optimizer.param_groups[0]["lr"])

                # after train_epoch()
                # gather the stats from all processes
                metric_logger.synchronize_between_processes()
                logging.info("Averaged stats: " + str(metric_logger.global_avg()))
                train_stats =  {
                    k: "{:.6e}".format(meter.global_avg)
                    for k, meter in metric_logger.meters.items()
                }
                self.log_stats(split_name="train", stats=train_stats)

                # evaluation phase
                # FT and NegGrad only trains for 1 epoch. Skip valid.
                if len(self.valid_splits) > 0 and cur_epoch > 1: 
                    valid_log = {}
                    self._save_checkpoint(cur_epoch, is_best=False)
                    for split_name in self.valid_splits:
                        logging.info("Evaluating on {}.".format(split_name))

                        logs = self.eval_epoch(
                            split_name=split_name, cur_epoch=cur_epoch
                        )

                        if split_name == 'test':
                            valid_log['dt'] = logs['acc']

                        if split_name == 'df':
                            if 'txt_r_mean' in logs:
                                image_embed_dr = torch.load(os.path.join(registry.get_path("output_dir"), str(cur_epoch), 'image_embed_dr.pt'))
                                text_embed_dr = torch.load(os.path.join(registry.get_path("output_dir"), str(cur_epoch), 'text_embed_dr.pt'))
                                image_embed_df = torch.load(os.path.join(registry.get_path("output_dir"), str(cur_epoch), 'image_embed_df.pt'))
                                text_embed_df = torch.load(os.path.join(registry.get_path("output_dir"), str(cur_epoch), 'text_embed_df.pt'))
                                dr_pred = (image_embed_dr @ text_embed_dr.t()).diagonal()
                                df_pred = (image_embed_df @ text_embed_df.t()).diagonal()

                                aucs = []
                                for i in range(5):
                                    s = i * df_pred.shape[0]
                                    e = s + df_pred.shape[0]

                                    y = [0] * df_pred.shape[0] + [1] * dr_pred[s:e].shape[0]
                                    p = torch.hstack([df_pred, dr_pred[s:e]]).flatten().sigmoid().numpy()

                                    a = roc_auc_score(y, p)
                                    aucs.append(a)

                            else:
                                dr_pred = torch.load(os.path.join(registry.get_path("output_dir"), str(cur_epoch), 'pred_dr.pt'))
                                df_pred = torch.load(os.path.join(registry.get_path("output_dir"), str(cur_epoch), 'pred_df.pt'))

                                aucs = []
                                for i in range(5):
                                    s = i * df_pred.shape[0]
                                    e = s + df_pred.shape[0]

                                    y = [0] * df_pred.shape[0] + [1] * dr_pred[s:e].shape[0]
                                    p = F.softmax(torch.vstack([df_pred, dr_pred[s:e]]), dim=1).numpy()

                                    if p.shape[1] == 2:
                                        a = roc_auc_score(y, p[:, 1])
                                        aucs.append(a)
                                    elif p.shape[1] == 3:
                                        a = roc_auc_score(y, p[:, 0])
                                        aucs.append(a)
                                        a = roc_auc_score(y, p[:, 1])
                                        aucs.append(a)
                                        a = roc_auc_score(y, p[:, 2])
                                        aucs.append(a)

                            print('aaaaaaaaaaaaaaaaa', split_name, aucs)
                            valid_log['auc'] = np.mean(aucs)

                    if is_main_process():
                        if (valid_log['dt'] + valid_log['auc']) / 2 > best_acc:
                            best_epoch, best_acc = cur_epoch, (valid_log['dt'] + valid_log['auc']) / 2

                            self._save_checkpoint(cur_epoch, is_best=True)
                            valid_log.update({"best_epoch": best_epoch})

                            file1 = str(cur_epoch)
                            file2 = os.path.join(self.output_dir, 'best_epoch')
                            os.system(f'ln -sfT {file1} {file2}')

                        self.log_stats(valid_log, split_name)
                        logging.info(valid_log)



                else:
                    # if no validation split is provided, we just save the checkpoint at the end of each epoch.
                    if not self.evaluate_only:
                        self._save_checkpoint(cur_epoch, is_best=False)
                        if is_main_process():
                            file1 = f'checkpoint_{cur_epoch}.pth'
                            file2 = os.path.join(self.output_dir, 'checkpoint_best.pth')
                            os.system(f'ln -sfT {file1} {file2}')

                if self.evaluate_only:
                    break

                if is_dist_avail_and_initialized():
                    dist.barrier()
            # skip_reload = True

        # testing phase
        # test_epoch = "best" if len(self.valid_splits) > 0 else cur_epoch

        # # FT and NegGrad only trains for 1 epoch. Skip reloading best ckpt.
        # if self.config.run_cfg.unlearn_method in ['ft', 'neggrad']:
        #     skip_reload = True
        # self.evaluate(cur_epoch=test_epoch, skip_reload=skip_reload)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Training time {}".format(total_time_str))

        