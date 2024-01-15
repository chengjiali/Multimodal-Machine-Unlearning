import os
import time
import json
import wandb
import logging
import datetime
from pathlib import Path
import numpy as np
import torch
from lavis.runners.runner_base import RunnerBase

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


class NewRunner(RunnerBase):
    @property
    def test_splits(self):
        test_splits = self.config.run_cfg.get("test_splits", [])
        if hasattr(self.config.run_cfg, "unlearn_method") and 'df' not in test_splits:
            test_splits.append('df')
        if hasattr(self.config.run_cfg, "unlearn_method") and 'dr' not in test_splits:
            test_splits.append('dr')

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

    def _load_checkpoint(self, url_or_filename):
        if is_url(url_or_filename):
            cached_file = download_cached_file(
                url_or_filename, check_hash=False, progress=True
            )
            checkpoint = torch.load(cached_file, map_location=self.device)
        elif os.path.isfile(url_or_filename):
            checkpoint = torch.load(url_or_filename, map_location=self.device)
        else:
            raise RuntimeError("checkpoint url or path is invalid")

        state_dict = checkpoint["model"]
        self.unwrap_dist_model(self.model).load_state_dict(state_dict, strict=False)

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        
        # For NegGrad, use original LR
        if self.config.run_cfg.unlearn_method == 'neggrad':
            for g in self.optimizer.param_groups:
                g['lr'] = self.config.run_cfg.init_lr

        if self.scaler and "scaler" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler"])

        self.start_epoch = checkpoint["epoch"] + 1
        logging.info("Resume checkpoint from {}".format(url_or_filename))

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
        if not os.path.exists(os.path.join(self.output_dir, 'checkpoint_best.pth')):
            if self.config.run_cfg.unlearn_method in ['ft', 'neggrad', 'dtd']:
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

                # evaluation phase
                # FT and NegGrad only trains for 1 epoch. Skip valid.
                if len(self.valid_splits) > 0 and self.config.run_cfg.unlearn_method not in ['ft', 'neggrad']: 
                    for split_name in self.valid_splits:
                        logging.info("Evaluating on {}.".format(split_name))

                        val_log = self.eval_epoch(
                            split_name=split_name, cur_epoch=cur_epoch
                        )
                        if val_log is not None:
                            if is_main_process():
                                assert (
                                    "agg_metrics" in val_log
                                ), "No agg_metrics found in validation log."

                                agg_metrics = val_log["agg_metrics"]
                                if agg_metrics > best_agg_metric and split_name == "val":
                                    best_epoch, best_agg_metric = cur_epoch, agg_metrics

                                    self._save_checkpoint(cur_epoch, is_best=True)

                                val_log.update({"best_epoch": best_epoch})
                                self.log_stats(val_log, split_name)

                else:
                    # if no validation split is provided, we just save the checkpoint at the end of each epoch.
                    if not self.evaluate_only:
                        self._save_checkpoint(cur_epoch, is_best=False)
                        if is_main_process():
                            file1 = os.path.join(self.output_dir, f'checkpoint_{cur_epoch}.pth')
                            file2 = os.path.join(self.output_dir, 'checkpoint_best.pth')
                            os.system(f'./ln -sT {file1} {file2}')

                if self.evaluate_only:
                    break

                dist.barrier()

        # testing phase
        # test_epoch = "best" if len(self.valid_splits) > 0 else cur_epoch

        # # # FT and NegGrad only trains for 1 epoch. Skip reloading best ckpt.
        # # if self.config.run_cfg.unlearn_method in ['ft', 'neggrad']:
        # #     skip_reload = True
        # self.evaluate(cur_epoch=test_epoch, skip_reload=False)

        # total_time = time.time() - start_time
        # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # logging.info("Training time {}".format(total_time_str))

    def get_rep(self):
        for split_name in self.test_splits:
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

                    with open(self.output_dir / split_name, 'w') as f:
                        json.dump(logs, f)
                        
                test_logs[split_name] = logs
                logging.info(f'{split_name}: {str(logs)}')
                wandb.log({split_name: logs})

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
        self.evaluate(cur_epoch=cur_epoch, skip_reload=True)
