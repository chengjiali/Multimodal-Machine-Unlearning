import logging
import os

import torch
import torch.distributed as dist
from lavis.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.common.registry import registry
from lavis.datasets.data_utils import prepare_sample
from lavis.tasks import BaseTask, RetrievalTask, VQATask


class NegativeGradientTask(BaseTask):
    def train_step(self, model, samples):
        loss = model(samples)["loss"]
        return -loss

class NegativeGradientVQATask(VQATask, NegativeGradientTask):
    def train_step(self, model, samples):
        return super().train_step(model, samples)

class NegativeGradientRetrievalTask(RetrievalTask, NegativeGradientTask):
    def train_step(self, model, samples):
        return super().train_step(model, samples)


class VLUnlearnRetrievalTask(BaseTask):
    def __init__(self, **kwargs):
        super().__init__()
        self.model_ori = kwargs['model_ori']
        self.model_unlearn = kwargs['model_unlearn']


    def build_model(self, cfg):
        model_config = cfg.model_cfg

        model_cls = registry.get_model_class(model_config.arch)
        return model_cls.from_config(model_config)

    def train_step(self, model_ori, model_unlearn, samples, ori_embed=None):
        with torch.no_grad():
            output_ori = model_ori(samples)

        output = model_unlearn(samples)

        dis_func = nn.MSELoss()

        # Modality decoupling
        sim_i2t = output_ori['sims']['sim_i2t']
        sim_t2i = output_ori['sims']['sim_t2i']

        loss_md = dis_func(inter_modal_ori, inter_modal_unlearn)


        # Unimodal representation
        image_embeds_ori = output_ori['intermediate_output']['image_embeds']
        text_embeds_ori = output_ori['intermediate_output']['text_embeds']

        image_embeds_unlearn = output_unlearn['intermediate_output']['image_embeds']
        text_embeds_unlearn = output_unlearn['intermediate_output']['text_embeds']

        loss_unimodal = dis_func(image_embeds_ori, image_embeds_unlearn) + dis_func(text_embeds_ori, text_embeds_unlearn)


        loss = self.alpha * loss_md + (1 - self.alpha) * loss_unimodal
        wandb.log({'train_loss': loss, 'train_loss_md': loss_md, 'train_loss_unimodal': loss_unimodal})

        return loss
