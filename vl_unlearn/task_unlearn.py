import logging
import os
import wandb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from lavis.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from lavis.common.logger import MetricLogger, SmoothedValue
from lavis.common.registry import registry
from lavis.datasets.data_utils import prepare_sample
from lavis.tasks import BaseTask, RetrievalTask, VQATask, MultimodalClassificationTask
torch.autograd.set_detect_anomaly(True)


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


class VLUnlearnRetrievalTask(RetrievalTask):
    def get_embed(self, model, samples):
        image = samples["image"]
        caption = samples["text_input"]
        idx = samples["image_id"]
        
        image_embeds = model.visual_encoder.forward_features(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
        image_feat = F.normalize(model.vision_proj(image_embeds[:, 0, :]), dim=-1)

        text = model.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=model.max_txt_len,
            return_tensors="pt",
        ).to(image_embeds.device)

        text_output = model.text_encoder.forward_text(text)

        text_embeds = text_output.last_hidden_state
        text_feat = F.normalize(model.text_proj(text_embeds[:, 0, :]), dim=-1)

        sim_i2t = image_feat @ text_feat.t()
        sim_t2i = text_feat @ image_feat.t()

        encoder_output = model.text_encoder(
            encoder_embeds=text_embeds,
            attention_mask=text.attention_mask,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            mode="fusion",
        )

        return {'encoder_output': encoder_output.last_hidden_state[:, 0, :], 'image_embeds': image_feat, 'text_embeds': text_feat, 'sim_i2t': sim_i2t, 'sim_t2i': sim_t2i}

    def train_step(self, args, cfg, model_ori, model_unlearn, samples_df, samples_dr, ori_embed=None):
        
        # print('batch size', len(samples_df['text_input']), len(samples_dr['text_input']))

        output_unlearn_df = self.get_embed(model_unlearn, samples_df)
        output_unlearn_dr = self.get_embed(model_unlearn, samples_dr)
        
        # Random sample non-connected image-text pair
        ori_text_input = list(samples_df['text_input'])
        indices = torch.randperm(len(samples_df['text_input']))
        random_text_input = [samples_df['text_input'][i] for i in indices]
        samples_df['text_input'] = random_text_input

        with torch.no_grad():
            output_ori_random = self.get_embed(model_ori, samples_df)
            output_ori_dr = self.get_embed(model_ori, samples_dr)

        dis_func = nn.MSELoss()

        # Modality decoupling
        if 'md' in args.unlearn_method:
            sim_i2t_ran = output_ori_random['sim_i2t']
            sim_t2i_ran = output_ori_random['sim_t2i']

            sim_i2t_df = output_unlearn_df['sim_i2t']
            sim_t2i_df = output_unlearn_df['sim_t2i']

            loss_md = dis_func(sim_i2t_ran, sim_i2t_df) + dis_func(sim_t2i_ran, sim_t2i_df)

        else:
            loss_md = torch.tensor(0)

        # Multimodal representation
        if 'multi' in args.unlearn_method:
            multi_embeds_ori = output_ori_dr['encoder_output']
            multi_embeds = output_unlearn_dr['encoder_output']

            loss_multi = dis_func(multi_embeds_ori, multi_embeds)

        else:
            loss_multi = torch.tensor(0)

        # Unimodal representation
        if 'uni' in args.unlearn_method:
            hidden_size = output_ori_dr['image_embeds'].shape[-1]

            image_embeds_ori = output_ori['image_embeds']
            text_embeds_ori = output_ori['text_embeds']

            image_embeds_unlearn = output_unlearn['image_embeds']
            text_embeds_unlearn = output_unlearn['text_embeds']

            loss_uni = dis_func(image_embeds_ori, image_embeds_unlearn) + dis_func(text_embeds_ori, text_embeds_unlearn)

        else:
            loss_uni = torch.tensor(0)


        loss = loss_md + loss_multi + loss_uni
        out = {
            'train_loss': loss, 
            'train_loss_md': loss_md, 
            'train_loss_multi': loss_multi, 
            'train_loss_uni': loss_uni
        }
        wandb.log({'train_loss': loss.item(), 'train_loss_md': loss_md.item(), 'train_loss_multi': loss_multi.item(), 'train_loss_uni': loss_uni.item()})
        # logging.info({'train_loss': loss.item(), 'train_loss_md': loss_md.item(), 'train_loss_multi': loss_multi.item(), 'train_loss_uni': loss_uni.item()})

        return out

    def _train_step(self, args, cfg, model_ori, model_unlearn, samples_df, samples_dr, ori_embed=None):
        
        # print('batch size', len(samples_df['text_input']), len(samples_dr['text_input']))

        output_unlearn_df = model_unlearn(samples_df)
        output_unlearn_dr = model_unlearn(samples_dr)
        
        # Random sample non-connected image-text pair
        ori_text_input = list(samples_df['text_input'])
        indices = torch.randperm(len(samples_df['text_input']))
        random_text_input = [samples_df['text_input'][i] for i in indices]
        samples_df['text_input'] = random_text_input

        # with torch.no_grad():
        #     output_ori_random = model_ori(samples_df)
        #     output_ori_dr = model_ori(samples_dr)

        dis_func = nn.MSELoss()

        # Modality decoupling
        # if 'md' in args.unlearn_method:
        #     sim_i2t_ran = output_ori_random['sims']['sim_i2t']
        #     sim_t2i_ran = output_ori_random['sims']['sim_t2i']

        #     sim_i2t_df = output_unlearn_df['sims']['sim_i2t']
        #     sim_t2i_df = output_unlearn_df['sims']['sim_t2i']

        #     loss_md = dis_func(sim_i2t_ran, sim_i2t_df) + dis_func(sim_t2i_ran, sim_t2i_df)

        # else:
        #     loss_md = torch.tensor(0)

        # Multimodal representation
        # if 'multi' in args.unlearn_method:
        #     multi_embeds_ori = output_ori_dr['intermediate_output']['encoder_output']['last_hidden_state'][:, 0, :]
        #     multi_embeds = output_unlearn_dr['intermediate_output']['encoder_output']['last_hidden_state'][:, 0, :]

        #     loss_multi = dis_func(multi_embeds_ori, multi_embeds)

        # else:
        #     loss_multi = torch.tensor(0)

        # Unimodal representation
        # if 'uni' in args.unlearn_method:
        #     hidden_size = output_ori_dr['image_embeds'].shape[-1]

        #     image_embeds_ori = output_ori['intermediate_output']['image_embeds']
        #     text_embeds_ori = output_ori['intermediate_output']['text_embeds']

        #     image_embeds_unlearn = output_unlearn['intermediate_output']['image_embeds']
        #     text_embeds_unlearn = output_unlearn['intermediate_output']['text_embeds']

        #     loss_uni = dis_func(image_embeds_ori, image_embeds_unlearn) + dis_func(text_embeds_ori, text_embeds_unlearn)

        # else:
        #     loss_uni = torch.tensor(0)


        # loss = loss_md + loss_multi + loss_uni
        lb = torch.ones_like(output_unlearn_df['sims']['sim_i2t'], device=output_unlearn_df['sims']['sim_i2t'].device)
        loss = dis_func(output_unlearn_df['sims']['sim_i2t'], lb)
        # breakpoint()
        out = {
            'train_loss': loss, 
            'train_loss_md': loss, 
            'train_loss_multi': loss, 
            'train_loss_uni': loss
        }
        # wandb.log({'train_loss': loss.item(), 'train_loss_md': loss_md.item(), 'train_loss_multi': loss_multi.item(), 'train_loss_uni': loss_uni.item()})
        # logging.info({'train_loss': loss.item(), 'train_loss_md': loss_md.item(), 'train_loss_multi': loss_multi.item(), 'train_loss_uni': loss_uni.item()})

        return out


class VLUnlearnClassificationTask(MultimodalClassificationTask,BaseTask):
    def train_step(self, args, cfg, model_ori, model_unlearn, samples_df, samples_dr, ori_embed=None):
        
        # print('batch size', len(samples_df['text_input']), len(samples_dr['text_input']))

        output_unlearn_df = model_unlearn(samples_df, is_train=True)
        output_unlearn_dr = model_unlearn(samples_dr, is_train=True)

        with torch.no_grad():
            output_ori_df = model_ori(samples_df, is_train=False)
            output_ori_dr = model_ori(samples_dr, is_train=False)
        
        # Random sample non-connected image-text pair
        ori_text_input = list(samples_df['text_input'])
        indices = torch.randperm(len(samples_df['text_input']))
        random_text_input = [samples_df['text_input'][i] for i in indices]
        samples_df['text_input'] = random_text_input

        with torch.no_grad():
            output_ori_random = model_ori(samples_df, is_train=False)

        dis_func = nn.MSELoss()

        # Modality decoupling
        # On Df and random, f'(I, T) ~ f(I, T)
        if 'md' in args.unlearn_method:
            encoder_out_ran = output_ori_random['encoder_output']
            encoder_out_df = output_unlearn_df['intermediate_output']['encoder_output']['last_hidden_state'][:, 0, :]

            loss_md = dis_func(encoder_out_ran, encoder_out_df)
        else:
            loss_md = torch.tensor(0)

        # Multimodal representation
        # On Dr, f'(I, T) ~ f(I, T)
        if 'multi' in args.unlearn_method:
            multi_embeds_ori = output_ori_dr['encoder_output']
            multi_embeds = output_unlearn_dr['intermediate_output']['encoder_output']['last_hidden_state'][:, 0, :]

            loss_multi = dis_func(multi_embeds_ori, multi_embeds)

        else:
            loss_multi = torch.tensor(0)

        # Unimodal representation
        # On Df, f'(I), f'(T) ~ f(I), f(T)
        if 'image' in args.unlearn_method:
            hidden_size = output_ori_df['image_embeds'].shape[-1]
            if args.task == 'nlvr':
                image_embeds_ori = torch.vstack([
                    output_ori_df['image_embeds'].view(-1, hidden_size),
                    output_ori_df['image1_embeds'].view(-1, hidden_size)
                ])
                image_embeds = output_unlearn_df['intermediate_output']['image_embeds'].view(-1, hidden_size)
            else:
                image_embeds_ori = output_ori_df['image_embeds'].view(-1, hidden_size)
                image_embeds = output_unlearn_df['intermediate_output']['image_embeds'].view(-1, hidden_size)

            loss_uni = dis_func(image_embeds_ori, image_embeds)

        else:
            loss_uni = torch.tensor(0)


        loss = loss_md + loss_multi + loss_uni
        
        out = {
            'train_loss': loss, 
            'train_loss_md': loss_md, 
            'train_loss_multi': loss_multi, 
            'train_loss_uni': loss_uni
        }
        wandb.log({'train_loss': loss.item(), 'train_loss_md': loss_md.item(), 'train_loss_multi': loss_multi.item(), 'train_loss_uni': loss_uni.item()})
        # logging.info({'train_loss': loss.item(), 'train_loss_md': loss_md.item(), 'train_loss_multi': loss_multi.item(), 'train_loss_uni': loss_uni.item()})

        return out