import os
import random
from argparse import ArgumentParser

import numpy as np
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import OPTForCausalLM, GPT2Tokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader

from video_expert_extract import VideoDS

CURRENT_DIR = os.getcwd()
GLOBAL_DTYPE = torch.float16
GLOBAL_DEVICE = 'cuda'


def save_checkpoint(state, filename='checkpoint'):
    torch.save(state, filename + '.pth.tar')


class VideoAligner(nn.Module):
    def __init__(self,
                 lm,
                 h_dim,
                 tokenizer,
                 double_priest_loss_p=0.5):
        super().__init__()
        self.lm = lm
        self.tokenizer = tokenizer
        self.h_dim = h_dim
        self.prepare_models()
        self.initialize_trainable()
        self.initialize_prompt()
        self.double_priest_loss_p = double_priest_loss_p

    def get_state_dict(self, efficient=True):
        if not efficient:
            return self.state_dict()
        else:
            ret_dict = {}
            all_dict = self.state_dict()
            important_keys = [f for f in all_dict.keys() if 'trainable' in f]
            for key in important_keys:
                ret_dict.update({key: all_dict[key]})
            return ret_dict

    def initialize_prompt(self):
        self.PRE_PROMPT = 'What is happening in this video?'
        self.TOKENIZED_PRE_PROMPT = self.tokenizer(self.PRE_PROMPT, add_special_tokens=True)['input_ids']
        target_tokens = torch.LongTensor(self.TOKENIZED_PRE_PROMPT).to(GLOBAL_DEVICE)
        self.EMB_PRE_PROMPT = self.lm.model.decoder.embed_tokens(target_tokens)

        self.Simple_Solution = 'Answer:Someone is '
        self.TOKENIZED_Simple_Solution = self.tokenizer(self.Simple_Solution, add_special_tokens=False)['input_ids']
        target_tokens = torch.LongTensor(self.TOKENIZED_Simple_Solution).to(GLOBAL_DEVICE)
        self.EMB_Simple_Solution = self.lm.model.decoder.embed_tokens(target_tokens)

        self.DP_PRE_PROMPT_1 = 'This is video number 1.'
        self.TOKEMIZED_PROMPT_1 = self.tokenizer(self.DP_PRE_PROMPT_1, add_special_tokens=True)['input_ids']
        target_tokens = torch.LongTensor(self.TOKEMIZED_PROMPT_1).to(GLOBAL_DEVICE)
        self.EMB_PROMPT_1 = self.lm.model.decoder.embed_tokens(target_tokens)

        self.DP_PRE_PROMPT_2 = 'This is video number 2.'
        self.TOKEMIZED_PROMPT_2 = self.tokenizer(self.DP_PRE_PROMPT_2, add_special_tokens=True)['input_ids']
        target_tokens = torch.LongTensor(self.TOKEMIZED_PROMPT_2).to(GLOBAL_DEVICE)
        self.EMB_PROMPT_2 = self.lm.model.decoder.embed_tokens(target_tokens)

        self.DP_PRE_PROMPT_3 = 'Are the videos showing the same action? Answer:'
        self.TOKEMIZED_PROMPT_3 = self.tokenizer(self.DP_PRE_PROMPT_3, add_special_tokens=False)['input_ids']
        target_tokens = torch.LongTensor(self.TOKEMIZED_PROMPT_3).to(GLOBAL_DEVICE)
        self.EMB_PROMPT_3 = self.lm.model.decoder.embed_tokens(target_tokens)

        self.DP_PRE_PROMPT_5 = 'Are the videos showing a different action? Answer:'
        self.TOKEMIZED_PROMPT_5 = self.tokenizer(self.DP_PRE_PROMPT_5, add_special_tokens=False)['input_ids']
        target_tokens = torch.LongTensor(self.TOKEMIZED_PROMPT_5).to(GLOBAL_DEVICE)
        self.EMB_PROMPT_5 = self.lm.model.decoder.embed_tokens(target_tokens)

        self.DP_PRE_PROMPT_4 = 'Which video is showing somebody '
        self.TOKEMIZED_PROMPT_4 = self.tokenizer(self.DP_PRE_PROMPT_4, add_special_tokens=False)['input_ids']
        target_tokens = torch.LongTensor(self.TOKEMIZED_PROMPT_4).to(GLOBAL_DEVICE)
        self.EMB_PROMPT_4 = self.lm.model.decoder.embed_tokens(target_tokens)

        self.Action_1 = 'applying eye makeup.'
        self.Action_2 = 'applying lipstick.'
        self.Action_3 = 'doing archery.'
        self.Action_4 = 'crawling.'
        self.Action_5 = 'balancing a beam.'
        self.Action_6 = 'marching in a band.'
        self.Action_7 = 'pitching at baseball.'
        self.Action_8 = 'playing basketball.'
        self.Action_9 = 'dunking at basketball.'
        self.Action_10 = 'bench press.'

        self.qAction_1 = 'applying eye makeup? Answer:'
        self.qAction_2 = 'applying lipstick? Answer:'
        self.qAction_3 = 'doing archery? Answer:'
        self.qAction_4 = 'crawling? Answer:'
        self.qAction_5 = 'balancing a beam? Answer:'
        self.qAction_6 = 'marching in a band? Answer:'
        self.qAction_7 = 'pitching at baseball? Answer:'
        self.qAction_8 = 'playing basketball? Answer:'
        self.qAction_9 = 'dunking at basketball? Answer:'
        self.qAction_10 = 'bench press? Answer:'

        self.TOKENIZED_Action_1 = self.tokenizer(self.Action_1, add_special_tokens=False)['input_ids']
        self.TOKENIZED_Action_2 = self.tokenizer(self.Action_2, add_special_tokens=False)['input_ids']
        self.TOKENIZED_Action_3 = self.tokenizer(self.Action_3, add_special_tokens=False)['input_ids']
        self.TOKENIZED_Action_4 = self.tokenizer(self.Action_4, add_special_tokens=False)['input_ids']
        self.TOKENIZED_Action_5 = self.tokenizer(self.Action_5, add_special_tokens=False)['input_ids']
        self.TOKENIZED_Action_6 = self.tokenizer(self.Action_6, add_special_tokens=False)['input_ids']
        self.TOKENIZED_Action_7 = self.tokenizer(self.Action_7, add_special_tokens=False)['input_ids']
        self.TOKENIZED_Action_8 = self.tokenizer(self.Action_8, add_special_tokens=False)['input_ids']
        self.TOKENIZED_Action_9 = self.tokenizer(self.Action_9, add_special_tokens=False)['input_ids']
        self.TOKENIZED_Action_10 = self.tokenizer(self.Action_10, add_special_tokens=False)['input_ids']

        self.qTOKENIZED_Action_1 = self.tokenizer(self.qAction_1, add_special_tokens=False)['input_ids']
        self.qTOKENIZED_Action_2 = self.tokenizer(self.qAction_2, add_special_tokens=False)['input_ids']
        self.qTOKENIZED_Action_3 = self.tokenizer(self.qAction_3, add_special_tokens=False)['input_ids']
        self.qTOKENIZED_Action_4 = self.tokenizer(self.qAction_4, add_special_tokens=False)['input_ids']
        self.qTOKENIZED_Action_5 = self.tokenizer(self.qAction_5, add_special_tokens=False)['input_ids']
        self.qTOKENIZED_Action_6 = self.tokenizer(self.qAction_6, add_special_tokens=False)['input_ids']
        self.qTOKENIZED_Action_7 = self.tokenizer(self.qAction_7, add_special_tokens=False)['input_ids']
        self.qTOKENIZED_Action_8 = self.tokenizer(self.qAction_8, add_special_tokens=False)['input_ids']
        self.qTOKENIZED_Action_9 = self.tokenizer(self.qAction_9, add_special_tokens=False)['input_ids']
        self.qTOKENIZED_Action_10 = self.tokenizer(self.qAction_10, add_special_tokens=False)['input_ids']

        self.tokenized_action_registry = [self.TOKENIZED_Action_1, self.TOKENIZED_Action_2, self.TOKENIZED_Action_3,
                                          self.TOKENIZED_Action_4, self.TOKENIZED_Action_5, self.TOKENIZED_Action_6,
                                          self.TOKENIZED_Action_7, self.TOKENIZED_Action_8, self.TOKENIZED_Action_9,
                                          self.TOKENIZED_Action_10]
        self.emb_action_registry = [self.lm.model.decoder.embed_tokens(torch.LongTensor(f).to(GLOBAL_DEVICE)) for f in
                                    self.tokenized_action_registry]

        self.qtokenized_action_registry = [self.qTOKENIZED_Action_1, self.qTOKENIZED_Action_2, self.qTOKENIZED_Action_3,
                                           self.qTOKENIZED_Action_4, self.qTOKENIZED_Action_5, self.qTOKENIZED_Action_6,
                                           self.qTOKENIZED_Action_7, self.qTOKENIZED_Action_8, self.qTOKENIZED_Action_9,
                                           self.qTOKENIZED_Action_10]
        self.qemb_action_registry = [self.lm.model.decoder.embed_tokens(torch.LongTensor(f).to(GLOBAL_DEVICE)) for f in
                                     self.qtokenized_action_registry]
        self.tokenized_solution_1 = self.tokenizer(' 1', add_special_tokens=False)['input_ids']
        self.emb_solution_1 = self.lm.model.decoder.embed_tokens(
            torch.LongTensor(self.tokenized_solution_1).to(GLOBAL_DEVICE))
        self.tokenized_solution_2 = self.tokenizer(' 2', add_special_tokens=False)['input_ids']
        self.emb_solution_2 = self.lm.model.decoder.embed_tokens(
            torch.LongTensor(self.tokenized_solution_2).to(GLOBAL_DEVICE))
        self.tokenized_solution_yes = self.tokenizer(' Yes', add_special_tokens=False)['input_ids']
        self.emb_solution_yes = self.lm.model.decoder.embed_tokens(
            torch.LongTensor(self.tokenized_solution_yes).to(GLOBAL_DEVICE))
        self.tokenized_solution_no = self.tokenizer(' No', add_special_tokens=False)['input_ids']
        self.emb_solution_no = self.lm.model.decoder.embed_tokens(
            torch.LongTensor(self.tokenized_solution_no).to(GLOBAL_DEVICE))

    def prepare_models(self):
        self.lm.eval()
        self.lm = self.lm.to(GLOBAL_DEVICE)
        for param in self.lm.parameters():
            param.requires_grad = False

    def initialize_trainable(self):
        self.trainable_game_mode_token = nn.Parameter(torch.randn(4, self.h_dim), requires_grad=True)
        self.trainable_projection = nn.Linear(in_features=768, out_features=self.h_dim)
        self.trainable_answer_prompt = nn.Parameter(torch.randn(1, self.h_dim), requires_grad=True)
        print(f"Initialized a Trainable Projection Layer from {768} to {self.h_dim} dims\n")
        self.trainable_module_names = ['trainable_game_mode_token', 'trainable_projection', 'trainable_answer_prompt']
        return

    def option_a(self, feats, labels):
        """
        Mix feats per 2, and ask which one is doing label_a or label_b
        """
        batch_size = feats.size()[0]
        half_bs = batch_size // 2
        trainable_add = self.trainable_game_mode_token.repeat(half_bs, 1, 1)
        feats_1 = feats[0:half_bs, :].unsqueeze(1) + trainable_add
        feats_2 = feats[half_bs:, :].unsqueeze(1) + trainable_add
        labels_1 = labels[0:half_bs]
        labels_2 = labels[half_bs:]

        ### This is video_number_1 [VIDEO_1] this is video_number_2 [VIDEO_2] Which video is showing somebody ###
        prompt_pt_1 = [feats_1, self.EMB_PROMPT_1.repeat(half_bs, 1, 1), feats_2,
                       self.EMB_PROMPT_2.repeat(half_bs, 1, 1), self.EMB_PROMPT_4.repeat(half_bs, 1, 1),
                       ]
        prompt_pt_1_as_embeddings = torch.concat(prompt_pt_1, dim=1)

        ## Doing Label 1
        gt_label_1_prompt = [self.qtokenized_action_registry[f] for f in labels_1]
        padded_gt_label_1_prompt = self.pad_list_of_lists(gt_label_1_prompt, return_as='list', pad_item=1437, max_pad=8)
        emb_padded_gt_label_1_prompt = [self.lm.model.decoder.embed_tokens(torch.LongTensor(f).to(GLOBAL_DEVICE)) for f
                                        in
                                        padded_gt_label_1_prompt]

        tok_answer_1 = self.tokenized_solution_1
        emb_answer_1 = self.emb_solution_1
        xxx = torch.stack(emb_padded_gt_label_1_prompt, dim=0)
        complete_emb_prompt_1 = torch.concat(
            [prompt_pt_1_as_embeddings, xxx, self.trainable_answer_prompt.repeat(half_bs, 1, 1),
             emb_answer_1.repeat(half_bs, 1, 1)], dim=1)

        ## Doing label 2
        gt_label_2_prompt = [self.qtokenized_action_registry[f] for f in labels_2]
        padded_gt_label_2_prompt = self.pad_list_of_lists(gt_label_2_prompt, return_as='list', pad_item=1437, max_pad=8)
        emb_padded_gt_label_2_prompt = [self.lm.model.decoder.embed_tokens(torch.LongTensor(f).to(GLOBAL_DEVICE)) for f
                                        in
                                        padded_gt_label_2_prompt]
        tok_answer_2 = self.tokenized_solution_2
        emb_answer_2 = self.emb_solution_2
        xxx = torch.stack(emb_padded_gt_label_2_prompt, dim=0)
        complete_emb_prompt_2 = torch.concat(
            [prompt_pt_1_as_embeddings, xxx, self.trainable_answer_prompt.repeat(half_bs, 1, 1),
             emb_answer_2.repeat(half_bs, 1, 1)], dim=1)

        y = torch.LongTensor(tok_answer_1 * half_bs + tok_answer_2 * half_bs).view(batch_size, -1).to(GLOBAL_DEVICE)
        final_llm_embs = torch.concat([complete_emb_prompt_1, complete_emb_prompt_2], dim=0)
        answer_idx = complete_emb_prompt_1.size()[1] - emb_answer_1.size()[0]
        return y, final_llm_embs, answer_idx

    def option_b(self, feats, labels):
        """
        Mix feats per 2, and ask if they are doing the same thing
        """
        batch_size = feats.size()[0]
        half_bs = batch_size // 2
        trainable_add = self.trainable_game_mode_token.repeat(half_bs, 1, 1)
        feats_1 = feats[0:half_bs, :].unsqueeze(1) + trainable_add
        feats_2 = feats[half_bs:, :].unsqueeze(1) + trainable_add
        labels_1 = labels[0:half_bs]
        labels_2 = labels[half_bs:]

        ### This is video_number_1 [VIDEO_1] this is video_number_2 [VIDEO_2] Are the videos showing the same action? Answer: ###
        prompt_pt_1 = [feats_1, self.EMB_PROMPT_1.repeat(half_bs, 1, 1), feats_2,
                       self.EMB_PROMPT_2.repeat(half_bs, 1, 1), self.EMB_PROMPT_3.repeat(half_bs, 1, 1),
                       ]
        prompt_pt_1_as_embeddings = torch.concat(prompt_pt_1, dim=1)

        ### This is video_number_1 [VIDEO_1] this is video_number_2 [VIDEO_2] Are the videos showing a different action? Answer: ###
        prompt_pt_2 = [feats_1, self.EMB_PROMPT_1.repeat(half_bs, 1, 1), feats_2,
                       self.EMB_PROMPT_2.repeat(half_bs, 1, 1), self.EMB_PROMPT_5.repeat(half_bs, 1, 1),
                       ]
        prompt_pt_2_as_embeddings = torch.concat(prompt_pt_2, dim=1)

        ### Compare labels_1 with labels_2 ###
        is_same = [bool(f) for f in (labels_1 == labels_2).cpu()]

        tokenized_answers_1 = []
        tokenized_answers_2 = []
        emb_answers_1 = []
        emb_answers_2 = []
        for f in is_same:
            if f:
                tokenized_answers_1.append(self.tokenized_solution_yes)
                emb_answers_1.append(self.emb_solution_yes)
                tokenized_answers_2.append(self.tokenized_solution_no)
                emb_answers_2.append(self.emb_solution_no)
            else:
                tokenized_answers_1.append(self.tokenized_solution_no)
                emb_answers_1.append(self.emb_solution_no)
                tokenized_answers_2.append(self.tokenized_solution_yes)
                emb_answers_2.append(self.emb_solution_yes)

        complete_emb_prompt_1 = torch.concat(
            [prompt_pt_1_as_embeddings, self.trainable_answer_prompt.repeat(half_bs, 1, 1),
             torch.stack(emb_answers_1, dim=0)], dim=1)
        complete_emb_prompt_2 = torch.concat(
            [prompt_pt_2_as_embeddings, self.trainable_answer_prompt.repeat(half_bs, 1, 1),
             torch.stack(emb_answers_2, dim=0)], dim=1)

        y = torch.LongTensor(tokenized_answers_1 + tokenized_answers_2).view(batch_size, -1).to(GLOBAL_DEVICE)
        final_llm_embs = torch.concat([complete_emb_prompt_1, complete_emb_prompt_2], dim=0)
        answer_idx = complete_emb_prompt_1.size()[1] - emb_answers_1[0].size()[0]
        return y, final_llm_embs, answer_idx

    def option_c(self, feats, labels):
        """
        Ask what's happening in the video
        """
        batch_size, feature_size = feats.size()[0], feats.size()[1]
        trainable_add = self.trainable_game_mode_token.repeat(batch_size, 1, 1)
        feats_1 = feats.unsqueeze(1) + trainable_add
        labels_1 = labels

        ### What is happening in this video? Answer: Someone is###
        prompt_concated = [feats_1, self.EMB_PRE_PROMPT.repeat(batch_size, 1, 1),
                           self.EMB_Simple_Solution.repeat(batch_size, 1, 1)]
        prompt_as_embeddings = torch.concat(prompt_concated, dim=1)
        gt_labels = [self.tokenized_action_registry[f] for f in labels_1]
        padded_gt_labels = self.pad_list_of_lists(gt_labels, return_as='list', pad_item=2)
        emb_padded_gt_labels = [self.lm.model.decoder.embed_tokens(torch.LongTensor(f).to(GLOBAL_DEVICE)) for f in
                                padded_gt_labels]
        final_llm_embs = torch.concat([prompt_as_embeddings, self.trainable_answer_prompt.repeat(batch_size, 1, 1),
                                       torch.stack(emb_padded_gt_labels, dim=0)], dim=1)
        return torch.LongTensor(padded_gt_labels).to(GLOBAL_DEVICE), final_llm_embs, prompt_as_embeddings.size()[1] + 1

    def pad_list_of_lists(self, inp, pad_item=2, return_as='pt', max_pad=None):
        # Max length #
        max_length = max([len(i) for i in inp])
        if max_pad is not None:
            max_length = max_pad
        new_inp = []
        for i in range(len(inp)):
            new_inp.append(inp[i] + [pad_item] * (max_length - len(inp[i])))
        if return_as == 'list':
            return new_inp
        else:
            return torch.LongTensor(new_inp)

    def preprocess_video(self, feats, labels):
        llm_projected_features = self.trainable_projection(feats)
        seed = random.uniform(0, 1)
        if (seed < self.double_priest_loss_p):
            if seed < self.double_priest_loss_p / 2:
                ### A: Mix 2 images and ask which is doing [Label] ###
                y, x_concat_y_e, answer_idx = self.option_a(feats=llm_projected_features, labels=labels)

            else:
                ### B: Mix 2 images and ask if they are showing the same thing [Label] ###
                y, x_concat_y_e, answer_idx = self.option_b(feats=llm_projected_features, labels=labels)
        else:
            ### C: Ask what is happening in the Video ###
            y, x_concat_y_e, answer_idx = self.option_c(feats=llm_projected_features, labels=labels)

        return y, x_concat_y_e, answer_idx

    def forward(self, feats, labels):
        y, x_concat_y_e, answer_idx = self.preprocess_video(feats, labels)

        xp_concat_yp_e = self.lm(inputs_embeds=x_concat_y_e).logits
        loss_fct = CrossEntropyLoss(ignore_index=2)
        ### It only makes sense to train it for the answer ###
        shift_logits = xp_concat_yp_e[..., answer_idx - 1:-1, :].contiguous()
        shift_labels = y.contiguous()

        loss = loss_fct(shift_logits.view(-1, xp_concat_yp_e.size()[-1]), shift_labels.view(-1))
        ### ACC ###
        act_pred_x = torch.argmax(xp_concat_yp_e[:, answer_idx - 1:-1, :], dim=2).view(-1).detach().cpu().numpy()
        act_pred_y = y.view(-1).detach().cpu().numpy()

        correct_answers = 1.0 * (act_pred_x[np.where(act_pred_y != 2)] == act_pred_y[np.where(act_pred_y != 2)])
        eligible_answers = act_pred_y[np.where(act_pred_y != 2)].shape[0]
        unbiased_metric = np.sum(correct_answers) / eligible_answers

        return loss, unbiased_metric


def load_llm(opt_version='facebook/opt-125m'):
    tokenizer = GPT2Tokenizer.from_pretrained(opt_version)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"cls_token": "<|image|>"})
    tokenizer.add_tokens('[RET]')
    ret_token_idx = tokenizer('[RET]', add_special_tokens=False).input_ids
    assert len(ret_token_idx) == 1, ret_token_idx
    lm = OPTForCausalLM.from_pretrained(opt_version, torch_dtype=GLOBAL_DTYPE)
    for param in lm.parameters():
        param.requires_grad = False
        if param.ndim == 1:
            param.data = param.data.to(torch.float32)

    h_dim = lm.lm_head.weight.size()[1]

    class CastOutputToFloat(nn.Sequential):
        def forward(self, x): return super().forward(x).to(torch.float32)

    lm.lm_head = CastOutputToFloat(lm.lm_head)
    return lm, h_dim, tokenizer


def train_align(run_name='latest',
                epochs=5_000,
                llm='facebook/opt-125m',
                grad_clip=1,
                accum_steps=1,
                path=None,
                batch_size=64,
                val_freq=3_000,
                data_used='subset'):
    best_metric = 0
    ##################################################################################################################
    lm, h_dim, tokenizer = load_llm(llm)
    llms = llm.split('/')[-1]
    model = VideoAligner(lm=lm,
                         h_dim=h_dim,
                         tokenizer=tokenizer)
    if path is not None:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    model = model.to(GLOBAL_DEVICE)
    ################################################################################################################
    optimizer = torch.optim.AdamW(
        params=[
            {'params': model.parameters(), 'lr': 0.001, 'weight_decay': 0.001}
        ],
        betas=(0.99, 0.95),
        eps=1e-8
    )
    if GLOBAL_DEVICE == 'cuda':
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None
    optimizer.zero_grad()
    optimizer.step()

    ################################################################################################################
    train_dataset = VideoDS(sources=['UCF_101'], train_randomness_multiplier=1, split='train', data_used=data_used)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = VideoDS(sources=['UCF_101'], train_randomness_multiplier=1, split='val', data_used=data_used)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    if GLOBAL_DEVICE == 'cuda':
        env = torch.cuda.amp.autocast(dtype=GLOBAL_DTYPE)
    else:
        env = open('_rand.txt', 'w')
    with env:
        for epoch_idx in range(epochs):
            run_epoch_loss = 0
            run_epoch_metric = 0
            for step_idx, batch in enumerate(pbar := tqdm(train_dataloader)):
                real_index = (epoch_idx * len(train_dataloader)) + step_idx
                feats, gt_caption = batch
                feats = feats.to(GLOBAL_DEVICE)
                gt_caption = gt_caption.to(GLOBAL_DEVICE)

                loss, metric = model(feats=feats,
                                     labels=gt_caption)
                run_epoch_loss += loss.item()
                run_epoch_metric += metric
                pbar.set_postfix(
                    {'Epoch': epoch_idx, 'Step': real_index, 'Loss': run_epoch_loss / (step_idx + 1),
                     'Accuracy': run_epoch_metric / (step_idx + 1)})
                loss = loss / accum_steps
                if GLOBAL_DEVICE == 'cuda':
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if (real_index + 1) % accum_steps == 0:
                    if grad_clip > 0:
                        if GLOBAL_DEVICE == 'cuda':
                            scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    if GLOBAL_DEVICE == 'cuda':
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

                if real_index % val_freq == val_freq - 1:
                    val_loss, val_metric = validate_align(model=model, dataloader=val_dataloader)

                    if val_metric > best_metric:
                        save_checkpoint({
                            'state_dict': model.get_state_dict(),
                            'optimizer': optimizer.state_dict(),
                        },
                            f'../video_model_runs/{run_name}_step_{real_index}_llm_{llms}')

                        checkpoint = torch.load(
                            f'../video_model_runs/{run_name}_step_{real_index}_llm_{llms}.pth.tar')
                        model.load_state_dict(checkpoint['state_dict'], strict=False)
                        best_metric = val_metric
                    else:
                        pass

        # Save At End of Final Epoch #
        save_checkpoint({
            'state_dict': model.get_state_dict(),
            'optimizer': optimizer.state_dict(),
        }, f'../video_model_runs/{run_name}_step_{real_index}_llm_{llms}')


def validate_align(path=None,
                   llm=None, model=None, dataloader=None, data_used='subset'):
    if path is not None:
        if llm is None:
            print("LLM can not be None when loading from a pre-trained file!\n")
        checkpoint = torch.load(path)
        lm, h_dim, tokenizer = load_llm(llm)
        model = VideoAligner(lm=lm, h_dim=h_dim, tokenizer=tokenizer)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        print("Loading directly from passed model...\n")

    if dataloader is None:
        print("You need to pass a dataset you moron! I will equip the test split of UCF101!")
        dataset = VideoDS(sources=['UCF_101'], train_randomness_multiplier=1, split='test', data_used=data_used)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = model.to(GLOBAL_DEVICE)
    model.eval()
    if GLOBAL_DEVICE == 'cuda':
        env = torch.cuda.amp.autocast(dtype=GLOBAL_DTYPE)
    else:
        env = open('_rand.txt', 'w')
    with env:
        with torch.no_grad():
            run_val_epoch_loss = 0
            run_val_epoch_metric = 0
            for step_idx, batch in enumerate(pbar := tqdm(dataloader)):
                feats, gt_caption = batch
                feats = feats.to(GLOBAL_DEVICE)
                gt_caption = gt_caption.to(GLOBAL_DEVICE)
                loss, metric = model(feats=feats,
                                     labels=gt_caption)
                run_val_epoch_loss += loss.item()
                run_val_epoch_metric += metric
                pbar.set_postfix(
                    {'Val Metrics: Step': step_idx, 'Loss': run_val_epoch_loss / (step_idx + 1),
                     'Accuracy': run_val_epoch_metric / (step_idx + 1)})
    return run_val_epoch_loss / len(dataloader), run_val_epoch_metric / len(dataloader)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-llm', default='facebook/opt-125m')
    parser.add_argument('-path', default=None)
    parser.add_argument('-mode', default='train')
    parser.add_argument('-bs', default=8, type=int)
    parser.add_argument('-acs', default=4, type=int)
    parser.add_argument('-run_name', default='ucf_vmae')
    parser.add_argument('-data_used', default='subset')
    args = parser.parse_args()
    if args.mode == 'train':
        train_align(run_name=args.run_name,
                    path=args.path,
                    llm=args.llm,
                    batch_size=args.bs,
                    accum_steps=args.acs, data_used=args.data_used)
    else:
        validate_align(path=args.path, llm=args.llm, data_used=args.data_used)
