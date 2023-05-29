import os
import random
from argparse import ArgumentParser
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import OPTForCausalLM, GPT2Tokenizer
from tqdm import tqdm
from torch.utils.data import DataLoader

from wintorchsb3.video_expert_extract import VideoDS

CURRENT_DIR = os.getcwd()
GLOBAL_DTYPE = torch.bfloat16
GLOBAL_DEVICE = 'cuda'


def save_checkpoint(state, filename='checkpoint'):
    torch.save(state, filename + '.pth.tar')


class VideoAligner(nn.Module):
    def __init__(self,
                 lm,
                 h_dim,
                 tokenizer,
                 caption_loss=0, double_priest_loss_p=0.5):
        super().__init__()
        self.lm = lm
        self.tokenizer = tokenizer
        self.h_dim = h_dim
        self.prepare_models()
        self.initialize_trainable()
        self.initialize_prompt()
        self.caption_loss = caption_loss
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

        self.Simple_Solution = 'Someone is '
        self.TOKENIZED_Simple_Solution = self.tokenizer(self.Simple_Solution, add_special_tokens=False)['input_ids']
        target_tokens = torch.LongTensor(self.TOKENIZED_Simple_Solution).to(GLOBAL_DEVICE)
        self.EMB_Simple_Solution = torch.LongTensor(target_tokens).to(GLOBAL_DEVICE)


        self.DP_PRE_PROMPT_1 = 'This is video number 1.'
        self.TOKEMIZED_PROMPT_1 = self.tokenizer(self.DP_PRE_PROMPT_1, add_special_tokens=True)['input_ids']
        target_tokens = torch.LongTensor(self.TOKEMIZED_PROMPT_1).to(GLOBAL_DEVICE)
        self.EMB_PROMPT_1 = torch.LongTensor(target_tokens).to(GLOBAL_DEVICE)

        self.DP_PRE_PROMPT_2 = 'This is video number 2.'
        self.TOKEMIZED_PROMPT_2 = self.tokenizer(self.DP_PRE_PROMPT_2, add_special_tokens=False)['input_ids']
        target_tokens = torch.LongTensor(self.TOKEMIZED_PROMPT_2).to(GLOBAL_DEVICE)
        self.EMB_PROMPT_2 = torch.LongTensor(target_tokens).to(GLOBAL_DEVICE)

        self.DP_PRE_PROMPT_3 = 'Are the videos showing the same action?'
        self.TOKEMIZED_PROMPT_3 = self.tokenizer(self.DP_PRE_PROMPT_3, add_special_tokens=False)['input_ids']
        target_tokens = torch.LongTensor(self.TOKEMIZED_PROMPT_3).to(GLOBAL_DEVICE)
        self.EMB_PROMPT_3 = torch.LongTensor(target_tokens).to(GLOBAL_DEVICE)

        self.DP_PRE_PROMPT_4 = 'Which video is showing somebody '
        self.TOKEMIZED_PROMPT_4 = self.tokenizer(self.DP_PRE_PROMPT_4, add_special_tokens=False)['input_ids']
        target_tokens = torch.LongTensor(self.TOKEMIZED_PROMPT_4).to(GLOBAL_DEVICE)
        self.EMB_PROMPT_4 = torch.LongTensor(target_tokens).to(GLOBAL_DEVICE)

        self.Action_1 = 'applying eye makeup?'
        self.Action_2 = 'applying lipstick?'
        self.Action_3 = 'doing archery?'
        self.Action_4 = 'crawling?'
        self.Action_5 = 'balancing a beam?'
        self.Action_6 = 'marching in a band?'
        self.Action_7 = 'pitching at baseball?'
        self.Action_8 = 'playing basketball?'
        self.Action_9 = 'dunking at basketball?'
        self.Action_10 = 'bench press?'

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

        self.tokenized_action_registry = [self.TOKENIZED_Action_1, self.TOKENIZED_Action_2, self.TOKENIZED_Action_3,
                                          self.TOKENIZED_Action_4, self.TOKENIZED_Action_5, self.TOKENIZED_Action_6,
                                          self.TOKENIZED_Action_7, self.TOKENIZED_Action_8, self.TOKENIZED_Action_9,
                                          self.TOKENIZED_Action_10]
        self.emb_action_registry = [torch.LongTensor(torch.LongTensor(f).to(GLOBAL_DEVICE)).to(GLOBAL_DEVICE) for f in
                                    self.tokenized_action_registry]
        self.tokenized_solution_1 = self.tokenizer(' 1', add_special_tokens=False)['input_ids']
        self.tokenized_solution_2 = self.tokenizer(' 2', add_special_tokens=False)['input_ids']
        self.tokenized_solution_yes = self.tokenizer(' Yes', add_special_tokens=False)['input_ids']
        self.tokenized_solution_no = self.tokenizer(' No', add_special_tokens=False)['input_ids']

    def prepare_models(self):
        if self.config == 'LinearProjectActionToInput':
            pass
        elif self.config == 'LinearProjectFeaturesToInput':
            pass
        else:
            raise NotImplementedError

        self.lm.eval()
        self.lm = self.lm.to(GLOBAL_DEVICE)
        for param in self.lm.parameters():
            param.requires_grad = False

    def initialize_trainable(self):
        self.trainable_game_mode_token = nn.Parameter(torch.randn(4, self.h_dim), requires_grad=True)
        if self.config == 'LinearProjectActionToInput':
            self.trainable_projection = nn.Linear(in_features=6, out_features=self.h_dim)
            print(f"Initialized a Trainable Projection Layer from {6} to {self.h_dim} dims\n")
        elif self.config == 'LinearProjectFeaturesToInput':
            self.trainable_projection = nn.Linear(in_features=768, out_features=self.h_dim)
            print(f"Initialized a Trainable Projection Layer from {768} to {self.h_dim} dims\n")
        else:
            raise NotImplementedError
        self.trainable_module_names = ['trainable_game_mode_token', 'trainable_projection']
        return

    def option_a(self, feats, labels):
        """
        Mix feats per 2, and ask which one is doing label_a or label_b
        """
        batch_size = feats.size[0]
        half_bs = batch_size // 2
        trainable_add = self.trainable_game_mode_token.repeat(half_bs, 1, 1)
        feats_1 = feats[0:half_bs, :] + trainable_add
        feats_2 = feats[half_bs:, :] + trainable_add
        labels_1 = labels[0:half_bs]
        labels_2 = labels[half_bs:]

        ### This is video_number_1 [VIDEO_1] this is video_number_2 [VIDEO_2] ###
        prompt_pt_1 = [self.EMB_PROMPT_1.repeat(half_bs, 1, 1), feats_1, self.EMB_PROMPT_2.repeat(half_bs, 1, 1), feats_2]
        ### Which video is showing somebody [Label_1]
        prompt_pt_2 = [self.EMB_PROMPT_4.repeat(half_bs, 1, 1), self.emb_action_registry[labels_1]]
        gt_labels_1 = [self.tokenized_solution_1] * half_bs
        ### Which video is showing somebody [Label_2]
        prompt_pt_3 = [self.EMB_PROMPT_4.repeat(half_bs, 1, 1), self.emb_action_registry[labels_2]]
        gt_labels_2 = [self.tokenized_solution_2] * half_bs

        return

    def option_b(self, feats, labels):
        """
        Mix feats per 2, and ask if they are doing the same thing
        """
        batch_size = feats.size[0]
        half_bs = batch_size // 2
        trainable_add = self.trainable_game_mode_token.repeat(half_bs, 1, 1)
        feats_1 = feats[0:half_bs, :] + trainable_add
        feats_2 = feats[half_bs:, :] + trainable_add
        labels_1 = labels[0:half_bs]
        labels_2 = labels[half_bs:]

        ### This is video_number_1 [VIDEO_1] this is video_number_2 [VIDEO_2] ###
        prompt_pt_1 = [self.EMB_PROMPT_1.repeat(half_bs, 1, 1), feats_1, self.EMB_PROMPT_2.repeat(half_bs, 1, 1), feats_2]
        ### Are the videos showing the same action?
        prompt_pt_2 = [self.EMB_PROMPT_3.repeat(half_bs, 1, 1)]

        ### Compare labels_1 with labels_2 ###
        is_same = labels_1 == labels_2

        return

    def option_c(self, feats, labels):
        """
        Ask what's happening in the video
        """
        batch_size = feats.size[0]
        trainable_add = self.trainable_game_mode_token.repeat(batch_size, 1, 1)
        feats_1 = feats + trainable_add
        labels_1 = labels


        ### What is happening in this video? ###
        prompt_pt_1 = [feats_1, self.EMB_PRE_PROMPT.repeat(batch_size, 1, 1)]
        ### Someone is
        prompt_pt_2 = [self.EMB_Simple_Solution.repeat(batch_size, 1, 1)]

        ### Compare labels_1 with labels_2 ###
        gt = [self.emb_action_registry[labels_1]]

        return


    def preprocess_video(self, feats, labels):
        if self.config == 'LinearProjectFeaturesToInput':
            llm_projected_features = self.trainable_projection(feats)
        else:
            raise NotImplementedError

        if random.uniform(0, 1) < self.double_priest_loss_p:
            ### A: Mix 2 images and ask which is doing [Label] ###
            ### B: Mix 2 images and ask if they are showing the same thing [Label] ###
            prompt, gt_labels = self.option_a(llm_projected_features, labels)
            prompt, gt_labels = self.option_b(llm_projected_features, labels)
        else:
            ### C: Ask what is happening in the Video ###
            prompt, gt_labels = self.option_c(llm_projected_features, labels)
        return prompt, gt_labels

    def forward(self, feats, labels):
        batch_size = feats.size()[0]
        llm_projected_features = self.preprocess_video(feats, labels)

        inp_tokens = llm_projected_features
        out_logits = self.lm(inputs_embeds=inp_tokens).logits

        x = out_logits
        y = labels
        loss_fct = CrossEntropyLoss()
        if self.caption_loss == 1:
            shift_logits = x[..., self.trainable_game_mode_token.size()[0]:-1, :].contiguous()
            shift_labels = y[:, 1:].contiguous()
        else:
            shift_logits = x[..., -2:-1, :].contiguous()
            shift_labels = y[:, -1:].contiguous()

        loss = loss_fct(shift_logits.view(-1, x.size()[-1]), shift_labels.view(-1))
        act_pred_x = torch.argmax(x[:, -2:-1, :], dim=2).view(-1)
        act_pred_y = y[:, -1:].view(-1)
        metric = (act_pred_x == act_pred_y).float().sum().item() / batch_size
        return loss, metric, x


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
                epochs=100,
                llm='facebook/opt-125m',
                grad_clip=1,
                accum_steps=1,
                path=None,
                batch_size=64,
                val_freq=1_000,
                caption_loss=0):
    best_metric = 0
    ##################################################################################################################
    lm, h_dim, tokenizer = load_llm(llm)
    llms = llm.split('/')[-1]
    model = VideoAligner(lm=lm,
                         h_dim=h_dim,
                         tokenizer=tokenizer,
                         caption_loss=caption_loss)
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
    train_dataset = VideoDS(sources=['UCF_101'], train_randomness_multiplier=1, split='train')
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = VideoDS(sources=['UCF_101'], train_randomness_multiplier=1, split='val')
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
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

                loss, metric, _ = model(feats=feats,
                                        actions=gt_caption)
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
                    val_loss, val_metric = validate_align(path=None,
                                                          llm=None,
                                                          model=model,
                                                          dataset=val_dataloader)
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
                   llm=None):
    if path is not None:
        if llm is None:
            print("LLM can not be None when loading from a pre-trained file!\n")
        checkpoint = torch.load(path)
        lm, h_dim, tokenizer = load_llm(llm)
        model = VideoAligner(lm=lm, h_dim=h_dim, tokenizer=tokenizer, caption_loss=0)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        print("Loading directly from passed model...\n")
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
            for step_idx, batch in enumerate(pbar := tqdm(dataset)):
                feats, gt_caption = batch
                feats = feats.to(GLOBAL_DEVICE)
                gt_caption = gt_caption.to(GLOBAL_DEVICE)
                loss, metric, _ = model(feats=feats,
                                        actions=gt_caption)
                run_val_epoch_loss += loss.item()
                run_val_epoch_metric += metric
                pbar.set_postfix(
                    {'Val Metrics: Step': step_idx, 'Loss': run_val_epoch_loss / (step_idx + 1),
                     'Accuracy': run_val_epoch_metric / (step_idx + 1)})
    return run_val_epoch_loss / len(dataset), run_val_epoch_metric / len(dataset)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-llm', default='facebook/opt-125m')
    parser.add_argument('-path', default=None)
    parser.add_argument('-mode', default='train')
    parser.add_argument('-bs', default=8, type=int)
    parser.add_argument('-acs', default=32, type=int)
    parser.add_argument('-caption_loss', default=1, type=int)
    parser.add_argument('-run_name', default='ucf_vmae')
    args = parser.parse_args()
    if args.mode == 'train':
        train_align(run_name=args.run_name,
                    path=args.path,
                    llm=args.llm,
                    batch_size=args.bs,
                    accum_steps=args.acs,
                    caption_loss=args.caption_loss)
    else:
        validate_align(path=args.path, llm=args.llm)
