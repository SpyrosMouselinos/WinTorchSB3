import os
import random
from argparse import ArgumentParser
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import OPTForCausalLM, GPT2Tokenizer
from tqdm import tqdm

import numpy as np
from torch.utils.data import DataLoader

from expert_trace_extract import ExtractedModelPolicy, ppo_load_pong, MultiModalDS

#os.chdir('/'.join(os.path.dirname(__file__).split('/')[:-1]))
CURRENT_DIR = os.getcwd()
GLOBAL_DTYPE = torch.float32


def save_checkpoint(state, filename='checkpoint'):
    torch.save(state, filename + '.pth.tar')


class SupervisedAligner(nn.Module):
    def __init__(self, rl_model, rl_preprocess, lm, h_dim, tokenizer, config, normalize_actions):
        super().__init__()
        self.rl_model = rl_model
        self.rl_preprocess = rl_preprocess
        self.lm = lm
        self.tokenizer = tokenizer
        self.h_dim = h_dim
        self.config = config
        self.prepare_models()
        self.initialize_trainable()
        self.initialize_commands()
        self.initialize_game_prompt()
        self.normalize_actions = normalize_actions

    def get_state_dict(self, efficient=True):
        if not efficient:
            return self.state_dict()
        else:
            ret_dict = {}
            all_dict = self.state_dict()
            for key in self.trainable_module_names:
                ret_dict.update({key: all_dict[key]})
            return ret_dict

    def initialize_commands(self):
        game = 'Pong'
        if game == 'Pong':
            # self.COMMANDS = {
            #     0: 'NOOP',
            #     1: 'FIRE',
            #     2: 'RIGHT',
            #     3: 'LEFT',
            #     4: 'RIGHTFIRE',
            #     5: 'LEFTFIRE'
            #
            # }
            self.COMMANDS = {
                0: '0',
                1: '1',
                2: '2',
                3: '3',
                4: '4',
                5: '5'

            }
            self.LANG_COMMANDS = {v: k for k, v in self.COMMANDS.items()}

        else:
            raise NotImplementedError

        if self.tokenizer is not None:
            self.TOKENIZED_COMMANDS = {
                k: self.tokenizer(v, add_special_tokens=False)
                for k, v in self.COMMANDS.items()}

        return

    def initialize_game_prompt(self):
        game = 'Pong'
        if game == 'Pong':
            self.PROMPT = f'You are an agent that plays the game of {game}. What is the best move? '
            self.VAL_POS_PROMPT = f'You are a good agent that plays the game of {game}. What is a good move? '
            self.VAL_NEG_PROMPT = f'You are a bad agent that plays the game of {game}. What is a bad move? '
            self.TOKENIZED_PROMPT = self.tokenizer(self.PROMPT, add_special_tokens=True)
            self.TOKENIZED_VAL_POS_PROMPT = self.tokenizer(self.VAL_POS_PROMPT, add_special_tokens=True)
            self.TOKENIZED_VAL_NEG_PROMPT = self.tokenizer(self.VAL_NEG_PROMPT, add_special_tokens=True)

    def initialize_fewshot_prompt(self, oracle_guess=None, sent='neg'):
        game = 'Pong'
        if sent == 'neg':
            oracle_token = str(oracle_guess)
            return f'You are an agent that plays the game of {game}. The best move is {oracle_token}. What is a bad move? '
        elif sent == 'pos':
            if game == 'Pong':
                oracle_token = str(random.choice(list({0, 1, 2, 3, 4, 5}.difference(set([oracle_guess])))))
            return f'You are an agent that plays the game of {game}. A bad move is {oracle_token}. What is a good move? '
        else:
            raise NotImplementedError

    def prepare_models(self):
        if self.config == 'LinearProjectActionToInput':
            pass
        elif self.config == 'LinearProjectFeaturesToInput':
            self.rl_model = ExtractedModelPolicy(fe=self.rl_model.fe, an=self.rl_model.an)
        else:
            raise NotImplementedError

        self.rl_model.eval()
        self.rl_model.cuda()
        self.lm.eval()
        self.lm.cuda()
        for param in self.lm.parameters():
            param.requires_grad = False
        for param in self.rl_model.parameters():
            param.requires_grad = False

    def initialize_trainable(self):
        self.trainable_game_mode_token = nn.Parameter(torch.randn(self.h_dim), requires_grad=True)
        if self.config == 'LinearProjectActionToInput':
            self.trainable_projection = nn.Linear(in_features=6, out_features=self.h_dim)
            print(f"Initialized a Trainable Projection Layer from {6} to {self.h_dim} dims\n")
        elif self.config == 'LinearProjectFeaturesToInput':
            self.trainable_projection = nn.Linear(in_features=512, out_features=self.h_dim)
            print(f"Initialized a Trainable Projection Layer from {512} to {self.h_dim} dims\n")
        else:
            raise NotImplementedError
        self.trainable_module_names = ['trainable_game_mode_token', 'trainable_projection']
        return

    def translate_logits_to_button(self, responses):
        if isinstance(responses, list):
            responses = torch.stack(responses, dim=1)
        best_next_commands = self.tokenizer.batch_decode(responses)
        # print(best_next_commands)
        parallel_envs = len(best_next_commands)
        env_moves = []
        for env_id in range(parallel_envs):
            for command in self.LANG_COMMANDS.keys():
                if command in ''.join(best_next_commands[env_id]):
                    env_moves.append(self.LANG_COMMANDS[command])
            if len(env_moves) == env_id:
                env_moves.append(0)
        return env_moves

    def preprocess_game(self, feats, actions):
        tokenized_gt_action = torch.LongTensor(
            [self.TOKENIZED_COMMANDS[f.item()]['input_ids'] for f in
             torch.argmax(actions, dim=-1).detach().cpu()]).cuda()
        tokenized_ngt_action = torch.LongTensor(
            [self.TOKENIZED_COMMANDS[f.item()]['input_ids'] for f in
             torch.argmin(actions, dim=-1).detach().cpu()]).cuda()
        oracle_answer = torch.argmax(actions, dim=-1).detach().cpu().numpy()

        if self.normalize_actions:
            actions = actions / (torch.norm(actions, p=1) + 1e-6)
        if self.config == 'LinearProjectActionToInput':
            llm_projected_features = self.trainable_projection(actions)
        elif self.config == 'LinearProjectFeaturesToInput':
            llm_projected_features = self.trainable_projection(feats)
        else:
            raise NotImplementedError
        return tokenized_gt_action, tokenized_ngt_action, llm_projected_features, oracle_answer

    def preprocess_raw_game(self, obs):
        observation, _ = self.rl_preprocess(obs)
        with torch.no_grad():
            feats, actions = self.rl_model.predict(observation)
        # actions = actions.resize((-1, 6))
        return self.preprocess_game(feats, actions)

    def preprocess_question_train(self,
                                  batch_size,
                                  use_negative_prompt,
                                  tokenized_gt_action,
                                  tokenized_ngt_action):

        if not use_negative_prompt:
            feature_game_prompt = self.lm.model.decoder.embed_tokens(
                torch.LongTensor(self.TOKENIZED_PROMPT['input_ids']).cuda().unsqueeze(0))
            # Get representation of the tokenized gt action/s #
            feature_gt_action = self.lm.model.decoder.embed_tokens(tokenized_gt_action)
            target_tokens = torch.cat(
                [torch.LongTensor(self.TOKENIZED_PROMPT['input_ids']).repeat(batch_size, 1).cuda(),
                 tokenized_gt_action], dim=1)
        else:
            feature_game_prompt = self.lm.model.decoder.embed_tokens(
                torch.LongTensor(self.TOKENIZED_VAL_NEG_PROMPT['input_ids']).cuda().unsqueeze(0))
            # Get representation of the tokenized gt action/s #
            feature_gt_action = self.lm.model.decoder.embed_tokens(tokenized_ngt_action)
            target_tokens = torch.cat(
                [torch.LongTensor(self.TOKENIZED_VAL_NEG_PROMPT['input_ids']).repeat(batch_size, 1).cuda(),
                 tokenized_ngt_action], dim=1)
        return feature_game_prompt, feature_gt_action, target_tokens

    def preprocess_question_test(self,
                                 question,
                                 fewshot,
                                 sent,
                                 oracle_answer):
        if question is None:
            if fewshot:
                question = self.initialize_fewshot_prompt(oracle_guess=oracle_answer, sent=sent)
            else:
                if sent == 'pos':
                    question = self.TOKENIZED_VAL_POS_PROMPT['input_ids']
                elif sent == 'neg':
                    question = self.TOKENIZED_VAL_NEG_PROMPT['input_ids']
                elif sent == 'default':
                    question = self.TOKENIZED_PROMPT['input_ids']

        else:
            question = self.tokenizer(question, add_special_tokens=True)['input_ids']
        feature_game_prompt = self.lm.model.decoder.embed_tokens(
            torch.LongTensor(question).cuda().unsqueeze(
                0))
        return feature_game_prompt

    def forward(self, feats, actions, use_negative_prompt=False):
        batch_size = feats.size()[0]
        tokenized_gt_action, \
        tokenized_ngt_action, \
        llm_projected_features, _ = self.preprocess_game(feats, actions)

        feature_game_prompt, feature_gt_action, target_tokens = self.preprocess_question_train(batch_size=batch_size,
                                                                                               use_negative_prompt=use_negative_prompt,
                                                                                               tokenized_gt_action=tokenized_gt_action,
                                                                                               tokenized_ngt_action=tokenized_ngt_action)

        inp_tokens = torch.cat(
            [self.trainable_game_mode_token.repeat(batch_size, 1, 1),  # SL 1 / TL 1
             llm_projected_features.unsqueeze(1),  # SL 1 / TL 2
             feature_game_prompt.repeat(batch_size, 1, 1),  # SL 19 / TL 21
             feature_gt_action], dim=1)  # SL 1 / TL 22

        out_logits = self.lm(inputs_embeds=inp_tokens).logits

        x = out_logits
        y = target_tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = x[..., 1:-1, :].contiguous()
        shift_labels = y.contiguous()
        loss = loss_fct(shift_logits.view(-1, x.size()[-1]), shift_labels.view(-1))

        act_pred_x = torch.argmax(x[:, -(feature_gt_action.size()[1] + 1):-1, :], dim=2).view(-1)
        act_pred_y = torch.argmax(feature_gt_action, dim=2).view(-1)
        metric = (act_pred_x == act_pred_y).float().sum().item()
        return loss, metric, x

    def reason(self, obs, question=None, sent='pos', fewshot=False):
        batch_size = 1
        _, \
        _, \
        llm_projected_features, oracle_answer = self.preprocess_raw_game(obs)

        feature_game_prompt = self.preprocess_question_test(question=question,
                                                            fewshot=fewshot,
                                                            sent=sent,
                                                            oracle_answer=oracle_answer)
        inp_tokens = torch.cat(
            [self.trainable_game_mode_token.repeat(batch_size, 1, 1),
             llm_projected_features.unsqueeze(1),
             feature_game_prompt.repeat(batch_size, 1, 1)], dim=1)
        response_logits = []
        for i in range(3):
            out_logits = self.lm(inputs_embeds=inp_tokens).logits
            next_token = torch.argmax(out_logits[:, -1, :], -1)
            response_logits.append(next_token)
            inp_tokens = torch.cat([inp_tokens, self.lm.model.decoder.embed_tokens(next_token).unsqueeze(1)], dim=1)
        return response_logits, oracle_answer


###############


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


def play(model, env_name='PongNoFrameskip-v4', sent='pos', render=True, fewshot=False):
    env = make_atari_env(env_name, n_envs=1)
    env = VecFrameStack(env, n_stack=4)
    with torch.no_grad():
        obs = env.reset()
        episodic_reward = []
        while True:
            logits, _ = model.reason(obs, question=None, sent=sent, fewshot=fewshot)
            actions = model.translate_logits_to_button(logits)
            obs, rewards, dones, info = env.step(actions)
            episodic_reward.append(rewards)
            if render:
                env.render()
            ##############
            if not dones:
                pass
            else:
                print(np.stack(episodic_reward).sum())
                break
    return


def align(epochs=100,
          expert_steps=100_000,
          llm='facebook/opt-125m',
          grad_clip=1,
          accum_steps=1,
          path=None,
          batch_size=256,
          randomized_actions=0.05,
          inverse_prompt=0.5,
          log_freq=1000,
          save_freq=5000,
          n_jobs=4):
    ##################################################################################################################
    env, env_name, extracted_model, extracted_preprocessing = ppo_load_pong()
    lm, h_dim, tokenizer = load_llm(llm)
    model = SupervisedAligner(rl_model=extracted_model,
                              rl_preprocess=extracted_preprocessing,
                              lm=lm,
                              h_dim=h_dim,
                              tokenizer=tokenizer,
                              config='LinearProjectFeaturesToInput',
                              normalize_actions=False)
    if path is not None:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    ################################################################################################################
    optimizer = torch.optim.AdamW(
        params=[
            {'params': model.parameters(), 'lr': 0.001, 'weight_decay': 0.001}
        ],
        betas=(0.99, 0.95),
        eps=1e-8
    )
    scaler = torch.cuda.amp.GradScaler()
    optimizer.zero_grad()
    optimizer.step()

    ################################################################################################################
    save_checkpoint({
        'state_dict': model.get_state_dict(),
        'optimizer': optimizer.state_dict(),
    }, '../model_runs/zero_step')

    train_dataloader = DataLoader(
        MultiModalDS(sources=['Pong'],
                     n_examples=[expert_steps],
                     n_jobs=[n_jobs], args=[{'random_act_prob': randomized_actions}]),
        batch_size=batch_size,
        shuffle=False)
    with torch.cuda.amp.autocast(dtype=GLOBAL_DTYPE):
        for epoch_idx in range(epochs):
            for step_idx, batch in enumerate(pbar := tqdm(train_dataloader)):
                real_index = (epoch_idx * len(train_dataloader)) + step_idx
                _, raw, feats, gt_action_logits = batch
                if random.uniform(0, 1) < inverse_prompt:
                    use_neg_prompt = True
                else:
                    use_neg_prompt = False
                loss, metric, _ = model(feats=feats, actions=gt_action_logits, use_negative_prompt=use_neg_prompt)
                pbar.set_postfix({'Epoch': epoch_idx, 'Step': real_index, 'Loss': loss.item(), 'Accuracy': metric})
                loss = loss / accum_steps
                scaler.scale(loss).backward()
                if (real_index + 1) % accum_steps == 0:
                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                if real_index % log_freq == log_freq - 1:
                    print(f"\nLoss @ step {real_index}: {loss.item()} Metric @ step: {metric}")
                    if metric > 0.5:
                        print("\nPositive Score:")
                        play(model, env_name, sent='pos', render=False)
                        print("\nNegative Score:")
                        play(model, env_name, sent='neg', render=False)
                        print("\nFewshot positive:")
                        play(model, env_name, sent='pos', render=False, fewshot=True)
                        print("\nFewshot negative:")
                        play(model, env_name, sent='neg', render=False, fewshot=True)
                if real_index % save_freq == save_freq - 1:
                    save_checkpoint({
                        'state_dict': model.get_state_dict(),
                        'optimizer': optimizer.state_dict(),
                    },
                        f'../model_runs/step_{real_index}_neg_{inverse_prompt}_rand_{randomized_actions}_bs_{batch_size}_as_{accum_steps}')
            # Save At End of Epoch #
            save_checkpoint({
                'state_dict': model.get_state_dict(),
                'optimizer': optimizer.state_dict(),
                },f'../model_runs/step_{real_index}_neg_{inverse_prompt}_rand_{randomized_actions}_bs_{batch_size}_as_{accum_steps}')


def test_align(path=None, sent='pos', fewshot=False):
    env, extracted_model, extracted_preprocessing = ppo_load_pong()
    lm, h_dim, tokenizer = load_llm()
    model = SupervisedAligner(rl_model=extracted_model,
                              rl_preprocess=extracted_preprocessing,
                              lm=lm, h_dim=h_dim,
                              tokenizer=tokenizer,
                              config='LinearProjectFeaturesToInput',
                              normalize_actions=False)
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()
    model.eval()

    with torch.no_grad():
        obs = env.reset()
        episodic_reward = []
        mismatchs = []
        question = input()
        if question == 'None':
            question = None
        while True:
            response_logits, oracle_answer = model.reason(obs, question=question, sent=sent, fewshot=fewshot)
            actions = model.translate_logits_to_button(response_logits)
            # print(f"Action: {actions[0]} / Oracle Action: {oracle_answer}")
            if actions[0] != oracle_answer:
                mismatchs.append((actions[0], oracle_answer))
            obs, rewards, dones, info = env.step(actions)
            episodic_reward.append(rewards)
            env.render()
            ##############
            if not dones:
                pass
            else:
                print("Episodic Reward")
                print(sum(episodic_reward))
                episodic_reward = []


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-algo', default='ppo')
    parser.add_argument('-model_dir', default='downloads/ppo/PongNoFrameskip-v4_1/PongNoFrameskip-v4.zip')
    args = parser.parse_args()
    align(path=None)
