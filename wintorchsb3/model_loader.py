import os
import random
from argparse import ArgumentParser

import gym
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import OPTForCausalLM, GPT2Tokenizer
from tqdm import tqdm

import numpy as np
from torch.utils.data import DataLoader

from expert_trace_extract import ExtractedModelPolicy, ppo_load_pong, MultiModalDS, ppo_load

# os.chdir('/'.join(os.path.dirname(__file__).split('/')[:-1]))
CURRENT_DIR = os.getcwd()
GLOBAL_DTYPE = torch.bfloat16


# fp = 'b16'


def save_checkpoint(state, filename='checkpoint'):
    torch.save(state, filename + '.pth.tar')


class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self,
                 gamma_neg=1,
                 gamma_pos=1,
                 clip=0.05,
                 eps=1e-8,
                 disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum(-1).mean()


class SupervisedAligner(nn.Module):
    def __init__(self, lm, h_dim, tokenizer, config, normalize_actions, caption_loss=0,
                 games=None):
        super().__init__()
        self.games = {k: v for v, k in enumerate(games)}
        # self.rl_model = rl_model
        # self.rl_preprocess = rl_preprocess
        self.lm = lm
        self.tokenizer = tokenizer
        self.h_dim = h_dim
        self.config = config
        self.prepare_models()
        self.initialize_trainable()
        self.initialize_commands()
        self.initialize_game_prompt()
        self.normalize_actions = normalize_actions
        self.caption_loss = caption_loss

    def get_prompt(self, game, sent='pos', seed=0):
        pos_template_list = [
            'This is a frame from the game of xxx. You are an agent playing the game. What would be a good move to play? ',
            'You are playing the game of xxx. What would be your next move if you wanted to win? ',
            'As a xxx player, you are presented with this game frame. What would you do to win? ',
            'You are a xxx player and your opponent has just attacked you. What move do you make to win? ',
            'Imagine you are playing xxx. What action would you take to successfully respond to your opponent? '
        ]
        neg_template_list = [
            'This is a frame from the game of xxx. You are an agent playing the game. What would be a bad move to play? ',
            'As a xxx player, what would be the worst move you could make when trying to respond to your opponent? ',
            'You are an agent playing the game of xxx. What would be a bad move to play if you want to win the game? ',
            'In this game of xxx, what is the biggest mistake a player can make when trying to score points? ',
            'Imagine you are playing xxx. What action would be a bad response to your opponent? '
        ]
        list_ = pos_template_list if sent == 'pos' else neg_template_list
        prompt = list_[seed]
        prompt = prompt.replace('xxx', game)
        return prompt

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

    def initialize_commands(self):
        self.COMMANDS = {
            0: '0',
            1: '1',
            2: '2',
            3: '3',
            4: '4',
            5: '5'

        }
        self.LANG_COMMANDS = {v: k for k, v in self.COMMANDS.items()}
        if self.tokenizer is not None:
            self.TOKENIZED_COMMANDS = {
                k: self.tokenizer(v, add_special_tokens=False)
                for k, v in self.COMMANDS.items()}

        return

    def initialize_game_prompt(self):
        # self.PRE_PROMPT = 'Game:'
        # self.TOKENIZED_PRE_PROMPT = self.tokenizer(self.PRE_PROMPT, add_special_tokens=True)['input_ids']

        self.VAL_POS_PROMPT = [f'You are an agent that plays {game}. What is a good move to play? ' for game in
                               self.games]
        self.TOKENIZED_VAL_POS_PROMPT = [self.tokenizer(f, add_special_tokens=True)['input_ids'] for f in
                                         self.VAL_POS_PROMPT]
        self.VAL_NEG_PROMPT = [f'You are an agent that plays {game}. What is a move you should not play? ' for
                               game
                               in
                               self.games]
        self.TOKENIZED_VAL_NEG_PROMPT = [self.tokenizer(f, add_special_tokens=True)['input_ids'] for f in
                                         self.VAL_NEG_PROMPT]

    def initialize_fewshot_prompt(self, game, oracle_guess=None, sent='neg'):
        if sent == 'neg':
            oracle_token = str(oracle_guess)
            return f'You are an agent that plays {game}. The best move to play is Move: {oracle_token}. What is a bad move? '
        elif sent == 'pos':
            oracle_token = str(random.choice(list({0, 1, 2, 3, 4, 5}.difference(set(oracle_guess)))))
            return f'You are an agent that plays {game}. A bad move to play is Move: {oracle_token}. What is a good move? '
        else:
            raise NotImplementedError

    def prepare_models(self):
        if self.config == 'LinearProjectActionToInput':
            pass
        elif self.config == 'LinearProjectFeaturesToInput':
            pass
        else:
            raise NotImplementedError

        self.lm.eval()
        self.lm.cuda()
        for param in self.lm.parameters():
            param.requires_grad = False

    def initialize_trainable(self):
        self.trainable_game_mode_token = nn.Parameter(torch.randn(4, self.h_dim), requires_grad=True)
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
        return env_moves, best_next_commands

    def semantic_swap(self, bnm):
        mapping = {
            0: 5,
            1: 2,
            2: 1,
            3: 4,
            4: 3,
            5: 0
        }
        swap = np.array([mapping[f] for f in bnm])
        return swap

    def preprocess_game(self, feats, actions):
        oracle_answer = torch.argmax(actions, dim=-1).detach().cpu().numpy()
        worst_next_move = self.semantic_swap(oracle_answer)
        tokenized_gt_action = [self.TOKENIZED_COMMANDS[f]['input_ids'] for f in oracle_answer]
        tokenized_ngt_action = [self.TOKENIZED_COMMANDS[f]['input_ids'] for f in worst_next_move]
        if self.normalize_actions:
            actions = actions / (torch.norm(actions, p=1) + 1e-6)
        if self.config == 'LinearProjectActionToInput':
            llm_projected_features = self.trainable_projection(actions)
        elif self.config == 'LinearProjectFeaturesToInput':
            llm_projected_features = self.trainable_projection(feats)
        else:
            raise NotImplementedError
        return tokenized_gt_action, tokenized_ngt_action, llm_projected_features, oracle_answer

    def preprocess_raw_game(self, obs, extracted_model, extracted_preprocessing):
        observation = extracted_preprocessing(obs)
        with torch.no_grad():
            feats, actions = extracted_model.predict(observation)
        return self.preprocess_game(feats, actions)

    def preprocess_question_train(self,
                                  games,
                                  use_negative_prompt,
                                  tokenized_gt_action,
                                  tokenized_ngt_action):

        if not use_negative_prompt:
            seed = random.choice([0, 1, 2, 3, 4])
            prompt = [self.get_prompt(game=f, sent='pos', seed=seed) for f in games]
            tok_prompt = [self.tokenizer(f, add_special_tokens=True)['input_ids'] for f in prompt]
            max_pad_length = max(len(i) for i in tok_prompt) + 1
            for i in range(len(tok_prompt)):
                tok_prompt[i].append(tokenized_gt_action[i][0])
                if max_pad_length - len(tok_prompt[i]) > 0:
                    pad_len = max_pad_length - len(tok_prompt[i])
                    for _ in range(pad_len):
                        tok_prompt[i].append(1)
            target_tokens = torch.LongTensor(tok_prompt).cuda()
            feature_game_prompt = self.lm.model.decoder.embed_tokens(target_tokens)

        else:
            seed = random.choice([0, 1, 2, 3, 4])
            prompt = [self.get_prompt(game=f, sent='neg', seed=seed) for f in games]
            tok_prompt = [self.tokenizer(f, add_special_tokens=True)['input_ids'] for f in prompt]
            max_pad_length = max(len(i) for i in tok_prompt) + 1
            for i in range(len(tok_prompt)):
                tok_prompt[i].append(tokenized_ngt_action[i][0])
                if max_pad_length - len(tok_prompt[i]) > 0:
                    pad_len = max_pad_length - len(tok_prompt[i])
                    for _ in range(pad_len):
                        tok_prompt[i].append(1)
            target_tokens = torch.LongTensor(tok_prompt).cuda()
            feature_game_prompt = self.lm.model.decoder.embed_tokens(target_tokens)
        return feature_game_prompt, target_tokens

    def preprocess_question_test(self,
                                 game,
                                 question,
                                 fewshot,
                                 sent,
                                 oracle_answer):
        if question is None:
            if fewshot:
                question = self.initialize_fewshot_prompt(oracle_guess=oracle_answer, sent=sent, game=game)
                question = self.tokenizer(question, add_special_tokens=False)['input_ids']
            else:
                if sent == 'pos':
                    question = self.TOKENIZED_VAL_POS_PROMPT[0]
                elif sent == 'neg':
                    question = self.TOKENIZED_VAL_NEG_PROMPT[0]

        else:
            question = self.tokenizer(question, add_special_tokens=False)['input_ids']
        feature_game_prompt = self.lm.model.decoder.embed_tokens(
            torch.LongTensor(question).cuda().unsqueeze(
                0))
        return feature_game_prompt

    def forward(self, games, feats, actions, use_negative_prompt=False):
        batch_size = feats.size()[0]
        tokenized_gt_action, \
        tokenized_ngt_action, \
        llm_projected_features, _ = self.preprocess_game(feats, actions)

        feature_game_prompt, target_tokens = self.preprocess_question_train(games=games,
                                                                            use_negative_prompt=use_negative_prompt,
                                                                            tokenized_gt_action=tokenized_gt_action,
                                                                            tokenized_ngt_action=tokenized_ngt_action)
        fusion_tokens = self.trainable_game_mode_token.repeat(batch_size, 1, 1) + llm_projected_features.unsqueeze(1)
        inp_tokens = torch.cat(
            [fusion_tokens,
             feature_game_prompt,
             ], dim=1)

        out_logits = self.lm(inputs_embeds=inp_tokens).logits

        x = out_logits
        y = target_tokens
        loss_fct = CrossEntropyLoss()
        if self.caption_loss == 1:
            shift_logits = x[..., 4:-1, :].contiguous()
            shift_labels = y.contiguous()
        else:
            shift_logits = x[..., -2:-1, :].contiguous()
            shift_labels = y[:, -1:].contiguous()

        loss = loss_fct(shift_logits.view(-1, x.size()[-1]), shift_labels.view(-1))
        act_pred_x = torch.argmax(x[:, -2:-1, :], dim=2).view(-1)
        act_pred_y = y[:, -1:].view(-1)
        metric = (act_pred_x == act_pred_y).float().sum().item() / batch_size
        return loss, metric, x

    def reason(self, obs, game, extracted_model, extracted_preprocessing, question=None, sent='pos', fewshot=False):
        batch_size = 1
        _, \
        _, \
        llm_projected_features, oracle_answer = self.preprocess_raw_game(obs, extracted_model, extracted_preprocessing)

        feature_game_prompt = self.preprocess_question_test(game=game,
                                                            question=question,
                                                            fewshot=fewshot,
                                                            sent=sent,
                                                            oracle_answer=oracle_answer)
        inp_tokens = torch.cat(
            [self.trainable_game_mode_token.repeat(batch_size, 1, 1),
             llm_projected_features.unsqueeze(1),
             feature_game_prompt], dim=1)
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
    lm.gradient_checkpointing_enable()
    return lm, h_dim, tokenizer


def play(model, game='Pong', sent='pos', render=True, fewshot=False):
    env, _, extracted_model, extracted_preprocessing, _ = ppo_load(game)
    with torch.no_grad():
        obs = env.reset()
        episodic_reward = []
        while True:
            logits, _ = model.reason(obs, game, extracted_model, extracted_preprocessing, question=None, sent=sent,
                                     fewshot=fewshot)
            actions, best_next_commands = model.translate_logits_to_button(logits)
            obs, rewards, dones, info = env.step(actions)
            episodic_reward.append(rewards)
            if render:
                print(best_next_commands)
                env.render()
            ##############
            if not dones:
                pass
            else:
                print(f"Game:{game}, Reward:{np.stack(episodic_reward).sum()}")
                env.close()
                break
    return


def align(run_name='latest',
          epochs=200,
          n_games=1,
          llm='facebook/opt-125m',
          grad_clip=1,
          accum_steps=1,
          path=None,
          batch_size=64,
          randomized_actions=0.01,
          inverse_prompt=0.25,
          log_freq=500,
          save_freq=5000,
          caption_loss=0, games=['Breakout', 'Pong']):
    ##################################################################################################################
    lm, h_dim, tokenizer = load_llm(llm)
    llms = llm.split('/')[-1]
    model = SupervisedAligner(lm=lm,
                              h_dim=h_dim,
                              tokenizer=tokenizer,
                              config='LinearProjectFeaturesToInput',
                              normalize_actions=False, caption_loss=caption_loss, games=games)
    if path is not None:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    model.cuda()
    ################################################################################################################
    optimizer = torch.optim.AdamW(
        params=[
            {'params': model.parameters(), 'lr': 0.005, 'weight_decay': 0.001}
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
        MultiModalDS(sources=games,
                     n_games=[n_games, n_games],
                     args=[{'random_act_prob': randomized_actions}, {'random_act_prob': randomized_actions}]),
        batch_size=batch_size,
        shuffle=True)
    with torch.cuda.amp.autocast(dtype=GLOBAL_DTYPE):
        for epoch_idx in range(epochs):
            run_epoch_loss = 0
            run_epoch_metric = 0
            for step_idx, batch in enumerate(pbar := tqdm(train_dataloader)):
                real_index = (epoch_idx * len(train_dataloader)) + step_idx
                games, raw, feats, gt_action_logits = batch
                if random.uniform(0, 1) < inverse_prompt:
                    use_neg_prompt = True
                else:
                    use_neg_prompt = False
                loss, metric, _ = model(games=games,
                                        feats=feats,
                                        actions=gt_action_logits,
                                        use_negative_prompt=use_neg_prompt)
                run_epoch_loss += loss.item()
                run_epoch_metric += metric
                pbar.set_postfix(
                    {'Epoch': epoch_idx, 'Step': real_index, 'Loss': run_epoch_loss / (step_idx + 1),
                     'Accuracy': run_epoch_metric / (step_idx + 1)})
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
                    if run_epoch_metric / (step_idx + 1) > 0.7:

                        for game in ['Pong', 'Breakout']:
                            print(f"\nGame: {game} Positive Score:")
                            play(model, game, sent='pos', render=False)
                            print(f"\nGame: {game} Negative Score:")
                            play(model, game, sent='neg', render=False)
                            print(f"\nGame: {game} Fewshot positive:")
                            play(model, game, sent='pos', render=False, fewshot=True)
                            print(f"\nGame: {game} Fewshot negative:")
                            play(model, game, sent='neg', render=False, fewshot=True)
                if real_index % save_freq == save_freq - 1:
                    save_checkpoint({
                        'state_dict': model.get_state_dict(),
                        'optimizer': optimizer.state_dict(),
                    },
                        f'../model_runs/{run_name}_step_{real_index}_llm_{llms}')
        # Save At End of Final Epoch #
        save_checkpoint({
            'state_dict': model.get_state_dict(),
            'optimizer': optimizer.state_dict(),
        }, f'../model_runs/{run_name}_step_{real_index}_llm_{llms}')


def test_align(path=None, sent='default', fewshot=False, fixed_question=None, llm=None, games=['Pong']):
    lm, h_dim, tokenizer = load_llm(llm)
    model = SupervisedAligner(lm=lm,
                              h_dim=h_dim,
                              tokenizer=tokenizer,
                              config='LinearProjectFeaturesToInput',
                              normalize_actions=False, caption_loss=0, games=games)
    if path is not None:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        print("YOU DIDNT LOAD SHIT!!")
    model.cuda()
    model.eval()
    count = 0
    for game in games:
        env, _, extracted_model, extracted_preprocessing, _ = ppo_load(game)
        env = gym.wrappers.RecordVideo(env, f"./videos/{game}/", step_trigger=lambda x: x % 100 == 0)
        with torch.no_grad():
            obs = env.reset()
            episodic_reward = []
            mismatchs = []
            question = fixed_question
            if question == 'None':
                question = None
            while count < 1:
                # print("Looping...\n", flush=True)
                response_logits, oracle_answer = model.reason(obs, game,
                                                              extracted_model,
                                                              extracted_preprocessing,
                                                              question=None, sent=sent,
                                                              fewshot=fewshot)
                if random.uniform(0, 1) < 0.01:
                    actions = [env.action_space.sample()]
                else:
                    actions = oracle_answer
                # actions = model.translate_logits_to_button(response_logits)
                # print(f"Action: {actions[0]} / Oracle Action: {oracle_answer}")
                obs, rewards, dones, info = env.step(actions)
                episodic_reward.append(rewards)
                env.render()
                ##############
                if not dones:
                    pass
                else:
                    print(f"Game:{game} Episodic Reward", flush=True)
                    print(sum(episodic_reward), flush=True)
                    episodic_reward = []
                    count += 1
                    env.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-algo', default='ppo')
    parser.add_argument('-llm', default='facebook/opt-125m')
    parser.add_argument('-model_dir', default='downloads/ppo/PongNoFrameskip-v4_1/PongNoFrameskip-v4.zip')
    parser.add_argument('-path', default=None)
    parser.add_argument('-mode', default='train')
    parser.add_argument('-test_sent', default='default')
    parser.add_argument('-test_fewshot', default='0')
    parser.add_argument('-bs', default=64, type=int)
    parser.add_argument('-acs', default=1, type=int)
    parser.add_argument('-caption_loss', default=0, type=int)
    parser.add_argument('-run_name', default='test')
    args = parser.parse_args()
    if args.mode == 'train':
        align(run_name=args.run_name, path=args.path, llm=args.llm, batch_size=args.bs, accum_steps=args.acs,
              caption_loss=args.caption_loss)
    else:
        test_align(path=args.path, llm=args.llm, sent=args.test_sent,
                   fewshot=True if args.test_fewshot == '1' else False)
