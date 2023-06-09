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
GLOBAL_DEVICE = 'cuda'


# fp = 'b16'

def most_frequent(List):
    return max(set(List), key=List.count)


def countFreq(pat, txt):
    M = len(pat)
    N = len(txt)
    res = 0

    # A loop to slide pat[] one by one
    for i in range(N - M + 1):

        # For current index i, check
        # for pattern match
        j = 0
        while j < M:
            if (txt[i + j] != pat[j]):
                break
            j += 1

        if (j == M):
            res += 1
            j = 0
    return res


def save_checkpoint(state, filename='checkpoint'):
    torch.save(state, filename + '.pth.tar')


class SupervisedAligner(nn.Module):
    def __init__(self, lm, h_dim, tokenizer, config, normalize_actions, caption_loss=0,
                 games=None):
        super().__init__()
        self.games = {k: v for v, k in enumerate(games)}
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
        # seed = 0
        pos_template_list = [
            # 'You are an agent that plays xxx. What is a good move to play to win? '
            'This is a frame from the game of xxx. You are an agent playing the game. What would be a good move to play? ',
            'This is a frame from the game of xxx. You are an agent playing the game. What would be a not bad move to play? ',
            'You are playing the game of xxx. What would be your next move if you wanted to win? ',
            'As a xxx player, you are presented with this game frame. What would you do to win? ',
            'You are a xxx player and your opponent has just attacked you. What move do you make to win? ',
            'Imagine you are playing xxx. What action would you take to successfully respond to your opponent? ',
            'Imagine you are playing xxx. What action would be optimal? '
        ]
        neg_template_list = [
            # 'You are an agent that plays xxx. What is a bad move to play? '
            'This is a frame from the game of xxx. You are an agent playing the game. What would be a bad move to play? ',
            'As a xxx player, what would be the worst move you could make when trying to respond to your opponent? ',
            'You are playing the game of xxx. What would be your next move if you wanted to lose? ',
            'You are an agent playing the game of xxx. What would be a not good move to play if you want to win the game? ',
            'In this game of xxx, what is the biggest mistake a player can make when trying to score points? ',
            'Imagine you are playing xxx. What action would be a bad response to your opponent? ',
            'Imagine you are playing xxx. What action would be a not optimal response to your opponent? '
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

        self.VAL_POS_PROMPT = [f'You are an agent that plays {game}. What is a good move to play to win? ' for game in
                               self.games]
        self.TOKENIZED_VAL_POS_PROMPT = [self.tokenizer(f, add_special_tokens=True)['input_ids'] for f in
                                         self.VAL_POS_PROMPT]
        self.VAL_NEG_PROMPT = [f'You are an agent that plays {game}. What is a bad move to play? ' for
                               game
                               in
                               self.games]
        self.TOKENIZED_VAL_NEG_PROMPT = [self.tokenizer(f, add_special_tokens=True)['input_ids'] for f in
                                         self.VAL_NEG_PROMPT]

    def initialize_fewshot_prompt(self, game, oracle_guess=None, sent='neg'):
        if sent == 'neg':
            oracle_token = str(oracle_guess)
            return f'You are an agent that plays {game}. Playing {oracle_token} would win the game. What would be your worst move? '
        elif sent == 'pos':
            oracle_token = str(random.choice(list({0, 1, 2, 3, 4, 5}.difference(set(oracle_guess)))))
            return f'You are an agent that plays {game}. Playing {oracle_token} would not win the game. What is your best move? '
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

    def translate_logits_to_button(self, responses, use_majority_vote=False):
        if isinstance(responses, list):
            responses = torch.stack(responses, dim=1)
        best_next_commands = self.tokenizer.batch_decode(responses)
        # print(best_next_commands)
        parallel_envs = len(best_next_commands)
        env_moves = []
        for env_id in range(parallel_envs):
            for command in self.LANG_COMMANDS.keys():
                for _ in range(countFreq(command, ''.join(best_next_commands[env_id]))):
                    env_moves.append(self.LANG_COMMANDS[command])
            if len(env_moves) == env_id:
                env_moves.append(0)
            elif use_majority_vote:
                env_moves = [most_frequent(env_moves)]
            else:
                env_moves = [env_moves[0]]
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

    def preprocess_raw_game(self, obs, extracted_model, extracted_preprocessing, feature_pipeline=None, use_vfe=None):
        if use_vfe:
            img = obs[0]
            obs = obs[1]
        observation = extracted_preprocessing(obs)
        with torch.no_grad():
            feats, actions = extracted_model.predict(observation)
            if use_vfe:
                feats = feature_pipeline(img)
        return self.preprocess_game(feats, actions)

    def preprocess_question_train(self,
                                  games,
                                  use_negative_prompt,
                                  tokenized_gt_action,
                                  tokenized_ngt_action):

        if not use_negative_prompt:
            seed = random.choice([0, 1, 2, 3, 4, 5, 6])
            prompt = [self.get_prompt(game=f, sent='pos', seed=seed) for f in games]
            tok_prompt = [self.tokenizer(f, add_special_tokens=True)['input_ids'] for f in prompt]
            max_pad_length = max(len(i) for i in tok_prompt) + 1
            for i in range(len(tok_prompt)):
                tok_prompt[i].append(tokenized_gt_action[i][0])
                if max_pad_length - len(tok_prompt[i]) > 0:
                    pad_len = max_pad_length - len(tok_prompt[i])
                    for _ in range(pad_len):
                        tok_prompt[i].append(1)
            target_tokens = torch.LongTensor(tok_prompt).to(GLOBAL_DEVICE)
            feature_game_prompt = self.lm.model.decoder.embed_tokens(target_tokens)

        else:
            seed = random.choice([0, 1, 2, 3, 4, 5, 6])
            prompt = [self.get_prompt(game=f, sent='neg', seed=seed) for f in games]
            tok_prompt = [self.tokenizer(f, add_special_tokens=True)['input_ids'] for f in prompt]
            max_pad_length = max(len(i) for i in tok_prompt) + 1
            for i in range(len(tok_prompt)):
                tok_prompt[i].append(tokenized_ngt_action[i][0])
                if max_pad_length - len(tok_prompt[i]) > 0:
                    pad_len = max_pad_length - len(tok_prompt[i])
                    for _ in range(pad_len):
                        tok_prompt[i].append(1)
            target_tokens = torch.LongTensor(tok_prompt).to(GLOBAL_DEVICE)
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
                question = self.tokenizer(question, add_special_tokens=True)['input_ids']
            else:
                if sent == 'pos':
                    question = self.TOKENIZED_VAL_POS_PROMPT[0]
                elif sent == 'neg':
                    question = self.TOKENIZED_VAL_NEG_PROMPT[0]


        else:
            question = self.tokenizer(question, add_special_tokens=False)['input_ids']
        feature_game_prompt = self.lm.model.decoder.embed_tokens(
            torch.LongTensor(question).unsqueeze(
                0).to(GLOBAL_DEVICE))
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
        if len(llm_projected_features.size()) < 3:
            llm_projected_features = llm_projected_features.unsqueeze(1)
        fusion_tokens = self.trainable_game_mode_token.repeat(batch_size, 1, 1) + llm_projected_features
        inp_tokens = torch.cat(
            [fusion_tokens,
             feature_game_prompt,
             ], dim=1)

        out_logits = self.lm(inputs_embeds=inp_tokens).logits

        x = out_logits
        y = target_tokens
        loss_fct = CrossEntropyLoss()
        if self.caption_loss == 1:
            shift_logits = x[..., fusion_tokens.size()[1]:-1, :].contiguous()
            shift_labels = y[:, 1:].contiguous()
        else:
            shift_logits = x[..., -2:-1, :].contiguous()
            shift_labels = y[:, -1:].contiguous()

        loss = loss_fct(shift_logits.view(-1, x.size()[-1]), shift_labels.view(-1))
        act_pred_x = torch.argmax(x[:, -2:-1, :], dim=2).view(-1)
        act_pred_y = y[:, -1:].view(-1)
        metric = (act_pred_x == act_pred_y).float().sum().item() / batch_size
        return loss, metric, x

    def reason(self, obs, game, extracted_model, extracted_preprocessing, question=None, sent='pos', fewshot=False,
               feature_pipeline=None, use_vfe=None):
        batch_size = 1
        _, \
        _, \
        llm_projected_features, oracle_answer = self.preprocess_raw_game(obs, extracted_model, extracted_preprocessing,
                                                                         feature_pipeline=feature_pipeline,
                                                                         use_vfe=use_vfe)

        feature_game_prompt = self.preprocess_question_test(game=game,
                                                            question=question,
                                                            fewshot=fewshot,
                                                            sent=sent,
                                                            oracle_answer=oracle_answer)
        if len(llm_projected_features.size()) < 3:
            llm_projected_features = llm_projected_features.unsqueeze(0)
        fusion_tokens = self.trainable_game_mode_token.repeat(batch_size, 1, 1) + llm_projected_features
        inp_tokens = torch.cat(
            [fusion_tokens,
             feature_game_prompt,
             ], dim=1)
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
    # lm.gradient_checkpointing_enable()
    return lm, h_dim, tokenizer


def play(model, game='Pong', sent='pos', render=True, fewshot=False, feature_pipeline=None, use_vfe=None):
    dual = not (use_vfe is None)
    env, _, extracted_model, extracted_preprocessing, _ = ppo_load(game, dual=dual)
    with torch.no_grad():
        if dual:
            obs = (env[0].reset(), env[1].reset())
        else:
            obs = env.reset()
        episodic_reward = []
        while True:
            logits, _ = model.reason(obs, game, extracted_model, extracted_preprocessing, question=None, sent=sent,
                                     feature_pipeline=feature_pipeline,
                                     fewshot=fewshot, use_vfe=use_vfe)
            actions, best_next_commands = model.translate_logits_to_button(logits, use_majority_vote=False)
            if dual:
                obs_c, rewards_1, dones_1, _ = env[0].step(actions[0])
                obs_g, rewards_2, dones_2, _ = env[1].step(actions[0])
                assert rewards_1 == rewards_2
                assert dones_1 == dones_2
                rewards = rewards_1
                dones = dones_1
                obs = (obs_c, obs_g)
            else:
                obs, rewards, dones, _ = env.step(actions[0])
            episodic_reward.append(rewards)
            if render:
                print(best_next_commands)
                if dual:
                    env[1].render('human')
                else:
                    env.render('human')
            ##############
            if not dones:
                pass
            else:
                print(f"Game:{game}, Reward:{np.stack(episodic_reward).sum()}")
                if dual:
                    env[0].close()
                    env[1].close()
                else:
                    env.close()
                break
    return


def align(run_name='latest',
          epochs=100,
          n_games=1,
          llm='facebook/opt-125m',
          grad_clip=1,
          accum_steps=1,
          path=None,
          batch_size=64,
          randomized_actions=0.0,
          inverse_prompt=0.25,
          log_freq=1_000,
          save_freq=2_000,
          caption_loss=0, games=['Pong'], use_vfe=None):
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
    train_dataset = MultiModalDS(sources=games,
                                 n_games=[n_games],
                                 args=[{'random_act_prob': randomized_actions, 'dual': not (use_vfe is None),
                                        'feature_pipeline': use_vfe}])
    feature_pipeline = train_dataset.get_pipeline(use_vfe)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True)
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
                games, feats, gt_action_logits = batch
                feats = feats.to(GLOBAL_DEVICE)
                gt_action_logits = gt_action_logits.to(GLOBAL_DEVICE)
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

                if real_index % log_freq == log_freq - 1:

                    print(f"\nLoss @ step {real_index}: {loss.item()} Metric @ step: {metric}")
                    if run_epoch_metric / (step_idx + 1) > 0.9:

                        for game in ['Pong']:
                            print(f"\nGame: {game} Positive Score:")
                            play(model, game, sent='pos', render=False, feature_pipeline=feature_pipeline,
                                 use_vfe=use_vfe)
                            print(f"\nGame: {game} Negative Score:")
                            play(model, game, sent='neg', render=False, feature_pipeline=feature_pipeline,
                                 use_vfe=use_vfe)
                            print(f"\nGame: {game} Fewshot positive:")
                            play(model, game, sent='pos', render=False, feature_pipeline=feature_pipeline,
                                 use_vfe=use_vfe)
                            print(f"\nGame: {game} Fewshot negative:")
                            play(model, game, sent='neg', render=False, feature_pipeline=feature_pipeline,
                                 use_vfe=use_vfe)
                if real_index % save_freq == save_freq - 1:
                    save_checkpoint({
                        'state_dict': model.get_state_dict(),
                        'optimizer': optimizer.state_dict(),
                    },
                        f'../model_runs/{run_name}_step_{real_index}_llm_{llms}_vfe_{use_vfe}')
                    print("Loading from Last CHKPT...\n")
                    checkpoint = torch.load(
                        f'../model_runs/{run_name}_step_{real_index}_llm_{llms}_vfe_{use_vfe}.pth.tar')
                    model.load_state_dict(checkpoint['state_dict'], strict=False)

        # Save At End of Final Epoch #
        save_checkpoint({
            'state_dict': model.get_state_dict(),
            'optimizer': optimizer.state_dict(),
        }, f'../model_runs/{run_name}_step_{real_index}_llm_{llms}_vfe_{use_vfe}')


def pest_align(path=None, sent='default', fewshot=False, fixed_question=None, llm=None, record=False,
               games=['Breakout'], use_vfe=None):
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
    model = model.to(GLOBAL_DEVICE)
    model.eval()
    count = 0
    if GLOBAL_DEVICE == 'cuda':
        env = torch.cuda.amp.autocast(dtype=GLOBAL_DTYPE)
    else:
        env = open('_rand.txt', 'w')
    with env:
        for game in games:
            env, _, extracted_model, extracted_preprocessing, _ = ppo_load(game)
            if record:
                env = gym.wrappers.RecordVideo(env, f"./videos/{game}/", step_trigger=lambda x: x % 100 == 0)
            with torch.no_grad():
                obs = env.reset()
                episodic_reward = []
                question = fixed_question
                if question == 'None':
                    question = None
                while count < 1:
                    # print("Looping...\n", flush=True)
                    response_logits, oracle_answer = model.reason(obs, game,
                                                                  extracted_model,
                                                                  extracted_preprocessing,
                                                                  question=None,
                                                                  sent=sent,
                                                                  fewshot=fewshot)
                    # if random.uniform(0, 1) < 0.01:
                    #     actions = [env.action_space.sample()]
                    # else:
                    #     actions = oracle_answer
                    actions, deco = model.translate_logits_to_button(response_logits, use_majority_vote=False)
                    print(f"Action: {actions[0]} / Oracle Action: {oracle_answer}")
                    try:
                        obs, rewards, dones, info = env.step(actions)
                    except:
                        obs, rewards, dones, info = env.step([env.action_space.sample()])
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
    parser.add_argument('-llm', default='facebook/opt-1.3b')
    parser.add_argument('-model_dir', default='../model_runs/')
    parser.add_argument('-path', default='../model_runs/tl.tar')
    parser.add_argument('-mode', default='train')
    parser.add_argument('-test_sent', default='neg')
    parser.add_argument('-test_fewshot', default='0')
    parser.add_argument('-bs', default=8, type=int)
    parser.add_argument('-acs', default=32, type=int)
    parser.add_argument('-caption_loss', default=1, type=int)
    parser.add_argument('-run_name', default='ft_vfe')
    parser.add_argument('-use_vfe', default='Clip')
    args = parser.parse_args()
    if args.mode == 'train':
        align(run_name=args.run_name, path=args.path, llm=args.llm, batch_size=args.bs, accum_steps=args.acs,
              caption_loss=args.caption_loss, use_vfe=args.use_vfe)
    else:
        pest_align(path=args.path, llm=args.llm, sent=args.test_sent,
                   fewshot=True if args.test_fewshot == '1' else False, record=True, use_vfe=args.use_vfe)
