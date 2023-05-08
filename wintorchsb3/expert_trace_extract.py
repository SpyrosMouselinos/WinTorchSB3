import copy
import random
import matplotlib as mpl
from stable_baselines3 import PPO, DQN
from huggingface_sb3 import load_from_hub
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
import os
import gym
from x_atari import AtariWrapper, FrameStack

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
mpl.use('TkAgg')  # !IMPORTANT


class ExtractedModelPolicy(nn.Module):
    def __init__(self, fe, an):
        super().__init__()
        self.fe = fe if fe is not None else nn.Identity()
        self.an = an if an is not None else nn.Identity()

    def forward(self, x):
        feats = self.fe(x)
        actions = self.an(feats)
        return feats, actions

    def predict(self, x):
        feats = self.fe(x)
        actions = self.an(feats)
        return feats, actions


def dqn_load_pong():
    checkpoint = load_from_hub("sb3/dqn-PongNoFrameskip-v4", "dqn-PongNoFrameskip-v4.zip")

    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
        "exploration_schedule": lambda _: 0.0,
        "optimize_memory_usage": False
    }

    model = DQN.load(checkpoint, custom_objects=custom_objects)
    model.policy.set_training_mode(False)
    extracted_preprocessing = lambda x: model.policy.obs_to_tensor(x.transpose((0, 3, 1, 2)).float())

    extracted_model = ExtractedModelPolicy(fe=model.policy.q_net, an=None)

    env = make_atari_env('PongNoFrameskip-v4', n_envs=1)
    env = VecFrameStack(env, n_stack=4)
    return env, extracted_model, extracted_preprocessing


def ppo_load_pong(n_envs=1, dual=False):
    checkpoint = load_from_hub("sb3/ppo-PongNoFrameskip-v4", "ppo-PongNoFrameskip-v4.zip")

    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }

    model = PPO.load(checkpoint, custom_objects=custom_objects, exact_match=False)
    if model.policy.normalize_images:
        normalizer = 255.0
    else:
        normalizer = 1.0
    model.policy.set_training_mode(False)
    extracted_shape_fix = lambda x: np.array(x)
    extracted_preprocessing = lambda x: \
        model.policy.obs_to_tensor(extracted_shape_fix(x).transpose((3, 0, 1, 2)).squeeze())[0].float() / normalizer
    extracted_model = ExtractedModelPolicy(fe=model.policy.features_extractor, an=model.policy.action_net)

    env = gym.make('PongNoFrameskip-v4')
    env.seed(1995)
    env_gray = AtariWrapper(env, grayscale=True)
    env_gray = FrameStack(env_gray, 4)
    if dual:
        env2 = copy.deepcopy(env)
        env_color = AtariWrapper(env2, grayscale=False)
        env_color = FrameStack(env_color, 4)

    buttons = env.action_space.n
    if dual:
        return (env_color, env_gray), 'Pong', extracted_model, extracted_preprocessing, buttons
    return env_gray, 'Pong', extracted_model, extracted_preprocessing, buttons


def ppo_load_breakout(n_envs=1, dual=False):
    checkpoint = load_from_hub("sb3/ppo-BreakoutNoFrameskip-v4", "ppo-BreakoutNoFrameskip-v4.zip")

    custom_objects = {
        "learning_rate": 0.0,
        "lr_schedule": lambda _: 0.0,
        "clip_range": lambda _: 0.0,
    }

    model = PPO.load(checkpoint, custom_objects=custom_objects, exact_match=False)
    if model.policy.normalize_images:
        normalizer = 255.0
    else:
        normalizer = 1.0
    model.policy.set_training_mode(False)
    extracted_shape_fix = lambda x: np.array(x)
    extracted_preprocessing = lambda x: \
        model.policy.obs_to_tensor(extracted_shape_fix(x).transpose((3, 0, 1, 2)).squeeze())[0].float() / normalizer
    extracted_model = ExtractedModelPolicy(fe=model.policy.features_extractor, an=model.policy.action_net)

    env = gym.make('BreakoutNoFrameskip-v4')
    env.seed(1995)
    env_gray = AtariWrapper(env, grayscale=True)
    env_gray = FrameStack(env_gray, 4)
    if dual:
        env2 = copy.deepcopy(env)
        env_color = AtariWrapper(env2, grayscale=False)
        env_color = FrameStack(env_color, 4)

    buttons = env.action_space.n
    if dual:
        return (env_color, env_gray), 'Breakout', extracted_model, extracted_preprocessing, buttons
    return env_gray, 'Breakout', extracted_model, extracted_preprocessing, buttons


def ppo_load(game, dual=False):
    if game == 'Pong':
        return ppo_load_pong(dual=dual)
    elif game == 'Breakout':
        return ppo_load_breakout(dual=dual)


def maybe_extract_trace(game, random_act_prob=0.01, dual=False, feature_pipeline=None):
    """
        Update so that the feature extraction happens after validating the episode
    """
    if dual:
        if feature_pipeline is None:
            raise ValueError("Dual = True, needs feature pipeline not None!\n")
    if game == 'Pong':
        expected_score = 15
        env, _, extracted_model, extracted_preprocessing, buttons = ppo_load_pong(1, dual=dual)
    elif game == 'Breakout':
        expected_score = 80
        env, _, extracted_model, extracted_preprocessing, buttons = ppo_load_pong(1, dual=dual)
    else:
        raise NotImplementedError

    if dual:
        obs, obs_g = env[0].reset(), env[1].reset()
        env[0].seed(1995)
        env[1].seed(1995)
    else:
        obs = env.reset()
    sum_rewards = 0
    n_examples = 0
    img_features = []
    actions_probs = []
    while True:
        if dual:
            image_features = obs
            observation = extracted_preprocessing(obs_g)
        else:
            observation = extracted_preprocessing(obs)  # 2939.6355
        with torch.no_grad():
            feats, actions = extracted_model.predict(observation)
            if dual:
                feats = image_features
        action = actions.cpu().numpy().reshape((1, buttons))
        action_pos = action.argmax(-1).item()
        if dual:
            img_features.append(feats)
        else:
            for f in feats:
                img_features.append(f)
        for a in action:
            actions_probs.append(a)

        ### Randomize Steps ###
        if random.uniform(0, 1) < random_act_prob:
            inplace = env if not dual else env[0]
            action_pos = random.choice(
                range(0, inplace.action_space.n)
            )

        ### Perform the Step ###
        if dual:
            obs, rewards_1, done_1, _ = env[0].step(action_pos)
            obs_g, rewards_2, done_2, _ = env[1].step(action_pos)
            assert rewards_1 == rewards_2
            assert done_1 == done_2
            rewards = rewards_1
            done = done_1
        else:
            obs, rewards, done, _ = env.step(action_pos)

        sum_rewards += rewards
        n_examples += 1
        if done or (not done and n_examples > 3_000):
            if sum_rewards >= expected_score:
                ### NOW RUN THE FEATURE EXTRACTOR ###
                if dual:
                    img_features = feature_pipeline(img_features)
                return img_features, actions_probs, buttons, n_examples
            else:
                sum_rewards = 0
                n_examples = 0
                img_features = []
                actions_probs = []
                if dual:
                    obs, obs_g = env[0].reset(), env[1].reset()
                    env[0].seed(1995)
                    env[1].seed(1995)
                else:
                    obs = env.reset()
                    env.seed(1995)


def extract_expert_trace(game='Pong', n_games=20, random_act_prob=0.01, dual=False, feature_pipeline=None):
    valid_games = 0
    img_features = []
    actions_probs = []
    while valid_games < n_games:
        img_features_, actions_probs_, buttons, examples = maybe_extract_trace(game=game,
                                                                               random_act_prob=random_act_prob,
                                                                               dual=dual,
                                                                               feature_pipeline=feature_pipeline)
        for f in img_features_:
            img_features.append(f)
        for f in actions_probs_:
            actions_probs.append(f)

        valid_games += 1
    return game, img_features, actions_probs, buttons


class MultiModalDS(Dataset):
    def __init__(self, sources, n_games, args):
        self.GAMES = ['Pong', 'Breakout']
        self.IMAGES = ['Cifar100']
        self.SOUNDS = ['Random']
        self.sources = sources
        self.n_games = n_games
        self.args = args
        self.repack_blobs(self.prepare_sources())

    def prepare_game(self, game, source_id):
        blob = extract_expert_trace(game=game, n_games=self.n_games[source_id],
                                    random_act_prob=self.args[source_id]['random_act_prob'],
                                    dual=self.args[source_id]['dual'],
                                    feature_pipeline=self.get_pipeline(self.args[source_id]['feature_pipeline']))
        return blob

    def get_pipeline(self, pipeline=None):
        SUPPORTED_PIPELINES = ['Clip-p', 'Clip', 'Vit']
        if pipeline is None:
            return None
        if pipeline not in SUPPORTED_PIPELINES:
            raise ValueError(f'Supported Pipelines are: {SUPPORTED_PIPELINES}')

        if pipeline == 'Clip':
            from transformers import AutoProcessor, CLIPVisionModel
            model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
            model.to('cuda')
            processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

            def iter_batch(inp):
                output = []
                inp_batched = torch.from_numpy(np.stack(inp).squeeze()).int()
                if len(inp_batched.size()) == 4:
                    inp_batched = inp_batched.unsqueeze(0)
                EPISODE_LENGTH = inp_batched.shape[0]
                FRAMESKIP_LENGTH = inp_batched.shape[1]
                inp_batched = inp_batched.reshape(inp_batched.shape[0] * inp_batched.shape[1], inp_batched.shape[2],
                                                  inp_batched.shape[3], inp_batched.shape[4])
                with torch.no_grad():
                    i = None
                    for i in range(0, inp_batched.size()[0] // 16):
                        inp_batched_proc = processor(images=inp_batched[i * 16:(i + 1) * 16], return_tensors="pt").to(
                            'cuda')
                        output_ = model(**inp_batched_proc).pooler_output
                        output.append(output_)
                    if i is None:
                        inp_batched_proc = processor(images=inp_batched, return_tensors="pt").to('cuda')
                    else:
                        inp_batched_proc = processor(images=inp_batched[(i+1) * 16:], return_tensors="pt").to('cuda')
                    output_ = model(**inp_batched_proc).pooler_output
                    output.append(output_)
                return torch.concat(output).resize(EPISODE_LENGTH, FRAMESKIP_LENGTH, output_.size()[1])

            return iter_batch

        elif pipeline == 'Clip-p':
            from transformers import AutoProcessor, CLIPVisionModelWithProjection
            model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
            processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
            preprocessor_pipeline = lambda x: torch.from_numpy(
                np.array(x).transpose((3, 0, 1, 2, 4)).squeeze()).int().cuda()
            processor_pipeline = lambda x: processor(images=preprocessor_pipeline(x), return_tensors="pt")
            model_pipeline = lambda x: model(**processor_pipeline(x)).image_embeds
            return model_pipeline
        elif pipeline == 'Vit':
            from transformers import AutoImageProcessor, ViTModel
            processor = AutoImageProcessor.from_pretrained('google/vit-base-patch32-384')
            model = ViTModel.from_pretrained("google/vit-base-patch32-384")
            preprocessor_pipeline = lambda x: torch.from_numpy(
                np.array(x).transpose((3, 0, 1, 2, 4)).squeeze()).int().cuda()
            processor_pipeline = lambda x: processor(images=preprocessor_pipeline(x), return_tensors="pt")
            model_pipeline = lambda x: model(**processor_pipeline(x)).pooler_output
            return model_pipeline

    def prepare_sources(self):
        source_blobs = []
        for source_id in range(len(self.sources)):
            if self.sources[source_id] in self.GAMES:
                source_blobs.append(self.prepare_game(self.sources[source_id], source_id))
        return source_blobs

    def repack_blobs(self, source_blobs):
        total_examples = 0
        blob_ids = []
        blob_expert_features = []
        blob_expert_responses = []
        max_buttons = max(blob[3] for blob in source_blobs)
        for blob in source_blobs:
            new_examples = max(len(blob[0]), len(blob[1]), len(blob[2]))
            total_examples += new_examples
            for f in [blob[0]] * new_examples:
                blob_ids.append(f)
            for f in blob[1]:
                blob_expert_features.append(f)
            for f in blob[2]:
                if len(f) < max_buttons:
                    blob_expert_responses.append(
                        np.pad(f, pad_width=(0, max_buttons - len(f)), mode='constant', constant_values=min(f)))
                else:
                    blob_expert_responses.append(f)

        self.total_examples = total_examples
        self.blob_ids = blob_ids
        self.blob_expert_features = blob_expert_features
        self.blob_expert_responses = blob_expert_responses
        self.max_buttons = max_buttons

    # def maybe_execute(self):
    #     if os.path.exists('prepacked_ng_{}_rnd_{}.pkl'):
    #         pass

    def __len__(self):
        return self.total_examples

    def __getitem__(self, idx):
        return self.blob_ids[idx], self.blob_expert_features[idx], self.blob_expert_responses[idx]


def pest_pong():
    REWARDS = 0
    env, _, extracted_model, extracted_preprocessing, buttons = ppo_load_pong(n_envs=1)
    obs = env.reset()
    while True:
        observation, _ = extracted_preprocessing(obs)
        with torch.no_grad():
            feats, actions = extracted_model.predict(observation)
        action = actions.cpu().numpy().reshape((1, buttons))
        action_pos = action.argmax(-1)
        obs, rewards, done, info = env.step(action_pos)
        REWARDS += rewards[0]
        env.render()
        if done:
            env.close()
            print(REWARDS)
            return


def pest_breakout():
    REWARDS = 0
    env, _, extracted_model, extracted_preprocessing, buttons = ppo_load_breakout(n_envs=1)
    obs = env.reset()
    while True:
        observation = extracted_preprocessing(obs)
        with torch.no_grad():
            feats, actions = extracted_model.predict(observation)
        action = actions.cpu().numpy().reshape((1, buttons))
        action_pos = action.argmax(-1)
        if random.uniform(0, 1) < 0.015:
            action_pos = [env.action_space.sample()]
        obs, rewards, done, info = env.step(action_pos)
        REWARDS += rewards[0]
        env.render()
        if done:
            env.close()
            print(REWARDS)
            return

        # from torch.utils.data import DataLoader

# train_dataloader = DataLoader(
#     MultiModalDS(sources=['Pong'], args=[{'random_act_prob': 0.05, 'dual': True, 'feature_pipeline': 'Clip'}],
#                  n_games=[1]),
#     batch_size=64, shuffle=True)

# test_dataloader = DataLoader(
#     MultiModalDS(sources=['Pong', 'Breakout'], n_examples=[20, 15], n_jobs=[1, 1], args=[{'random_act_prob': 0.00}]),
#     batch_size=5, shuffle=False)

# for batch in train_dataloader:
#     print('hey')
#     print('hoi')
