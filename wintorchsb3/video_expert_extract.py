import os
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib as mpl
import pytorchvideo.data

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    Resize,
)
from tqdm import tqdm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
mpl.use('TkAgg')


def locate_dataset(source):
    if source == 'UCF_101':
        if os.path.exists('../UCF101_subset'):
            print('Dataset Found!')
            for split in ['train', 'val', 'test']:
                if os.path.exists(f'../UCF101_subset/{split}'):
                    print(f'Split {split} found!')
        from glob import glob
        all_video_file_paths = glob('../UCF101_subset/*/*/*', recursive=True)
        class_labels = sorted({str(path)[3:].replace('\\', '/').split("/")[2] for path in all_video_file_paths})
        label2id = {label: i for i, label in enumerate(class_labels)}
        id2label = {i: label for label, i in label2id.items()}

        print(f"Unique classes: {list(label2id.keys())}.")
        return label2id, id2label


def load_model(source, pipeline):
    label2id, id2label = locate_dataset(source)
    SUPPORTED_PIPELINES = ['VideoMae', 'VideoMaeFT']
    if pipeline is None:
        return None
    if pipeline not in SUPPORTED_PIPELINES:
        raise ValueError(f'Supported Pipelines are: {SUPPORTED_PIPELINES}')
    from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
    model = VideoMAEForVideoClassification.from_pretrained("MCG-NJU/videomae-base", label2id=label2id,
                                                           id2label=id2label,
                                                           ignore_mismatched_sizes=False)
    model.classifier = torch.nn.Identity()
    model.to('cuda')
    processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    mean = processor.image_mean
    std = processor.image_std
    if "shortest_edge" in processor.size:
        height = width = processor.size["shortest_edge"]
    else:
        height = processor.size["height"]
        width = processor.size["width"]
    resize_to = (height, width)

    num_frames_to_sample = model.config.num_frames
    sample_rate = 4
    fps = 30
    clip_duration = num_frames_to_sample * sample_rate / fps
    return model, mean, std, resize_to, clip_duration, num_frames_to_sample


def video_collate_fn(examples):
    # permute to (num_frames, num_channels, height, width)
    pixel_values = torch.stack(
        [example["video"].permute(1, 0, 2, 3) for example in examples]
    )
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


class VideoDS(Dataset):
    def __init__(self, sources, train_randomness_multiplier, split, batch_size_extract=8, data_used='subset'):
        self.VIDEOS = ['UCF_101']
        self.data_used = data_used
        self.batch_size_extract = batch_size_extract
        self.sources = sources
        self.randomness_multiplier = train_randomness_multiplier
        self._prepare_everything(source=sources[0], pipeline='VideoMae')
        self.prepare_set(split=split)
        ### CLEAN UP! ###
        del self.model

    def _prepare_everything(self, source, pipeline):
        model, mean, std, resize_to, clip_duration, num_frames_to_sample = load_model(source, pipeline)
        self.model = model
        self.mean = mean
        self.std = std
        self.resize_to = resize_to
        self.clip_duration = clip_duration
        self.num_frames_to_sample = num_frames_to_sample

    def _load_data(self, split):
        if split == 'train':
            train_transform = Compose(
                [
                    ApplyTransformToKey(
                        key="video",
                        transform=Compose(
                            [
                                UniformTemporalSubsample(self.num_frames_to_sample),
                                Lambda(lambda x: x / 255.0),
                                Normalize(self.mean, self.std),
                                RandomShortSideScale(min_size=256, max_size=320),
                                RandomCrop(self.resize_to),
                                RandomHorizontalFlip(p=0.5),
                            ]
                        ),
                    ),
                ]
            )
            d = pytorchvideo.data.Ucf101(
                data_path=os.path.join(f'../UCF101_{self.data_used}', "train"),
                clip_sampler=pytorchvideo.data.make_clip_sampler("random", self.clip_duration),
                decode_audio=False,
                transform=train_transform,
            )
        else:
            val_transform = Compose(
                [
                    ApplyTransformToKey(
                        key="video",
                        transform=Compose(
                            [
                                UniformTemporalSubsample(self.num_frames_to_sample),
                                Lambda(lambda x: x / 255.0),
                                Normalize(self.mean, self.std),
                                Resize(self.resize_to),
                            ]
                        ),
                    ),
                ]
            )
            if split == 'val':
                d = pytorchvideo.data.Ucf101(
                    data_path=os.path.join(f'../UCF101_{self.data_used}', "val"),
                    clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self.clip_duration),
                    decode_audio=False,
                    transform=val_transform,
                )
            else:
                d = pytorchvideo.data.Ucf101(
                    data_path=os.path.join(f'../UCF101_{self.data_used}', "test"),
                    clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self.clip_duration),
                    decode_audio=False,
                    transform=val_transform,
                )
        return d

    def prepare_set(self, split):
        if split == 'train':
            rand_mul = self.randomness_multiplier
        elif split == 'val':
            rand_mul = 1
        elif split == 'test':
            rand_mul = 1
        else:
            raise ValueError

        if not os.path.exists(f'../UCF101_{self.data_used}_repr/'):
            os.mkdir(f'../UCF101_{self.data_used}_repr/')
            os.mkdir(f'../UCF101_{self.data_used}_repr/train')
            os.mkdir(f'../UCF101_{self.data_used}_repr/val')
            os.mkdir(f'../UCF101_{self.data_used}_repr/test')

        if not os.path.exists(f'../UCF101_{self.data_used}_repr/{split}/{split}_data.pkl'):
            data = self._load_data(split)

            self.output_representations = []
            self.output_labels = []
            dataloader = DataLoader(data, batch_size=self.batch_size_extract, shuffle=False,
                                    collate_fn=video_collate_fn)
            with torch.no_grad():
                for random_transformation_id in range(rand_mul):
                    for step_idx, batch in enumerate(tqdm(dataloader)):
                        pixel_values, labels = batch['pixel_values'], batch['labels']
                        pixel_values = pixel_values.to('cuda')
                        representations = self.model(pixel_values=pixel_values)
                        for f, l in zip(representations.logits.cpu(), labels):
                            self.output_representations.append(f)
                            self.output_labels.append(l)

            self.total_examples = len(self.output_representations)
            pickle_store = {'video_features': self.output_representations, 'labels': self.output_labels}
            with open(f'../UCF101_{self.data_used}_repr/{split}/{split}_data.pkl', 'wb') as fout:
                pickle.dump(pickle_store, fout)
        else:
            with open(f'../UCF101_{self.data_used}_repr/{split}/{split}_data.pkl', 'rb') as fin:
                data = pickle.load(fin)
                self.output_representations = data['video_features']
                self.output_labels = data['labels']
                self.total_examples = len(self.output_representations)
        return

    def __len__(self):
        return self.total_examples

    def __getitem__(self, idx):
        return self.output_representations[idx], self.output_labels[idx]

# train_dataloader = DataLoader(
#     VideoDS(sources=['UCF_101'], train_randomness_multiplier=5, split='val'),
#     batch_size=16, shuffle=False)

# test_dataloader = DataLoader(
#     MultiModalDS(sources=['Pong', 'Breakout'], n_examples=[20, 15], n_jobs=[1, 1], args=[{'random_act_prob': 0.00}]),
#     batch_size=5, shuffle=False)

# for batch in train_dataloader:
#     print('hey')
#     print('hoi')
