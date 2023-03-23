from torch.utils.data import Dataset
from einops import rearrange
import torch
import numpy as np
import re
import os
import cv2


class TuneAVideoDataset(Dataset):
    def __init__(
            self,
            video_path: str,
            prompt: str,
            width: int = 512,
            height: int = 512,
            n_sample_frames: int = 8,
            sample_start_idx: int = 0,
            sample_frame_rate: int = 1,
    ):
        self.video_path = video_path
        self.prompt = prompt
        self.prompt_ids = None

        self.width = width
        self.height = height
        self.n_sample_frames = n_sample_frames
        self.sample_start_idx = sample_start_idx
        self.sample_frame_rate = sample_frame_rate

    def __len__(self):
        return 1

    def __getitem__(self, index):
        os.makedirs("./temp/input_image", exist_ok=True)
        file_list = sorted(os.listdir(self.video_path), key=lambda s: sum(((s, int(n)) for s, n in re.findall(r'(\D+)(\d+)', 'a%s0' % s)), ()))
        sample_frames = []
        sample_index = list(range(self.sample_start_idx, len(file_list), self.sample_frame_rate))[:self.n_sample_frames]
        for i in sample_index:
            if i >= len(file_list):
                raise ValueError(f"Unexpected sample index, got {i}, expected less than {len(file_list)}")
            img = cv2.imread(os.path.join(self.video_path, file_list[i]), cv2.IMREAD_UNCHANGED)
            cv2.imwrite(f"./temp/input_image/original_{sample_index.index(i)}.png", img)
            h_input, w_input = img.shape[0:2]
            if h_input != self.height or w_input != self.width:
                img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_LANCZOS4)
            img = img.astype(np.float32)
            if np.max(img) > 256:  # 16-bit image
                max_range = 65535
            else:
                max_range = 255
            img = img / (max_range / 2) - 1.0
            img_array = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_array = rearrange(img_array, "h w c -> c h w")
            sample_frames.append(torch.from_numpy(img_array))

        example = {
            "pixel_values": torch.stack(sample_frames),
            "prompt_ids": self.prompt_ids
        }

        return example
