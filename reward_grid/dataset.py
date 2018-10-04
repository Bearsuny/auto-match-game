import os
from torch.utils.data import Dataset
import numpy as np

from PIL import Image
from reward_grid.config import RewardGridConfig as RGC


class RewardGridDataset(Dataset):
    def __init__(self, dataset_path, item_size):
        self.dataset_path = dataset_path
        self.item_size = item_size
        self.name_list = [item for item in os.listdir(self.dataset_path)]

    def __getitem__(self, item):
        image = Image.open(os.path.join(self.dataset_path, self.name_list[item]))
        image = image.convert(RGC.color_convert)
        image = image.resize(self.item_size)
        for i, reward_name_item in enumerate(RGC.reward_name):
            if reward_name_item in self.name_list[item]:
                return np.array(image, dtype=np.float32).transpose((2, 0, 1)), i

    def __len__(self):
        return len(self.name_list)
