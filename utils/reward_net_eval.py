import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def get_reward_roi(image_path, crop_box):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = image.crop(crop_box)
    reward_grid = []
    row, col = 8, 8
    for i in range(row):
        for j in range(col):
            image_item = image.crop((119 * i, 119 * j, 119 * (i + 1), 119 * (j + 1)))
            # image_item.save(f'../data/train/{i}{j}.png')
            image_item_data = np.array(image_item, dtype=np.float32).transpose((2, 0, 1))
            reward_grid.append(image_item_data)
    reward_grid = np.array(reward_grid, dtype=np.float32)
    return reward_grid


if __name__ == '__main__':
    box = (483, 80, 483 + 119 * 8, 80 + 119 * 8)
    grid = get_reward_roi('../data/origin/20181001134334_1.jpg', box)
    print(grid.shape)

    reward_net = torch.load('../output/reward_net.pkl')
    reward_net.to(reward_net.device)
    reward_net.eval()

    grid_result = []
    with torch.no_grad():
        for i in range(8):
            data = grid[i * 8:(i + 1) * 8]
            data = torch.from_numpy(data)
            data = data.to(reward_net.device)
            output = reward_net(data)
            result = torch.argmax(output, dim=1)
            result = result.cpu().numpy()
            grid_result.append(result)
    grid_result = np.array(grid_result, dtype=np.int).transpose()
    print(grid_result)
