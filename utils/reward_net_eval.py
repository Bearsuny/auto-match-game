import os
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
            image_item_data = np.array(image_item, dtype=np.float32).transpose((2, 0, 1))
            reward_grid.append(image_item_data)
    reward_grid = np.array(reward_grid, dtype=np.float32)
    return reward_grid


def check_roi(image_path, crop_box, grid_result):
    image = Image.open(image_path)
    image = image.convert('RGB')
    image = image.crop(crop_box)
    image_origin = np.array(image)
    reward_name = ['Bronze', 'Silver', 'Gold', 'Pouch', 'Brown', 'Green', 'Red', 'Vault']
    reward_ref_image = os.listdir('../data/assets')
    reward_image = []
    for item in reward_name:
        for ref_item in reward_ref_image:
            if item in ref_item:
                image = Image.open(os.path.join('../data/assets', ref_item))
                image = image.resize((119, 119))
                image = image.convert('RGB')
                image = np.array(image).transpose((2, 0, 1))
                reward_image.append(image)
    reward_image = np.array(reward_image)
    image_check = []
    for i in range(8):
        for j in range(8):
            image_check.append(reward_image[grid_result[j][i]])
    image_check = np.array(image_check).transpose((0, 2, 3, 1))

    image_check_part = [np.concatenate(image_check[i * 8:(i + 1) * 8]) for i in range(8)]
    image_check_part = np.concatenate(image_check_part, axis=1)
    image_check_part = image_check_part

    image_check = image_check_part
    image_contrast = Image.fromarray(np.concatenate([image_origin, image_check], axis=1), mode='RGB')
    image_contrast.save('../output/grid_contrast.png')

    image_contrast_minus = Image.fromarray(image_origin - image_check, mode='RGB')
    image_contrast_minus.save('../output/grid_contrast_minus.png')


if __name__ == '__main__':
    check_image_path = '../data/origin/20181001134708_1.jpg'
    box = (483, 80, 483 + 119 * 8, 80 + 119 * 8)
    grid = get_reward_roi(check_image_path, box)
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

    check_roi(check_image_path, box, grid_result)
