import os
import torch
import numpy as np
from PIL import Image
from reward_grid.config import RewardGridConfig as RGC


def get_reward_grid_roi(image_path, crop_box):
    image = Image.open(image_path)
    image = image.convert(RGC.color_convert)
    image = image.crop(crop_box)
    roi = []
    for item in range(RGC.reward_grid_row):
        for col_item in range(RGC.reward_grid_col):
            image_item = image.crop((RGC.reward_item_size[0] * item, RGC.reward_item_size[1] * col_item,
                                     RGC.reward_item_size[0] * (item + 1), RGC.reward_item_size[1] * (col_item + 1)))
            image_item = np.array(image_item).transpose((2, 0, 1))
            roi.append(image_item)
    roi = np.array(roi)
    return roi


def reward_ref_image_init(reward_ref_image_path):
    ref_image = os.listdir(reward_ref_image_path)
    ref_image_data = []
    for item in RGC.reward_name:
        for ref_item in ref_image:
            if item in ref_item:
                image = Image.open(os.path.join(reward_ref_image_path, ref_item))
                image = image.resize(RGC.reward_item_size)
                image = image.convert(RGC.color_convert)
                image = np.array(image).transpose((2, 0, 1))
                ref_image_data.append(image)
    ref_image_data = np.array(ref_image_data)
    return ref_image_data


def reshape_reward_grid_roi(roi):
    reshape = [np.concatenate(roi[i * 8: (i + 1) * 8]) for i in range(RGC.reward_grid_row)]
    return np.concatenate(reshape, axis=1)


def generate_contrast_reward_grid_roi(origin_roi, eval_roi, ref_image):
    eval_reward_grid_roi = []
    for row in range(RGC.reward_grid_row):
        for col in range(RGC.reward_grid_col):
            eval_reward_grid_roi.append(ref_image[eval_roi[col][row]])
    eval_reward_grid_roi = np.array(eval_reward_grid_roi).transpose((0, 2, 3, 1))
    eval_reward_grid_roi = reshape_reward_grid_roi(eval_reward_grid_roi)

    origin_roi = origin_roi.transpose((0, 2, 3, 1))
    origin_roi = reshape_reward_grid_roi(origin_roi)

    contrast_reward_grid_roi = Image.fromarray(np.concatenate([origin_roi, eval_reward_grid_roi], axis=1), mode=RGC.color_convert)
    contrast_reward_grid_roi.save(RGC.contrast_reward_ref_image_path)


if __name__ == '__main__':
    reward_grid_roi = get_reward_grid_roi(RGC.eval_reward_grid_image_path, RGC.roi_crop_box)
    reward_ref_image = reward_ref_image_init(RGC.reward_ref_image_path)

    reward_grid_net = torch.load(RGC.reward_grid_net_save_path)
    reward_grid_net.to(reward_grid_net.device)
    reward_grid_net.eval()

    reward_grid_eval_result = []
    with torch.no_grad():
        for row_item in range(RGC.reward_grid_row):
            data = reward_grid_roi[row_item * RGC.reward_grid_col:(row_item + 1) * RGC.reward_grid_col]
            data = torch.from_numpy(data).float()
            data = data.to(reward_grid_net.device)
            output = reward_grid_net(data)
            result = torch.argmax(output, dim=1)
            result = result.cpu().numpy()
            reward_grid_eval_result.append(result)
    reward_grid_eval_result = np.array(reward_grid_eval_result, dtype=np.int).transpose()
    print(reward_grid_eval_result)

    generate_contrast_reward_grid_roi(reward_grid_roi, reward_grid_eval_result, reward_ref_image)
