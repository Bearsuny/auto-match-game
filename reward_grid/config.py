class RewardGridConfig:
    # dataset
    color_convert = 'RGB'
    reward_name = ['Bronze', 'Silver', 'Gold', 'Pouch', 'Brown', 'Green', 'Red', 'Vault']
    reward_grid_dataset_path = '/home/bearsuny/Projects/auto-match-game/reward_grid/data/train'
    reward_item_size = (119, 119)

    # train
    batch_size = 8
    lr = 0.001
    reward_grid_net_save_path = '/home/bearsuny/Projects/auto-match-game/output/reward_net.pkl'

    # eval
    reward_grid_row = 8
    reward_grid_col = 8

    eval_reward_grid_image_path = '/home/bearsuny/Projects/auto-match-game/reward_grid/data/origin/20181001134708_1.jpg'
    roi_crop_box = (483, 80, 483 + 119 * 8, 80 + 119 * 8)
    reward_ref_image_path = '/home/bearsuny/Projects/auto-match-game/reward_grid/data/assets'
    contrast_reward_ref_image_path = '/home/bearsuny/Projects/auto-match-game/output/roi_contrast.png'
