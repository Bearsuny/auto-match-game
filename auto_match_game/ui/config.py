class AutoMatchGameUIConfig:
    title = 'auto-match-game'
    reward_item_size = (119, 119)
    reward_row = 8
    reward_col = 8
    widget_size = (reward_item_size[0] * reward_row, reward_item_size[1] * reward_col + 50)
    widget1_size = (reward_item_size[0] * reward_row, reward_item_size[1] * reward_col)
    widget2_size = (reward_item_size[0] * reward_row, 50)
    reward_name = ['Bronze', 'Silver', 'Gold', 'Pouch', 'Brown', 'Green', 'Red', 'Vault']
    reward_ref_image_path = '/home/bearsuny/Projects/auto-match-game/reward_grid/data/assets'
    scale_size = (int(119 * 0.85), int(119 * 0.85))
    screen_output_path = '/home/bearsuny/Projects/auto-match-game/output/screen.png'
    crop_box = (0, 0, 119 * reward_row, 119 * reward_col)
