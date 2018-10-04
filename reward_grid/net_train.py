import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from reward_grid.dataset import RewardGridDataset
from reward_grid.net import RewardGridNet
from reward_grid.config import RewardGridConfig as RGC

if __name__ == '__main__':
    reward_grid_dataset = RewardGridDataset(RGC.reward_grid_dataset_path, RGC.reward_item_size)
    reward_grid_data_loader = DataLoader(reward_grid_dataset, batch_size=RGC.batch_size, shuffle=True, drop_last=True)
    reward_grid_net = RewardGridNet()
    reward_grid_net.to(reward_grid_net.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(reward_grid_net.parameters(), lr=RGC.lr)

    for epoch in range(350):
        reward_grid_net.train()
        loss = 0
        for step, (data, label) in enumerate(reward_grid_data_loader):
            data = data.to(reward_grid_net.device)
            label = label.to(reward_grid_net.device)

            optimizer.zero_grad()
            output = reward_grid_net(data)
            step_loss = criterion(output, label)
            step_loss.backward()
            optimizer.step()

            loss += step_loss.item()
            step_result = torch.argmax(output, dim=1)
            step_accurate = [1 for result_item, label_item in zip(step_result, label) if result_item == label_item]
            print(f'epoch: {epoch:03}, step: {step:03}, loss: {loss/(reward_grid_dataset.__len__() // 8):.8f}, step_accurate: {len(step_accurate)}')
    torch.save(reward_grid_net, RGC.reward_grid_net_save_path)
