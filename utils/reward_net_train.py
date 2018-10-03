import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from utils.reward_dataset import RewardDataset
from nets.reward_net import RewardNet

if __name__ == '__main__':
    reward_dataset = RewardDataset('../data/train', (119, 119))
    reward_data_loader = DataLoader(reward_dataset, batch_size=8, shuffle=True, drop_last=True)
    reward_net = RewardNet()
    reward_net.to(reward_net.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(reward_net.parameters(), lr=0.001)

    for epoch in range(500):
        reward_net.train()
        loss = 0
        for step, (data, label) in enumerate(reward_data_loader):
            data = data.to(reward_net.device)
            label = label.to(reward_net.device)

            optimizer.zero_grad()
            output = reward_net(data)
            step_loss = criterion(output, label)
            step_loss.backward()
            optimizer.step()

            loss += step_loss.item()
            step_result = torch.argmax(output, dim=1)
            step_accurate = [1 for result_item, label_item in zip(step_result, label) if result_item == label_item]
            print(f'epoch: {epoch:03}, step: {step:03}, loss: {loss/(reward_dataset.__len__() // 8):.8f}, step_accurate: {len(step_accurate)}')
    torch.save(reward_net, '../output/reward_net.pkl')
