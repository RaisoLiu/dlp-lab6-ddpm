import os
from argparse import ArgumentParser
import glob
import yaml

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from diffusers import DDPMScheduler

from dataset import iclevrDataset
from ddpm import conditionalDDPM



def get_random_timesteps(batch_size, total_timesteps, device):
    return torch.randint(0, total_timesteps, (batch_size,)).long().to(device)

def save_checkpoint(model, optimizer, path, epoch):
    save_dir = os.path.join(path, f'checkpoint_{epoch}.pth')
    torch.save({'model': model.state_dict(), 
                'optimizer': optimizer.state_dict()}, save_dir)

def load_checkpoint(model, optimizer, path):
    model.load_state_dict(torch.load(path)['model'])
    optimizer.load_state_dict(torch.load(path)['optimizer'])

def train_one_epoch(epoch, model, optimizer, train_loader, loss_function, noise_scheduler, total_timesteps, device):
    model.train()
    train_loss = []
    progress_bar = tqdm(train_loader, desc=f'Epoch: {epoch}', leave=True)
    for i, (x, label) in enumerate(progress_bar):
        batch_size = x.shape[0]
        x, label = x.to(device), label.to(device)
        noise = torch.randn_like(x)
        
        timesteps = get_random_timesteps(batch_size, total_timesteps, device)
        noisy_x = noise_scheduler.add_noise(x, noise, timesteps)
        output = model(noisy_x, timesteps, label)
        
        loss = loss_function(output, noise)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss.append(loss.item())
        progress_bar.set_postfix({'Loss': np.mean(train_loss)})
        
    return np.mean(train_loss)

def get_next_experiment_number(base_dir):
    os.makedirs(base_dir, exist_ok=True)
    existing_dirs = glob.glob(os.path.join(base_dir, 'experiment_*'))
    if not existing_dirs:
        return 0
    numbers = [int(d.split('_')[-1]) for d in existing_dirs]
    return max(numbers) + 1

def arg_parser():
    parser = ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--num-epochs', type=int, default=300)
    parser.add_argument('--total-timesteps', type=int, default=1000)
    parser.add_argument('--beta-schedule', type=str, default='squaredcos_cap_v2')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=20)
    parser.add_argument('--dataset', type=str, default='iclevr')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--resume', action='store_true', default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--save-dir', type=str, default=None)
    parser.add_argument('--log-dir', type=str, default=None)
    parser.add_argument('--save-freq', type=int, default=5)
    args = parser.parse_args()
    
    # 如果沒有指定 checkpoint，則自動設置 resume 為 True
    if args.checkpoint is None:
        args.resume = True
    
    # 如果沒有指定 save_dir，則在 results 下創建自動編號的資料夾
    if args.save_dir is None:
        exp_num = get_next_experiment_number('results')
        args.save_dir = f'results/experiment_{exp_num}'
    
    # 如果沒有指定 log_dir，則使用與 save_dir 相同的路徑
    if args.log_dir is None:
        args.log_dir = args.save_dir
    
    return args

def save_hyperparameters(args, save_dir):
    """將超參數保存到 yaml 檔案中"""
    config = {
        'learning_rate': args.lr,
        'num_epochs': args.num_epochs,
        'total_timesteps': args.total_timesteps,
        'beta_schedule': args.beta_schedule,
        'batch_size': args.batch_size,
        'num_workers': args.num_workers,
        'dataset': args.dataset,
        'device': args.device,
        'save_freq': args.save_freq
    }
    
    # 如果有 checkpoint，也記錄下來
    if args.checkpoint is not None:
        config['checkpoint'] = args.checkpoint
    
    yaml_path = os.path.join(save_dir, 'config.yaml')
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f'超參數已保存到：{yaml_path}')

def main():
    args = arg_parser()
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # 保存超參數
    save_hyperparameters(args, args.save_dir)
    
    # 初始化 wandb
    wandb.init(
        project="ddpm-iclevr",
        config={
            "learning_rate": args.lr,
            "num_epochs": args.num_epochs,
            "total_timesteps": args.total_timesteps,
            "beta_schedule": args.beta_schedule,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "dataset": args.dataset,
            "device": args.device
        }
    )
    
    dataset = iclevrDataset(args.dataset, "train")
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = conditionalDDPM().to(args.device)
    mse_loss = nn.MSELoss()
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.total_timesteps, beta_schedule=args.beta_schedule)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    if args.resume and args.checkpoint is not None:
        load_checkpoint(model, optimizer, args.checkpoint)
        
    for epoch in range(args.num_epochs):
        loss = train_one_epoch(epoch, model, optimizer, train_loader, mse_loss, noise_scheduler, args.total_timesteps, args.device)
        # 使用 wandb 記錄損失
        wandb.log({"train_loss": loss}, step=epoch)
        if epoch % args.save_freq == 0:
            save_checkpoint(model, optimizer, args.save_dir, epoch)
    save_checkpoint(model, optimizer, args.save_dir, args.num_epochs)
    wandb.finish()

if __name__ == '__main__':
    main()