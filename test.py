import os
from argparse import ArgumentParser
from datetime import datetime
import csv

import numpy as np

from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler

from dataset import iclevrDataset
from ddpm import conditionalDDPM
from evaluator import evaluation_model
from torchvision.utils import make_grid, save_image

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_scheduler(timesteps):
    return DDPMScheduler(num_train_timesteps=timesteps, beta_schedule='squaredcos_cap_v2')

def load_model(ckpt):
    model = conditionalDDPM().to(device)
    checkpoint = torch.load(ckpt)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    return model

def inference(dataloader, noise_scheduler, timesteps, model, eval_model, save_prefix='test', save_dir='result', batch_size=1):
    os.makedirs(save_dir, exist_ok=True)
    all_results = []
    acc = []
    # 創建 CSV 檔案來保存每張圖片的準確率
    csv_path = os.path.join(save_dir, f'{save_prefix}_image_accuracies.csv')
    os.makedirs(os.path.join(save_dir, save_prefix), exist_ok=True)
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'accuracy'])  # 寫入標題列
        
        progress_bar = tqdm(dataloader)
        for idx, y in enumerate(progress_bar):
            y = y.to(device)
            x = torch.randn(batch_size, 3, 64, 64).to(device)
            denoising_results = []  # 用於存儲每個 batch 中所有圖片的去噪過程
            for i, t in enumerate(noise_scheduler.timesteps):
                with torch.no_grad():
                    residual = model(x, t, y)

                x = noise_scheduler.step(residual, t, x).prev_sample
                if i % (timesteps // 10) == 0:
                    denoising_results.append(x.clone())  # 保存整個 batch

            # 計算當前 batch 中每張圖片的準確率
            current_acc = eval_model.eval(x, y)
            for b in range(batch_size):
                acc.append(current_acc)
                # 寫入當前圖片的準確率
                writer.writerow([idx * batch_size + b, f'{current_acc:.4f}'])

            progress_bar.set_postfix_str(f'batch: {idx}, accuracy: {current_acc:.4f}')

            # 處理每個 batch 中的圖片
            denoising_results.append(x)  # 添加最終結果
            denoising_results = torch.stack(denoising_results)  # [timesteps, batch, channel, height, width]
            
            # 對每張圖片分別創建去噪過程的網格
            for b in range(batch_size):
                # 提取當前圖片的所有時間步驟
                current_image_steps = denoising_results[:, b]  # [timesteps, channel, height, width]
                row_image = make_grid((current_image_steps + 1) / 2, nrow=current_image_steps.shape[0], pad_value=0)
                save_image(row_image, f'{save_dir}/{save_prefix}/{idx * batch_size + b}.png')
            
            all_results.append(x)

    # 將所有結果合併
    all_results = torch.cat(all_results, dim=0)
    all_results = make_grid(all_results, nrow=8)
    save_image((all_results + 1) / 2, f'{save_dir}/{save_prefix}_result.png')
    
    # 保存平均準確率
    with open(os.path.join(save_dir, f'{save_prefix}_summary.txt'), 'w') as f:
        f.write(f'Average accuracy: {np.mean(acc):.4f}\n')
        f.write(f'Standard deviation: {np.std(acc):.4f}\n')
        f.write(f'Min accuracy: {np.min(acc):.4f}\n')
        f.write(f'Max accuracy: {np.max(acc):.4f}\n')
    
    return acc

def parse_args():
    parser = ArgumentParser(description='DDPM 測試腳本')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/checkpoint_300.pth',
                      help='模型檢查點路徑')
    parser.add_argument('--timesteps', type=int, default=1000,
                      help='去噪步驟數')
    parser.add_argument('--save-dir', type=str, default=None,
                      help='結果保存目錄，如果未指定則在 checkpoint 目錄下創建 test_x 資料夾')
    parser.add_argument('--batch-size', type=int, default=1,
                      help='批次大小')
    parser.add_argument('--num-workers', type=int, default=4,
                      help='數據加載的工作進程數')
    parser.add_argument('--dataset-path', type=str, default='iclevr',
                      help='數據集路徑')
    parser.add_argument('--skip-new-test', action='store_true',
                      help='是否跳過新測試集')
    args = parser.parse_args()
    
    # 如果未指定 save_dir，則根據 checkpoint 創建對應的 test 目錄
    if args.save_dir is None:
        # 獲取 checkpoint 的目錄和檔案名
        ckpt_dir = os.path.dirname(args.checkpoint)
        ckpt_name = os.path.basename(args.checkpoint)
        
        # 從檔案名中提取 checkpoint 編號
        # 假設格式為 checkpoint_XXX.pth
        try:
            ckpt_num = ckpt_name.split('_')[1].split('.')[0]
            args.save_dir = os.path.join(ckpt_dir, f'test_{ckpt_num}')
        except:
            print(f'警告：無法從 checkpoint 名稱 {ckpt_name} 中提取編號，使用預設目錄 test')
            args.save_dir = os.path.join(ckpt_dir, 'test')
    
    return args

def save_results_to_csv(save_dir, checkpoint_num, test_acc, new_test_acc=None):
    """將測試結果保存到 CSV 檔案中"""
    csv_path = os.path.join(os.path.dirname(save_dir), 'test_results.csv')
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 準備要寫入的資料
    row = {
        'timestamp': current_time,
        'checkpoint': checkpoint_num,
        'test_accuracy': f'{np.mean(test_acc):.4f}',
        'new_test_accuracy': f'{np.mean(new_test_acc):.4f}' if new_test_acc is not None else 'N/A'
    }
    
    # 檢查檔案是否存在
    file_exists = os.path.isfile(csv_path)
    
    # 寫入 CSV
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        
        # 如果檔案不存在，寫入標題列
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row)
    
    print(f'測試結果已保存到：{csv_path}')

if __name__ == "__main__":
    args = parse_args()
    
    model = load_model(args.checkpoint)
    noise_scheduler = get_scheduler(args.timesteps)
    eval_model = evaluation_model()
    
    # 從 checkpoint 路徑中提取編號
    ckpt_name = os.path.basename(args.checkpoint)
    try:
        checkpoint_num = ckpt_name.split('_')[1].split('.')[0]
    except:
        checkpoint_num = 'unknown'
    
    test_loader = DataLoader(
        iclevrDataset(args.dataset_path, "test"),
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    test_acc = inference(test_loader, noise_scheduler, args.timesteps, model, eval_model, 'test', args.save_dir, args.batch_size)
    print(f'test accuracy: {np.mean(test_acc)}')
    
    new_test_acc = None
    if not args.skip_new_test:
        new_test_loader = DataLoader(
            iclevrDataset(args.dataset_path, "new_test"),
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
        new_test_acc = inference(new_test_loader, noise_scheduler, args.timesteps, model, eval_model, 'new_test', args.save_dir, args.batch_size)
        print(f'new test accuracy: {np.mean(new_test_acc)}')
    
    # 保存測試結果到 CSV
    save_results_to_csv(args.save_dir, checkpoint_num, test_acc, new_test_acc)