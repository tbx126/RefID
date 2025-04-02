import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.dehaze_dataset import DehazeDataset
from models.lte_extractor import LTEExtractor
from models.attention_module import AttentionEmbedding
from models.hard_attention import HardAttention
from models.soft_attention import SoftAttention
from models.csfi import CSFI
from utils.metrics import get_loss_functions, get_metrics

# 添加解码器将特征转换回RGB图像
class DehazeDecoder(nn.Module):
    """解码器：将特征图转换为RGB图像"""
    def __init__(self, in_channels=256):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        return self.decoder(x) * 0.5 + 0.5  # 将[-1,1]范围映射到[0,1]

def train(config):
    # 创建模型保存目录
    os.makedirs(config['save_dir'], exist_ok=True)
    
    # 初始化TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(config['save_dir'], 'logs'))
    
    # 加载数据集
    train_dataset = DehazeDataset(
        dehazed1_dir=config['data_paths']['dehazed1'],
        dehazed2_dir=config['data_paths']['dehazed2'],
        clear_dir=config['data_paths']['clear'],
        target_size=config['target_size']
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers']
    )
    
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lte_model = LTEExtractor().to(device)
    attention_model = AttentionEmbedding(scale_factor=1.0 / (256 ** 0.5)).to(device)
    hard_attention = HardAttention().to(device)
    soft_attention = SoftAttention(chunk_size=config['chunk_size']).to(device)
    csfi = CSFI(in_channels=256).to(device)
    decoder = DehazeDecoder().to(device)
    
    # 创建模型参数列表用于优化
    model_params = list(lte_model.parameters()) + \
                   list(attention_model.parameters()) + \
                   list(hard_attention.parameters()) + \
                   list(soft_attention.parameters()) + \
                   list(csfi.parameters()) + \
                   list(decoder.parameters())
    
    # 初始化优化器
    optimizer = optim.Adam(model_params, lr=config['learning_rate'])
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config['lr_step_size'], 
        gamma=config['lr_gamma']
    )
    
    # 初始化损失函数和评估指标
    loss_functions = get_loss_functions(device)
    metrics = get_metrics()
    
    # 训练循环
    best_psnr = 0
    for epoch in range(config['num_epochs']):
        # 训练模式
        lte_model.train()
        attention_model.train()
        hard_attention.train()
        soft_attention.train()
        csfi.train()
        decoder.train()
        
        # 进度条
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        epoch_loss = 0
        
        for batch in pbar:
            # 将数据移动到GPU
            dehazed1 = batch['dehazed1'].to(device)
            dehazed2 = batch['dehazed2'].to(device)
            clear = batch['clear'].to(device)
            
            # 清除梯度
            optimizer.zero_grad()
            
            # 前向传播
            Q, K, V = lte_model(dehazed1)
            S, H = attention_model(Q, K, V)
            P = hard_attention(V, H)
            soft_output = soft_attention(P, S)
            features = csfi(soft_output, dehazed1)
            output = decoder(features)
            
            # 计算损失 - 仅使用SSIM损失（负值）
            loss = loss_functions['ssim'](output, clear)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 更新进度条
            epoch_loss += loss.item()
            pbar.set_postfix({"Loss": f"{epoch_loss/(pbar.n+1):.4f}"})
            
            # 记录第一个批次的图像
            if pbar.n == 0 and epoch % config['log_interval'] == 0:
                writer.add_images('Input/Hazy', dehazed1, epoch)
                writer.add_images('Ground Truth', clear, epoch)
                writer.add_images('Output/Dehazed', output, epoch)
        
        # 更新学习率
        scheduler.step()
        
        # 保存当前训练损失
        avg_loss = epoch_loss / len(train_loader)
        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('LR', scheduler.get_last_lr()[0], epoch)
        
        # 评估和保存模型
        if (epoch + 1) % config['eval_interval'] == 0:
            # 评估模式
            lte_model.eval()
            attention_model.eval()
            hard_attention.eval()
            soft_attention.eval()
            csfi.eval()
            decoder.eval()
            
            # 计算验证集PSNR和SSIM
            val_metrics = {name: 0.0 for name in metrics.keys()}
            with torch.no_grad():
                for i, batch in enumerate(train_loader):
                    if i >= config['eval_batches']:
                        break
                    
                    dehazed1 = batch['dehazed1'].to(device)
                    clear = batch['clear'].to(device)
                    
                    # 前向传播
                    Q, K, V = lte_model(dehazed1)
                    S, H = attention_model(Q, K, V)
                    P = hard_attention(V, H)
                    soft_output = soft_attention(P, S)
                    features = csfi(soft_output, dehazed1)
                    output = decoder(features)
                    
                    # 计算指标
                    for name, metric_fn in metrics.items():
                        val_metrics[name] += metric_fn(output, clear).item()
                
                # 计算平均指标
                for name in val_metrics.keys():
                    val_metrics[name] /= config['eval_batches']
            
            # 记录指标
            for name, value in val_metrics.items():
                writer.add_scalar(f'Metrics/{name}', value, epoch)
                print(f"Epoch {epoch+1} Validation {name.upper()}: {value:.4f}")
            
            # 保存最佳模型 (根据PSNR)
            if val_metrics['psnr'] > best_psnr:
                best_psnr = val_metrics['psnr']
                # 保存所有模型
                model_dict = {
                    'lte': lte_model.state_dict(),
                    'attention': attention_model.state_dict(),
                    'hard_attention': hard_attention.state_dict(),
                    'soft_attention': soft_attention.state_dict(),
                    'csfi': csfi.state_dict(),
                    'decoder': decoder.state_dict(),
                    'epoch': epoch,
                    'metrics': val_metrics
                }
                torch.save(model_dict, os.path.join(config['save_dir'], 'best_model.pth'))
        
        # 定期保存检查点
        if (epoch + 1) % config['save_interval'] == 0:
            model_dict = {
                'lte': lte_model.state_dict(),
                'attention': attention_model.state_dict(),
                'hard_attention': hard_attention.state_dict(),
                'soft_attention': soft_attention.state_dict(),
                'csfi': csfi.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch
            }
            torch.save(model_dict, os.path.join(config['save_dir'], f'checkpoint_epoch_{epoch+1}.pth'))
    
    writer.close()
    print(f"Training completed. Best PSNR: {best_psnr:.2f} dB")

if __name__ == "__main__":
    # 加载配置
    with open('./configs/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    train(config)