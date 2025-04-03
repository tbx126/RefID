import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from pytorch_msssim import SSIM

from data.dehaze_dataset import DehazeDataset
from models.LTE import LTE
from models.search_transfer import SearchTransfer
from models.csfi import CSFI
from utils import calc_psnr_and_ssim, MeanShift


class Trainer:
    def __init__(self, config):
        """初始化训练器"""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 创建模型保存目录
        os.makedirs(config['save_dir'], exist_ok=True)
        
        # 初始化TensorBoard
        self.writer = SummaryWriter(log_dir=os.path.join(config['save_dir'], 'logs'))
        
        # 加载数据集
        self._setup_dataset()
        
        # 初始化模型
        self._setup_models()
        
        # 初始化优化器
        self._setup_optimizer()
        
        # 初始化损失函数
        self.loss_functions = self._get_loss_functions()
        
        # 初始化性能跟踪
        self.best_psnr = 0
        self.best_epoch = 0
    
    def _setup_dataset(self):
        """设置数据集和数据加载器"""
        train_dataset = DehazeDataset(
            dehazed1_dir=self.config['data_paths']['dehazed1'],
            dehazed2_dir=self.config['data_paths']['dehazed2'],
            clear_dir=self.config['data_paths']['clear'],
            target_size=self.config['target_size']
        )
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=self.config['num_workers']
        )
    
    def _setup_models(self):
        """初始化所有模型组件"""
        self.lte_model = LTE(requires_grad=True).to(self.device)
        self.search_transfer = SearchTransfer().to(self.device)
        # 使用更新后的CSFI，包含解码器功能
        self.csfi = CSFI(
            in_channels=self.config.get('csfi_params', {}).get('in_channels', 256), 
            n_feats=self.config.get('csfi_params', {}).get('n_feats', 64), 
            res_depth=self.config.get('csfi_params', {}).get('res_depth', [3, 3, 3, 3]),
            res_scale=self.config.get('csfi_params', {}).get('res_scale', 0.1)
        ).to(self.device)
    
    def _setup_optimizer(self):
        """初始化优化器，为不同模块设置不同学习率"""
        # 可以为LTE模型和其他模块设置不同学习率
        lte_params = list(self.lte_model.parameters())
        other_params = list(self.search_transfer.parameters()) + \
                      list(self.csfi.parameters())
                      # 移除decoder参数
        
        self.params = [
            {"params": other_params, "lr": self.config['learning_rate']},
            {"params": lte_params, "lr": self.config['learning_rate'] * 0.1}  # LTE使用较小学习率
        ]
        
        self.optimizer = optim.Adam(self.params, betas=(0.9, 0.999), eps=1e-8)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=self.config['lr_step_size'], 
            gamma=self.config['lr_gamma']
        )
    
    def _get_loss_functions(self):
        """返回训练损失函数"""
        return {
            'rec_loss': nn.L1Loss().to(self.device),
            'ssim_loss': SSIM(data_range=1.0, size_average=True, channel=3).to(self.device)
        }
    
    def prepare(self, batch):
        """将批次数据移动到设备上"""
        dehazed1 = batch['dehazed1'].to(self.device)
        dehazed2 = batch['dehazed2'].to(self.device)
        clear = batch['clear'].to(self.device)
        return dehazed1, dehazed2, clear
    
    def _forward(self, dehazed1, dehazed2, clear):
        """更新后的前向传播函数，兼容完整的CSFI"""
        # 特征提取
        dehazed1_lv1, dehazed1_lv2, dehazed1_lv3 = self.lte_model((dehazed1 + 1.) / 2)
        dehazed2_lv1, dehazed2_lv2, dehazed2_lv3 = self.lte_model((dehazed2 + 1.) / 2)
        clear_lv1, clear_lv2, clear_lv3 = self.lte_model((clear + 1.) / 2)
        
        # 特征搜索和转移
        S, T_lv3, T_lv2, T_lv1 = self.search_transfer(
            dehazed1_lv3, dehazed2_lv3, 
            clear_lv1, clear_lv2, clear_lv3
        )
        
        # 使用集成的CSFI模块
        output = self.csfi(dehazed1, S, T_lv3, T_lv2, T_lv1)
        
        return output, S, T_lv3, T_lv2, T_lv1
    
    def _save_model(self, epoch, metrics=None, is_best=False):
        """保存模型检查点"""
        model_dict = {
            'lte': self.lte_model.state_dict(),
            'search_transfer': self.search_transfer.state_dict(),
            'csfi': self.csfi.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'epoch': epoch
        }
        
        if metrics:
            model_dict['metrics'] = metrics
        
        if is_best:
            save_path = os.path.join(self.config['save_dir'], 'best_model.pth')
        else:
            save_path = os.path.join(self.config['save_dir'], f'checkpoint_epoch_{epoch+1}.pth')
        
        torch.save(model_dict, save_path)
        print(f"Model saved to {save_path}")
    
    def load(self, model_path):
        """加载预训练模型"""
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.lte_model.load_state_dict(checkpoint['lte'])
        self.search_transfer.load_state_dict(checkpoint['search_transfer'])
        self.csfi.load_state_dict(checkpoint['csfi'])
        # 移除decoder的加载
        
        if 'optimizer' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        
        start_epoch = checkpoint.get('epoch', -1) + 1
        return start_epoch
    
    def train(self, start_epoch=0):
        """训练模型"""
        print(f"Starting training from epoch {start_epoch}")
        print(f"Learning rates: Main={self.optimizer.param_groups[0]['lr']}, LTE={self.optimizer.param_groups[1]['lr']}")
        
        for epoch in range(start_epoch, self.config['num_epochs']):
            # 设置训练模式
            self.lte_model.train()
            self.search_transfer.train()
            self.csfi.train()
            
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")
            epoch_loss = 0
            
            for batch_idx, batch in enumerate(pbar):
                # 准备数据
                dehazed1, dehazed2, clear = self.prepare(batch)
                
                # 清除梯度
                self.optimizer.zero_grad()
                
                # 前向传播
                output, S, T_lv3, T_lv2, T_lv1 = self._forward(dehazed1, dehazed2, clear)
                
                # 计算损失
                rec_loss = self.loss_functions['rec_loss'](output, clear) * self.config['lambda_l1']
                ssim_loss = (1 - self.loss_functions['ssim_loss'](output, clear)) * self.config['lambda_ssim']
                loss = rec_loss + ssim_loss
                
                # 反向传播和优化
                loss.backward()
                self.optimizer.step()
                
                # 更新进度条
                epoch_loss += loss.item()
                pbar.set_postfix({"Loss": f"{epoch_loss/(batch_idx+1):.4f}"})
                
                # 记录第一个批次的图像
                if batch_idx == 0 and epoch % self.config['log_interval'] == 0:
                    self.writer.add_images('Input/Hazy', dehazed1, epoch)
                    self.writer.add_images('Ground Truth', clear, epoch)
                    self.writer.add_images('Output/Dehazed', output, epoch)
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()
            print(f"Epoch {epoch+1} learning rate: {current_lr}")
            
            # 记录损失
            avg_loss = epoch_loss / len(self.train_loader)
            self.writer.add_scalar('Loss/train', avg_loss, epoch)
            self.writer.add_scalar('LR/main', current_lr[0], epoch)
            self.writer.add_scalar('LR/lte', current_lr[1], epoch)
            
            # 评估模型
            if (epoch + 1) % self.config['eval_interval'] == 0:
                psnr = self.evaluate(epoch)
                if psnr > self.best_psnr:
                    self.best_psnr = psnr
                    self.best_epoch = epoch
                    self._save_model(epoch, {'psnr': psnr}, is_best=True)
                print(f"Best PSNR: {self.best_psnr:.2f} at epoch {self.best_epoch+1}")
            
            # 定期保存检查点
            if (epoch + 1) % self.config['save_interval'] == 0:
                self._save_model(epoch)
        
        self.writer.close()
        print(f"Training completed. Best PSNR: {self.best_psnr:.2f} at epoch {self.best_epoch+1}")
    
    def evaluate(self, epoch):
        """评估模型性能"""
        # 设置评估模式
        self.lte_model.eval()
        self.search_transfer.eval()
        self.csfi.eval()
        
        total_psnr = 0.0
        eval_batches = min(self.config['eval_batches'], len(self.train_loader))
        
        with torch.no_grad():
            for i, batch in enumerate(self.train_loader):
                if i >= eval_batches:
                    break
                
                dehazed1, dehazed2, clear = self.prepare(batch)
                output, _, _, _, _ = self._forward(dehazed1, dehazed2, clear)
                
                # 计算PSNR
                psnr = calc_psnr_and_ssim(output, clear)
                total_psnr += psnr
            
            # 计算平均PSNR
            avg_psnr = total_psnr / eval_batches
        
        # 记录指标
        self.writer.add_scalar('Metrics/PSNR', avg_psnr, epoch)
        print(f"Epoch {epoch+1} Validation PSNR: {avg_psnr:.4f}")
        
        return avg_psnr


if __name__ == "__main__":
    # 加载配置
    with open('./configs/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    trainer = Trainer(config)
    
    # 如果指定了预训练模型，加载它
    if 'pretrained_model' in config and config['pretrained_model']:
        start_epoch = trainer.load(config['pretrained_model'])
    else:
        start_epoch = 0
    
    # 开始训练
    trainer.train(start_epoch)