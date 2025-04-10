import os
import argparse
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
import logging
import torch.nn.functional as F
from pytorch_msssim import SSIM

# 导入自定义模块
from models.LTE import LTE
from models.search_transfer import SearchTransfer
from models.csfi import CSFI
from data.dehaze_dataset import DehazeDataset
from utils import calc_psnr_and_ssim

# 配置日志记录器
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def tensor2img(tensor):
    """将张量转换为numpy数组用于保存图像"""
    img = tensor.detach().cpu().numpy().transpose(1, 2, 0)
    img = (img + 1) / 2.0 * 255.0  # 从[-1,1]转换到[0,255]
    return np.clip(img, 0, 255).astype(np.uint8)

def remove_padding(tensor, paste_x, paste_y, scaled_w, scaled_h, original_w, original_h):
    """从带有padding的张量中提取原始图像区域并缩放回原始尺寸"""
    # 提取有效区域
    x = int(paste_x)
    y = int(paste_y)
    w = int(scaled_w)
    h = int(scaled_h)
    orig_w = int(original_w)
    orig_h = int(original_h)
    
    # 确保区域不超出张量边界
    x_end = min(x + w, tensor.shape[2])
    y_end = min(y + h, tensor.shape[1])
    
    # 提取有效区域
    valid_region = tensor[:, y:y_end, x:x_end]
    
    # 调整大小回原始尺寸
    resized = F.interpolate(valid_region.unsqueeze(0), size=(orig_h, orig_w), 
                          mode='bicubic', align_corners=False)
    
    return resized.squeeze(0)

class Tester:
    def __init__(self, args):
        """初始化测试器"""
        self.args = args
        self.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
        logger.info(f"使用设备: {self.device}")
        
        # 加载配置
        with open(args.config, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 创建结果目录
        self.result_dir = os.path.join(args.save_dir, 'results', 'test_result')
        os.makedirs(self.result_dir, exist_ok=True)
        
        # 设置数据集
        self._setup_dataset()
        
        # 加载模型
        self._setup_models()
    
    def _setup_dataset(self):
        """初始化测试数据集和数据加载器"""
        test_dataset = DehazeDataset(
            dh_img_dir=self.args.test_dh_img_dir,
            dh_ref_dir=self.args.test_dh_ref_dir,
            cl_ref_dir=self.args.test_cl_ref_dir,
            target_size=self.config.get('target_size', (512, 512))
        )
        
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.config.get('num_workers', 4)
        )
        
        logger.info(f"测试数据集加载完成，共有 {len(test_dataset)} 个样本")
    
    def _setup_models(self):
        """加载预训练模型"""
        # 初始化模型组件
        self.lte_model = LTE(requires_grad=False).to(self.device)
        self.search_transfer = SearchTransfer().to(self.device)
        self.csfi = CSFI(
            in_channels=self.config.get('csfi_params', {}).get('in_channels', 256), 
            n_feats=self.config.get('csfi_params', {}).get('n_feats', 64), 
            res_depth=self.config.get('csfi_params', {}).get('res_depth', [3, 3, 3, 3]),
            res_scale=self.config.get('csfi_params', {}).get('res_scale', 0.1)
        ).to(self.device)
        
        # 加载预训练权重
        logger.info(f"加载模型权重: {self.args.model}")
        checkpoint = torch.load(self.args.model, map_location=self.device)
        
        self.lte_model.load_state_dict(checkpoint['lte'])
        self.search_transfer.load_state_dict(checkpoint['search_transfer'])
        self.csfi.load_state_dict(checkpoint['csfi'])
        
        # 设置为评估模式
        self.lte_model.eval()
        self.search_transfer.eval()
        self.csfi.eval()
        
        logger.info("模型加载完成")
    
    def prepare(self, batch):
        """将批次数据移动到设备上"""
        dh_img = batch['dh_img'].to(self.device)
        dh_ref = batch['dh_ref'].to(self.device)
        cl_ref = batch['cl_ref'].to(self.device)
        return {"dh_img": dh_img, "dh_ref": dh_ref, "cl_ref": cl_ref}
    
    def forward(self, dh_img, dh_ref, cl_ref):
        """前向传播函数"""
        # 特征提取
        dh_img_lv1, dh_img_lv2, dh_img_lv3 = self.lte_model((dh_img + 1.) / 2)
        dh_ref_lv1, dh_ref_lv2, dh_ref_lv3 = self.lte_model((dh_ref + 1.) / 2)
        cl_ref_lv1, cl_ref_lv2, cl_ref_lv3 = self.lte_model((cl_ref + 1.) / 2)
        
        # 特征搜索和转移
        S, T_lv3, T_lv2, T_lv1 = self.search_transfer(
            dh_img_lv3, dh_ref_lv3, 
            cl_ref_lv1, cl_ref_lv2, cl_ref_lv3
        )
        
        # 应用参考特征权重(如果需要)
        if self.args.ref_weight != 1.0:
            T_lv1 = T_lv1 * self.args.ref_weight
            T_lv2 = T_lv2 * self.args.ref_weight
            T_lv3 = T_lv3 * self.args.ref_weight
        
        # 使用CSFI生成输出
        output = self.csfi(dh_img, S, T_lv3, T_lv2, T_lv1)
        
        return output, S, T_lv3, T_lv2, T_lv1
    
    def test(self):
        """测试模型性能"""
        logger.info('开始测试过程...')
        
        _psnr, _ssim = 0., 0.
        _psnr_input, _ssim_input = 0., 0.  # 输入图像相对于清晰图像的评估
        
        # 总批次数
        total_batches = len(self.test_loader)
        
        # SSIM计算器
        ssim_calculator = SSIM(data_range=2.0, size_average=True, channel=3).to(self.device)
        
        with torch.no_grad():
            for i_batch, sample_batch in enumerate(tqdm(self.test_loader, desc="测试进度")):
                # 进度显示
                progress = (i_batch + 1) / total_batches * 100
                logger.info(f'处理批次 {i_batch + 1}/{total_batches} - 进度: {progress:.2f}%')
                
                # 准备数据
                sample_dict = self.prepare(sample_batch)
                dh_img = sample_dict['dh_img']
                dh_ref = sample_dict['dh_ref']
                cl_ref = sample_dict['cl_ref']
                
                # 模型预测
                output, _, _, _, _ = self.forward(dh_img, dh_ref, cl_ref)
                
                # 移除padding后计算指标
                batch_outputs = []
                batch_cl_refs = []
                batch_dh_imgs = []

                for i in range(len(dh_img)):
                    # 获取原始图像信息
                    paste_x = sample_batch['paste_x'][i]
                    paste_y = sample_batch['paste_y'][i]
                    scaled_w = sample_batch['scaled_w'][i]
                    scaled_h = sample_batch['scaled_h'][i]
                    original_w = sample_batch['original_w'][i]
                    original_h = sample_batch['original_h'][i]
                    
                    # 移除padding并恢复原始尺寸
                    restored_output = remove_padding(
                        output[i], paste_x, paste_y, 
                        scaled_w, scaled_h, 
                        original_w, original_h
                    )
                    restored_cl_ref = remove_padding(
                        cl_ref[i], paste_x, paste_y, 
                        scaled_w, scaled_h, 
                        original_w, original_h
                    )
                    restored_dh_img = remove_padding(
                        dh_img[i], paste_x, paste_y, 
                        scaled_w, scaled_h, 
                        original_w, original_h
                    )
                    
                    batch_outputs.append(restored_output.unsqueeze(0))
                    batch_cl_refs.append(restored_cl_ref.unsqueeze(0))
                    batch_dh_imgs.append(restored_dh_img.unsqueeze(0))

                # 合并批次
                batch_outputs = torch.cat(batch_outputs, dim=0)
                batch_cl_refs = torch.cat(batch_cl_refs, dim=0)
                batch_dh_imgs = torch.cat(batch_dh_imgs, dim=0)

                # 计算指标
                batch_psnr = calc_psnr_and_ssim(batch_outputs, batch_cl_refs)
                batch_ssim = ssim_calculator(batch_outputs, batch_cl_refs).item()

                input_psnr = calc_psnr_and_ssim(batch_dh_imgs, batch_cl_refs)
                input_ssim = ssim_calculator(batch_dh_imgs, batch_cl_refs).item()
                
                # 累加指标
                _psnr += batch_psnr
                _ssim += batch_ssim
                _psnr_input += input_psnr
                _ssim_input += input_ssim
                
                # 仅保存处理后的输出图像
                for i in range(len(dh_img)):
                    # 使用零填充文件名
                    index = i_batch * self.args.batch_size + i
                    filename_base = str(index).zfill(5)
                    
                    # 获取原始图像信息
                    paste_x = sample_batch['paste_x'][i]
                    paste_y = sample_batch['paste_y'][i]
                    scaled_w = sample_batch['scaled_w'][i]
                    scaled_h = sample_batch['scaled_h'][i]
                    original_w = sample_batch['original_w'][i]
                    original_h = sample_batch['original_h'][i]
                    
                    # 移除padding并恢复原始尺寸
                    restored_output = remove_padding(
                        output[i], paste_x, paste_y, 
                        scaled_w, scaled_h, 
                        original_w, original_h
                    )
                    
                    # 保存处理后的图像
                    plt.imsave(os.path.join(self.result_dir, f"{filename_base}.png"), 
                              tensor2img(restored_output))
            
            # 计算平均指标
            avg_psnr = _psnr / total_batches
            avg_ssim = _ssim / total_batches
            avg_psnr_input = _psnr_input / total_batches
            avg_ssim_input = _ssim_input / total_batches
        
        # 输出测试结果
        logger.info(f'输入图像 PSNR: {avg_psnr_input:.3f} \t SSIM: {avg_ssim_input:.4f}')
        logger.info(f'RefID输出 PSNR: {avg_psnr:.3f} \t SSIM: {avg_ssim:.4f}')
        logger.info(f'PSNR提升: {(avg_psnr - avg_psnr_input):.3f}, \t {((avg_psnr - avg_psnr_input) / avg_psnr_input * 100):.2f}%')
        logger.info(f'SSIM提升: {(avg_ssim - avg_ssim_input):.4f}, \t {((avg_ssim - avg_ssim_input) / avg_ssim_input * 100):.2f}%')
        logger.info(f'结果保存路径: {self.result_dir}')
        logger.info('测试完成.')

def main():
    parser = argparse.ArgumentParser(description='RefID测试脚本')
    parser.add_argument('--model', type=str, required=True, help='预训练模型路径')
    parser.add_argument('--config', type=str, default='./configs/training_config.yaml', help='配置文件路径')
    parser.add_argument('--test_dh_img_dir', type=str, required=True, help='测试有雾图像目录')
    parser.add_argument('--test_dh_ref_dir', type=str, required=True, help='测试参考有雾图像目录')
    parser.add_argument('--test_cl_ref_dir', type=str, required=True, help='测试参考清晰图像目录')
    parser.add_argument('--save_dir', type=str, default='./output', help='结果保存目录')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--ref_weight', type=float, default=1.0, help='参考特征权重(0-1)')
    parser.add_argument('--cuda', action='store_true', help='使用CUDA加速计算')
    args = parser.parse_args()
    
    # 创建测试器并运行测试
    tester = Tester(args)
    tester.test()

if __name__ == "__main__":
    main()