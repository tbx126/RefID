import os
import argparse
import torch
import yaml
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
import torch.nn.functional as F

# 导入自定义模块
from models.LTE import LTE
from models.search_transfer import SearchTransfer
from models.csfi import CSFI
from utils import calc_psnr_and_ssim

def parse_args():
    parser = argparse.ArgumentParser(description='RefID 图像去雾推理')
    parser.add_argument('--input', type=str, required=True, help='输入有雾图像路径或目录')
    parser.add_argument('--output', type=str, default='./results', help='输出目录')
    parser.add_argument('--model', type=str, required=True, help='预训练模型路径')
    parser.add_argument('--config', type=str, default='./configs/training_config.yaml', help='配置文件路径')
    parser.add_argument('--second_input', type=str, default=None, help='第二个有雾输入图像(可选)')
    parser.add_argument('--reference', type=str, default=None, help='参考清晰图像(可选)')
    parser.add_argument('--cuda', action='store_true', help='使用CUDA加速推理')
    parser.add_argument('--fixed_size', action='store_true', help='使用固定大小输出(256x256)')
    parser.add_argument('--process_size', type=int, default=512, help='网络处理时的图像大小')
    return parser.parse_args()

def load_model(model_path, config, device):
    # 初始化模型
    lte_model = LTE(requires_grad=False).to(device)
    search_transfer = SearchTransfer().to(device)
    csfi = CSFI(
        in_channels=config.get('csfi_params', {}).get('in_channels', 256), 
        n_feats=config.get('csfi_params', {}).get('n_feats', 64), 
        res_depth=config.get('csfi_params', {}).get('res_depth', [3, 3, 3, 3]),
        res_scale=config.get('csfi_params', {}).get('res_scale', 0.1)
    ).to(device)
    
    # 加载预训练权重
    checkpoint = torch.load(model_path, map_location=device)
    lte_model.load_state_dict(checkpoint['lte'])
    search_transfer.load_state_dict(checkpoint['search_transfer'])
    csfi.load_state_dict(checkpoint['csfi'])
    
    # 设置为评估模式
    lte_model.eval()
    search_transfer.eval()
    csfi.eval()
    
    return lte_model, search_transfer, csfi

def process_image(hazy_path, second_hazy_path, reference_path, lte_model, search_transfer, csfi, device, fixed_size=False, process_size=256):
    # 加载第一个有雾图像并保存原始尺寸
    hazy_img = Image.open(hazy_path).convert('RGB')
    original_size = hazy_img.size  # 保存原始宽高 (width, height)
    
    # 图像预处理 - 使用process_size进行网络处理
    transform = transforms.Compose([
        transforms.Resize((process_size, process_size)),  # 调整为处理尺寸
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 转换到[-1,1]范围
    ])
    
    # 转换第一个有雾图像
    dehazed1 = transform(hazy_img).unsqueeze(0).to(device)
    
    # 加载第二个有雾图像(如果提供)，否则使用第一个
    if second_hazy_path and os.path.exists(second_hazy_path):
        hazy_img2 = Image.open(second_hazy_path).convert('RGB')
        dehazed2 = transform(hazy_img2).unsqueeze(0).to(device)
    else:
        dehazed2 = dehazed1  # 复用第一个有雾图像
        
    # 加载参考清晰图像(如果提供)，否则使用第一个有雾图像
    if reference_path and os.path.exists(reference_path):
        clear_img = Image.open(reference_path).convert('RGB')
        clear = transform(clear_img).unsqueeze(0).to(device)
    else:
        clear = dehazed1  # 使用有雾图像作为自参考
    
    with torch.no_grad():
        # 特征提取
        dehazed1_lv1, dehazed1_lv2, dehazed1_lv3 = lte_model((dehazed1 + 1.) / 2)
        dehazed2_lv1, dehazed2_lv2, dehazed2_lv3 = lte_model((dehazed2 + 1.) / 2)
        clear_lv1, clear_lv2, clear_lv3 = lte_model((clear + 1.) / 2)
        
        # 特征搜索和转移
        S, T_lv3, T_lv2, T_lv1 = search_transfer(
            dehazed1_lv3, dehazed2_lv3, 
            clear_lv1, clear_lv2, clear_lv3
        )
        
        # 使用CSFI生成输出
        output = csfi(dehazed1, S, T_lv3, T_lv2, T_lv1)
        
        # 如果不使用固定大小，将输出调整回原始尺寸
        if not fixed_size:
            output = F.interpolate(output, size=(original_size[1], original_size[0]), 
                                 mode='bicubic', align_corners=True)  # 改为True
    
    # 如果提供了参考图像，计算PSNR
    psnr = None
    if reference_path and os.path.exists(reference_path) and reference_path != hazy_path:
        try:
            # 如果要计算准确的PSNR，需要将参考图像也调整到相同尺寸
            if not fixed_size:
                reference_transform = transforms.Compose([
                    transforms.Resize((original_size[1], original_size[0])),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                ])
                clear_orig_size = reference_transform(clear_img).unsqueeze(0).to(device)
                psnr = calc_psnr_and_ssim(output, clear_orig_size)
            else:
                psnr = calc_psnr_and_ssim(output, clear)
        except Exception as e:
            print(f"计算PSNR时出错: {e}")
    
    return output, psnr, original_size

def main():
    args = parse_args()
    
    # 设置设备
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 加载模型
    lte_model, search_transfer, csfi = load_model(args.model, config, device)
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 检查输入是文件还是目录
    if os.path.isfile(args.input):
        # 处理单个文件
        output, psnr, original_size = process_image(
            args.input, 
            args.second_input, 
            args.reference, 
            lte_model, search_transfer, csfi, device,
            fixed_size=args.fixed_size,
            process_size=args.process_size
        )
        output_filename = os.path.join(args.output, os.path.basename(args.input))
        # 将输出从[-1,1]转换到[0,1]并保存
        save_image((output+1)/2, output_filename, nrow=1, padding=0)
        print(f"结果已保存到 {output_filename}，原始尺寸: {original_size}")
        
        if psnr:
            print(f"PSNR: {psnr:.2f} dB")
    else:
        # 处理整个目录
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        for filename in os.listdir(args.input):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                input_path = os.path.join(args.input, filename)
                
                # 查找对应的第二输入图像和参考图像(如果提供了目录)
                second_input_path = None
                if args.second_input and os.path.isdir(args.second_input):
                    potential_path = os.path.join(args.second_input, filename)
                    if os.path.exists(potential_path):
                        second_input_path = potential_path
                elif args.second_input:
                    second_input_path = args.second_input
                
                reference_path = None
                if args.reference and os.path.isdir(args.reference):
                    potential_path = os.path.join(args.reference, filename)
                    if os.path.exists(potential_path):
                        reference_path = potential_path
                elif args.reference:
                    reference_path = args.reference
                
                output, psnr, original_size = process_image(
                    input_path, 
                    second_input_path, 
                    reference_path, 
                    lte_model, search_transfer, csfi, device,
                    fixed_size=args.fixed_size,
                    process_size=args.process_size
                )
                output_filename = os.path.join(args.output, filename)
                save_image((output+1)/2, output_filename, nrow=1, padding=0)
                print(f"已处理 {filename} -> {output_filename}，原始尺寸: {original_size}")
                if psnr:
                    print(f"PSNR: {psnr:.2f} dB")

if __name__ == "__main__":
    main()