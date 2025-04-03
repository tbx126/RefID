# batch_inference.py
import os
import argparse
import yaml
import torch
from tqdm import tqdm
from inference import inference, load_image

def batch_inference(hazy_dir, gt_dir, checkpoint_path, output_dir, config):
    """
    批量处理文件夹中的图像
    
    参数:
        hazy_dir: 有雾图像文件夹
        gt_dir: 真实无雾图像文件夹(可选)
        checkpoint_path: 模型检查点路径
        output_dir: 输出目录
        config: 配置字典
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_files = [f for f in os.listdir(hazy_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    
    if len(image_files) == 0:
        print(f"警告: 在 {hazy_dir} 中未找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 张图像，开始批量去雾处理...")
    
    # 遍历所有图像进行处理
    for image_file in tqdm(image_files, desc="处理进度"):
        hazy_path = os.path.join(hazy_dir, image_file)
        
        # 检查是否存在对应的真实图像
        gt_path = None
        if gt_dir is not None:
            potential_gt = os.path.join(gt_dir, image_file)
            if os.path.exists(potential_gt):
                gt_path = potential_gt
        
        # 对当前图像进行去雾处理
        try:
            inference(hazy_path, gt_path, checkpoint_path, output_dir, config)
        except Exception as e:
            print(f"处理 {image_file} 时出错: {str(e)}")
    
    print(f"批量处理完成！所有结果保存在 {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量图像去雾处理")
    parser.add_argument("--hazy_dir", required=True, help="包含有雾图像的文件夹")
    parser.add_argument("--gt_dir", default=None, help="包含真实无雾图像的文件夹(可选)")
    parser.add_argument("--checkpoint", required=True, help="模型检查点路径")
    parser.add_argument("--output", default="./batch_results", help="输出目录")
    parser.add_argument("--config", default="./configs/training_config.yaml", help="配置文件")
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # 执行批量处理
    batch_inference(args.hazy_dir, args.gt_dir, args.checkpoint, args.output, config)