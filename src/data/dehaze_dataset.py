#src/dehaze_dataset.py

import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from glob import glob

class DehazeDataset(Dataset):
    """保持长宽比的padding数据集实现"""
    def __init__(self, dh_img_dir, dh_ref_dir, cl_ref_dir, target_size=(256, 256)):
        self.target_size = target_size
        self.pad_value = [0.7, 0.7, 0.7]
        self.file_triplets = self._match_files(dh_img_dir, dh_ref_dir, cl_ref_dir)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    def _match_files(self, dir1, dir2, dir3):
        """匹配三个目录中的对应文件"""
        files1 = sorted(glob(os.path.join(dir1, '*.png')))
        files2 = sorted(glob(os.path.join(dir2, '*.png')))
        files3 = sorted(glob(os.path.join(dir3, '*.png')))
        assert len(files1) == len(files2) == len(files3)
        return list(zip(files1, files2, files3))
    
    def _smart_pad(self, img):
        """智能padding保持长宽比"""
        w, h = img.size
        scale = min(self.target_size[1]/w, self.target_size[0]/h)
        new_w, new_h = int(w*scale), int(h*scale)
        img = img.resize((new_w, new_h), Image.BICUBIC)
        padded_img = Image.new('RGB', (self.target_size[1], self.target_size[0]), 
                             color=tuple(int(255*x) for x in self.pad_value))
        paste_pos = ((self.target_size[1]-new_w)//2, (self.target_size[0]-new_h)//2)
        padded_img.paste(img, paste_pos)
        return padded_img
    
    def __len__(self):
        # 确保返回实际图像列表的长度
        return len(self.file_triplets)
    
    def __getitem__(self, idx):
        # 确保能正确加载图像
        img1 = Image.open(self.file_triplets[idx][0]).convert('RGB')
        img2 = Image.open(self.file_triplets[idx][1]).convert('RGB')
        img3 = Image.open(self.file_triplets[idx][2]).convert('RGB')
        
        # 记录原始尺寸
        original_size = img1.size  # (width, height)
        orig_w, orig_h = original_size
        
        # 计算缩放和padding信息
        w, h = original_size
        scale = min(self.target_size[1]/w, self.target_size[0]/h)
        new_w, new_h = int(w*scale), int(h*scale)
        paste_pos = ((self.target_size[1]-new_w)//2, (self.target_size[0]-new_h)//2)
        paste_x, paste_y = paste_pos
        
        # 应用智能padding
        img1 = self._smart_pad(img1)
        img2 = self._smart_pad(img2)
        img3 = self._smart_pad(img3)
        
        # 返回字典包含原始图像信息，拆分元组为单独字段
        return {
            'dh_img': self.normalize(self.to_tensor(img1)),
            'dh_ref': self.normalize(self.to_tensor(img2)),
            'cl_ref': self.normalize(self.to_tensor(img3)),
            'original_w': orig_w,
            'original_h': orig_h,
            'scaled_w': new_w,
            'scaled_h': new_h,
            'paste_x': paste_x,
            'paste_y': paste_y,
            'filename': os.path.basename(self.file_triplets[idx][0])
        }