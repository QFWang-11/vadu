import os.path as osp
import numpy as np
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt
import torch
from scipy.io import loadmat
from .cuhk_sysu import CUHKSYSU

class CUHKSYSUPaired(CUHKSYSU):
    def __init__(self, root, transforms, split):
        # 移除原始view参数，加载全部数据
        super(CUHKSYSUPaired, self).__init__(root, transforms, split)
        
        # 分离地面和空中数据
        self.ground_annos = [a for a in self.annotations if a['img_name'].startswith('S')]
        print('ground_annos',len(self.ground_annos))
        self.aerial_annos = [a for a in self.annotations if a['img_name'].startswith('D')]
        print('aerial_annos',len(self.aerial_annos))

        # 训练时补全数据
        if split == 'train':
            self._pad_aerial_dataset()
            
        # 生成配对索引
        self.pair_indices = self._build_pair_indices()   #pairs是一个列表，列表中每个元素表示一个地面配对元组
        print('pair_num',len(self.pair_indices))

    def _pad_aerial_dataset(self):
        """补全空中数据到与地面数据相同数量"""
        num_ground = len(self.ground_annos)
        num_aerial = len(self.aerial_annos)
        if num_aerial < num_ground:
            pad_num = num_ground - num_aerial
            pad_indices = np.random.choice(len(self.aerial_annos), pad_num, replace=False)  #不允许重复
            self.aerial_annos += [self.aerial_annos[i] for i in pad_indices]       #补全空中视角的annotations大列表

    def _parse_metadata(self, img_name):
        """解析文件名元数据
        Args:
            img_name (str): 文件名 (e.g. D1_50_000553.jpg)
        Returns:
            (scene, height, timestamp): (1, 50, 553)
        """
        parts = img_name.split('_')
        scene = parts[0][1:]  # 提取场景编号 (D1 -> 1)
        height = parts[1]     # 提取 50
        timestamp = parts[2].split('.')[0]  # 去除扩展名
        return scene, height, timestamp    #'1'  '50'  '000553' str数据类型

    def _build_pair_indices(self):
        """构建配对索引字典"""
        # 构建空中数据索引 { (scene,height): {timestamp: index} }
        aerial_index = defaultdict(dict)   
        for idx, anno in enumerate(self.aerial_annos):
            scene, height, ts = self._parse_metadata(anno['img_name'])
            aerial_index[(scene, height)][int(ts)] = idx   #key是(scene, height), 值是一个字典{timestamp: index}键是时间戳，值是图像索引idx，根据场景和高度索引图象，同时按照时间戳排序
        
        pairs = []
        for ground_idx, ground_anno in enumerate(self.ground_annos):   #遍历地面视角图像
            g_scene, g_height, g_ts = self._parse_metadata(ground_anno['img_name'])
            key = (g_scene, g_height)   #构建key
            
            # 寻找匹配的空中数据,即相同场景，相同高度
            if key not in aerial_index:    #如果不存在（即该地面场景和高度没有对应的空中图像），则跳过该地面图像，继续处理下一个地面图像。
                continue
                
            # 寻找最接近的时间戳
            aerial_ts = np.array(list(aerial_index[key].keys()))   #如果找到了匹配的空中数据，代码从 aerial_index[key] 中获取所有可能的时间戳（aerial_ts）
            ts_diff = np.abs(aerial_ts - int(g_ts))                #计算每个空中图像时间戳与地面图像时间戳 g_ts 的绝对时间差（ts_diff）
            closest_idx = np.argmin(ts_diff)                       #使用 np.argmin(ts_diff) 找到时间差最小的索引 closest_idx，即时间戳最接近的空中图像。
            aerial_idx = aerial_index[key][aerial_ts[closest_idx]] #获取该空中图像的索引 aerial_idx。
            
            pairs.append( (ground_idx, aerial_idx) )               #将当前地面图像的索引 ground_idx 和最接近的空中图像的索引 aerial_idx 作为一个元组 (ground_idx, aerial_idx) 添加到 pairs 列表中。
        
        return pairs    #pairs是一个列表，列表中每个元素表示一个地面配对元组

    def __len__(self):
        return len(self.pair_indices)   #表示生成的数据集是成对出现的

    def __getitem__(self, idx):
        """返回配对的样本"""
        g_idx, a_idx = self.pair_indices[idx]
        
        # 加载地面数据,图像和标签
        g_anno = self.ground_annos[g_idx]
        g_img = Image.open(g_anno['img_path']).convert('RGB')
        g_boxes = torch.as_tensor(g_anno["boxes"], dtype=torch.float32)
        g_labels = torch.as_tensor(g_anno["pids"], dtype=torch.int64)
        g_target = {"img_name": g_anno["img_name"], "boxes": g_boxes, "labels": g_labels}
        if self.transforms is not None:
            g_img, g_target = self.transforms(g_img, g_target)
        
        # 加载空中数据,图像和标签
        a_anno = self.aerial_annos[a_idx]
        a_img = Image.open(a_anno['img_path']).convert('RGB')
        a_boxes = torch.as_tensor(a_anno["boxes"], dtype=torch.float32)
        a_labels = torch.as_tensor(a_anno["pids"], dtype=torch.int64)
        a_target = {"img_name": a_anno["img_name"], "boxes": a_boxes, "labels": a_labels}
        if self.transforms is not None:
            a_img, a_target = self.transforms(a_img, a_target)
        return [[g_img,g_target],[a_img,a_target]]

def uni_collate_fn(batch):
    imgs,targets=[],[]
    for pair in batch:
        imgs.extend([pair[0][0],pair[1][0]])     #pair[0][0]地面视角图片, pair[1][0]空中视角图片
        targets.extend([pair[0][1],pair[1][1]])  #pair[0][1]地面target, pair[1][1]空中target
    return imgs,targets
