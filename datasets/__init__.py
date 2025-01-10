import torch.utils.data
import torchvision

from .high import build as build_high

data_path = {
    'HIGH':r'C:\Users\kakar\Desktop\high_2048', #用于测试
    # 'HIGH':'G:\exp\datasets\high_2048'#完整版数据集
}

def build_dataset(image_set, args):
    args.data_path = data_path[args.dataset_file]
    if args.dataset_file == 'HIGH':
        return build_high(image_set, args)
    raise ValueError(f'dataset {args.dataset_file} not supported')
