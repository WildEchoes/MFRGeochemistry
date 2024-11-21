import os
import argparse
import time

from osgeo import gdal
import numpy as np

from model import MFRGeoChemistry
from dataset import GeochemDatasetTest
from test_patchprocess import getPatchMultiThread, smooth_edge

import torch
from torch.utils.data import DataLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--img_path', type=str)
    parser.add_argument('--tif_patch', type=str)
    parser.add_argument('--save_path', type=str)
    
    parser.add_argument('--label', type=str)
    parser.add_argument('--labels', type=list, default=['Fe2O3', 'K2O', 'Na2O', 'SiO2'])
    parser.add_argument('--patchsize', type=int, default=50)
    
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=128)

    return parser.parse_args()


def save_result(opt, outputs):
    ds = gdal.Open(opt.tif_patch)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    
    driver = gdal.GetDriverByName('GTiff')
    ds_out = driver.Create(os.path.join(opt.save_path, f'{opt.label}.tif'), ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Float32)

    ds_out.SetGeoTransform(gt)
    ds_out.SetProjection(proj)
    
    # 初始化一个与原图相同大小的数组，用于存储重组后的结果
    final_results = np.zeros((ds.RasterYSize, ds.RasterXSize), dtype=np.float32)

    patch_size = opt.patchsize  # Patch的大小
    n_patches_x = ds.RasterXSize // patch_size  # 横向的patch数量

    # 根据outputs的形状来确定如何迭代和放置每个预测结果
    for idx, output in enumerate(outputs):
        # 计算当前patch在原图中的位置
        row_idx = idx // n_patches_x
        col_idx = idx % n_patches_x
        x = col_idx * patch_size
        y = row_idx * patch_size

        # 将当前patch的预测结果放置到final_results中对应的位置
        final_results[y:y+patch_size, x:x+patch_size] = output  # output已经是(50, 50)形状
    
    # 使用GDAL写入结果
    ds_out.GetRasterBand(1).WriteArray(final_results)
    ds_out.FlushCache()
  
def save_result_smooth(opt, outputs):
    ds = gdal.Open(opt.tif_patch)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjection()
    
    driver = gdal.GetDriverByName('GTiff')
    ds_out = driver.Create(os.path.join(opt.save_path, f'{opt.label}.tif'), ds.RasterXSize, ds.RasterYSize, 1, gdal.GDT_Float32)

    ds_out.SetGeoTransform(gt)
    ds_out.SetProjection(proj)
    
    # 初始化一个与原图相同大小的数组，用于存储重组后的结果
    final_results = np.zeros((ds.RasterYSize, ds.RasterXSize), dtype=np.float32)

    patch_size = opt.patchsize  # Patch的大小
    n_patches_x = ds.RasterXSize // patch_size  # 横向的patch数量

    # 根据outputs的形状来确定如何迭代和放置每个预测结果
    for idx, output in enumerate(outputs):
        # 计算当前patch在原图中的位置
        row_idx = idx // n_patches_x
        col_idx = idx % n_patches_x
        x = col_idx * patch_size
        y = row_idx * patch_size

        # 将当前patch的预测结果放置到final_results中对应的位置
        final_results[y:y+patch_size, x:x+patch_size] = output  # output已经是(50, 50)形状
    
    # 对final_results进行平滑处理
    final_results = smooth_edge(final_results, patch_size)
    
    # 使用GDAL写入结果
    ds_out.GetRasterBand(1).WriteArray(final_results)
    ds_out.FlushCache()


def test(opt):
    # dataset
    print('Loading dataset...')
    datalist = getPatchMultiThread(opt.img_path, patch_size=opt.patchsize)
    dataset = GeochemDatasetTest(datalist)
    del datalist
    
    dataset_loader = DataLoader(dataset, batch_size=opt.batch_size, num_workers=6, shuffle=False)
    
    for label in opt.labels:
        if not os.path.exists(os.path.join(opt.model_path, label, f'{label}.pth')):
            print(f'{label} model does not exist!')
            return
        else:
            model_path = os.path.join(opt.model_path, label, f'{label}.pth')
        opt.label = label 
        
        model = MFRGeoChemistry()
        model.load_state_dict(torch.load(model_path))
        
        if torch.cuda.is_available():
            model.cuda(opt.gpu)
        else:
            print("CUDA is not available. The model will run on CPU.")
    
        model.eval()
        outputs = []
        print('Evaluating...')
        stime = time.time()
        with torch.no_grad():
            for data in dataset_loader:
                if torch.cuda.is_available():
                    data = data.cuda(opt.gpu)
                else:
                    data = data.cpu()
                
                output = model(data) # Nx1x50x50
                outputs.append(output.cpu().numpy())
        # 去掉单维，合并
        outputs = np.concatenate([np.squeeze(batch_output) for batch_output in outputs], axis=0) 
        etime = time.time()
        
        # save_result(opt, outputs)
        save_result_smooth(opt, outputs)
        print(f'{label} has been saved! Time: {etime-stime:.2f}s')
        
        del outputs
        
    del dataset_loader, dataset
    
    
if __name__ == '__main__':
    opt = parse_args()
    
    opt.save_path = './results'
    
    if not os.path.exists(opt.save_path):
        os.makedirs(opt.save_path)
        
    labels = ['']
    model_path = './model'
    
    opt.model_path = model_path
    opt.batch_size = 256
    opt.labels = labels
    
    torch.cuda.empty_cache()
    test(opt)
    