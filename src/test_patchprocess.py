import os
import glob
import time
from osgeo import gdal
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def processPatch(imgpath: str, i: int, j: int, m_PatchSize: int, num_patch_per_row: int) -> tuple:
    """split image into patches

    Args:
        imgpath (str): img data path
        i (int): y index
        j (int): x index
        m_PatchSize (int): patch size

    Returns:
        tuple[int, np.ndarray]: index and patch
    """
    ds = gdal.Open(imgpath, gdal.GA_ReadOnly)
    imgPatch = ds.ReadAsArray(j, i, m_PatchSize, m_PatchSize)

    # 将坐标 (i, j) 转化为单一的索引值
    index: int = (i // m_PatchSize) * num_patch_per_row + j // m_PatchSize

    return (index, imgPatch)  # 返回结果时包含单一索引和 patch


def getPatchMultiThread(img_path: str, patch_size: int = 50) ->list:
    file_list = glob.glob(os.path.join(img_path, '*.tif'))
    
    img_path = file_list[0]
    ds = gdal.Open(img_path)
    xsize, ysize = ds.RasterXSize, ds.RasterYSize
    del ds
    
    nppr = xsize // patch_size

    # 全图区域
    with Pool(processes=cpu_count()-1) as pool:
        jobs1 = [(img_path, i, j, patch_size, nppr) for i in range(0, ysize, patch_size) for j in range(0, xsize, patch_size)]

        results1 = list(tqdm(pool.starmap(processPatch, jobs1), total=len(jobs1), desc='Processing patches: '))

    # 恢复 patch 的顺序
    results_sorted1 = sorted(results1, key=lambda x: x[0])
    # 只保留 patch 数据
    final_results1:list = [result[1] for result in results_sorted1]
    
    return final_results1


def reshapeAllTiff(img_path: str) -> None:
    # make tiff file fix patch size
    file_list = glob.glob(os.path.join(img_path, '*.tif'))
    ds = [gdal.Open(i) for i in file_list]
    
    gt = ds[0].GetGeoTransform()
    proj = ds[0].GetProjection()

    band_data = [data.GetRasterBand(band).ReadAsArray() for data in ds for band in range(1, data.RasterCount + 1)]
    del ds
    
    img = np.stack(band_data)
    # --------修改这里的裁剪范围--------
    img = img[:, 0:2650, 0:11300]
    
    m_driver = gdal.GetDriverByName('GTiff')
    m_ds = m_driver.Create(os.path.join(img_path, 'xkl_forTestUse.tif'), img.shape[2], img.shape[1], img.shape[0], gdal.GDT_Float32)
    
    m_ds.SetGeoTransform(gt)
    m_ds.SetProjection(proj)
    
    for i in range(img.shape[0]):
        m_ds.GetRasterBand(i+1).WriteArray(img[i])
    
    m_ds.FlushCache()
    print(f"Finish all files!")
    

def smooth_edge(img: np.ndarray, patch_size: int) -> np.ndarray:
    """smooth the edge of the image

    Args:
        img (np.ndarray): image data
        patch_size (int): patch size

    Returns:
        np.ndarray: smoothed image
    """
    
    h: int = img.shape[0]
    w: int = img.shape[1]
    
    new_img = np.copy(img)

    # Iterate over each patch boundary vertically
    for x in range(patch_size -1, w, patch_size):  
        if x + 2 < w:  
            new_img[:, x] = img[:, x-1]
            new_img[:, x+1] = img[:, x+2]
            new_img[:, x] = (new_img[:, x] + new_img[:, x+1])/2
            new_img[:, x+1] = (new_img[:, x] + new_img[:, x+1])/2
            
    
    # Iterate over each patch boundary horizontally
    for y in range(patch_size - 1, h, patch_size):  
        if y + 2 < h:  
            new_img[y, :] = new_img[y-1, :]
            new_img[y+1, :] = new_img[y+2, :]
            new_img[y, :] = (new_img[y, :] + new_img[y+1, :])/2
            new_img[y+1, :] = (new_img[y, :] + new_img[y+1, :])/2   
            
    return new_img


if __name__ == "__main__":
    img_path = 'E:\\WorkSpace\\新疆三考数据\\202409 研究区\\xikunlun\\train_data'
    reshapeAllTiff(img_path)
    print("Done!")
    