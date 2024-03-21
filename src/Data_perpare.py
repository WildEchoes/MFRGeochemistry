"""
This file is used to prepare the data for the model
"""

import os
import glob
import numpy as np
from osgeo import gdal, ogr
from tqdm import tqdm

import lmdb

"""1 tif dataset"""
def extract_patche(tif_path, shp_path, output_size=512):
    """test function code / save to TIFF"""
    assert os.path.exists(tif_path), f"{tif_path} not exists!"
    assert os.path.exists(shp_path), f"{shp_path} not exists!"
    
    # 打开栅格和矢量数据集
    raster_ds = gdal.Open(tif_path, gdal.GA_ReadOnly)
    vector_ds = ogr.Open(shp_path)
    layer = vector_ds.GetLayer()

    # 获取栅格信息
    gt = raster_ds.GetGeoTransform()
    rb = raster_ds.GetRasterBand(1)

    # 遍历Shapefile中的每个几何体
    for feature in layer:
        geom = feature.GetGeometryRef()
        # 获取与当前几何体相交的栅格影像区域
        minx, maxx, miny, maxy = geom.GetEnvelope()

        # 计算栅格影像中的像素坐标范围
        xoff = int((minx - gt[0]) / gt[1])
        yoff = int((maxy - gt[3]) / gt[5])
        xcount = output_size
        ycount = output_size

        # 坐标读取数据
        data = rb.ReadAsArray(xoff, yoff, xcount, ycount)

        # 创建一个与目标尺寸匹配的内存数据集
        # 创建内存驱动程序，用于临时存储裁剪的影像
        mem_driver = gdal.GetDriverByName("GTiff")
        out_ds = mem_driver.Create("D:\\MyProject\\MFRVim\\src2\\1.tif", output_size, output_size, 1, gdal.GDT_Float32)
        
        out_ds.SetGeoTransform((xoff * gt[1] + gt[0], gt[1], 0, yoff * gt[5] + gt[3], 0, gt[5]))
        out_ds.SetProjection(raster_ds.GetProjection())
        out_band = out_ds.GetRasterBand(1)

        # 写入数据
        out_band.WriteArray(data)

        # 保存数据集并释放内存
        out_ds = None

    raster_ds = None
    vector_ds = None


"""2 npy dataset"""
def extract_and_merge_patches(tif_path, shp_path, aux_image_paths, output_path, ii, output_size=512):
    """
    提取遥感影像和其它地球数据
    
    tif_path: 主影像文件路径
    shp_path: 矢量文件路径
    aux_image_paths: 一维影像文件路径列表，长度应为3
    output_path: 输出.npy文件的保存路径
    output_size: 输出影像块的大小（像素）
    pixel_size: 栅格的像素大小（米）
    """
    # 打开主影像文件
    main_raster_ds = gdal.Open(tif_path, gdal.GA_ReadOnly)
    vector_ds = ogr.Open(shp_path)
    layer = vector_ds.GetLayer()
    
    gt = main_raster_ds.GetGeoTransform()
    num_main_bands = main_raster_ds.RasterCount

    # 准备辅助影像数据
    aux_data = [gdal.Open(path, gdal.GA_ReadOnly) for path in aux_image_paths]

    for feature in layer:
        geom = feature.GetGeometryRef()
        minx, maxx, miny, maxy = geom.GetEnvelope()
        xoff = int((minx - gt[0]) / gt[1])
        yoff = int((maxy - gt[3]) / gt[5])
        xcount = output_size
        ycount = output_size
        
        # 读取主影像数据
        main_data = np.zeros((num_main_bands, output_size, output_size), dtype=np.float32)
        for band in range(1, num_main_bands + 1):
            data = main_raster_ds.GetRasterBand(band).ReadAsArray(xoff, yoff, xcount, ycount)
            main_data[band-1, :, :] = data[:output_size, :output_size]
        
        # 读取并添加辅助影像数据
        for i, aux_ds in enumerate(aux_data):
            aux_band = aux_ds.GetRasterBand(1)
            aux_data = aux_band.ReadAsArray(xoff, yoff, xcount, ycount)
            # 假设辅助影像也是单波段，直接扩展数组维度以匹配主数据
            aux_data_reshaped = aux_data[:output_size, :output_size].reshape(1, output_size, output_size)
            main_data = np.vstack((main_data, aux_data_reshaped))

        # 保存为.npy文件
        np.save(os.path.join(output_path, f"{ii}.npy"), main_data)

    # 关闭数据集
    main_raster_ds = None
    vector_ds = None
    for ds in aux_data:
        ds = None


def extract_geochem_patches(tif_path, shp_path, output_path, ii, output_size=512):
    """
    提取地球化学数据
    
    tif_path: 主影像文件路径
    shp_path: 矢量文件路径
    aux_image_paths: 一维影像文件路径列表，长度应为3
    output_path: 输出.npy文件的保存路径
    output_size: 输出影像块的大小（像素）
    pixel_size: 栅格的像素大小（米）
    """
    assert os.path.exists(tif_path), f"{tif_path} not exists!"
    assert os.path.exists(shp_path), f"{shp_path} not exists!"
    
    # 打开影像文件
    main_raster_ds = gdal.Open(tif_path, gdal.GA_ReadOnly)
    vector_ds = ogr.Open(shp_path)
    layer = vector_ds.GetLayer()
    
    gt = main_raster_ds.GetGeoTransform()
    num_main_bands = main_raster_ds.RasterCount

    for feature in layer:
        geom = feature.GetGeometryRef()
        minx, maxx, miny, maxy = geom.GetEnvelope()
        xoff = int((minx - gt[0]) / gt[1])
        yoff = int((maxy - gt[3]) / gt[5])
        xcount = output_size
        ycount = output_size
        
        # 读取主影像数据
        main_data = np.zeros((num_main_bands, output_size, output_size), dtype=np.float32)
        for band in range(1, num_main_bands + 1):
            data = main_raster_ds.GetRasterBand(band).ReadAsArray(xoff, yoff, xcount, ycount)
            main_data[band-1, :, :] = data[:output_size, :output_size] 

        # 保存为.npy文件
        np.save(os.path.join(output_path, f"{ii}.npy"), main_data)

    # 关闭数据集
    main_raster_ds = None
    vector_ds = None


"""3 npy dataset ----split into smaller patches"""
def extract_merge_and_split_patches(tif_path, shp_path, aux_image_paths, output_path, ii, output_size=504, split_size=21):
    """
    提取遥感影像与其他地球观测数据，将大尺寸patch合并后分割成多个小尺寸patch并保存。
    大patch被等分割，按从左至右、从上至下的顺序进行编号和保存。

    参数:
    tif_path: 主影像文件路径。
    shp_path: 矢量文件路径。
    aux_image_paths: 辅助影像文件路径列表，预期长度为3。
    output_path: 输出文件的保存路径。
    ii: 输出文件的起始编号。
    output_size: 输出影像块的大小（像素），默认512。
    split_size: 分割后小影像块的大小（像素），默认21。
    """
    # 打开主影像文件
    main_raster_ds = gdal.Open(tif_path, gdal.GA_ReadOnly)
    vector_ds = ogr.Open(shp_path)
    layer = vector_ds.GetLayer()
    gt = main_raster_ds.GetGeoTransform()

    # 准备辅助影像数据
    aux_data = [gdal.Open(path, gdal.GA_ReadOnly) for path in aux_image_paths]
    num_main_bands = main_raster_ds.RasterCount + len(aux_data)  # 更新带有辅助数据的波段计数

    patch_id_start = (ii - 1) * ((output_size/split_size)**2) + 1
    patch_id = int(patch_id_start)  # 初始化patch编号

    for feature in layer:
        geom = feature.GetGeometryRef()
        minx, maxx, miny, maxy = geom.GetEnvelope()
        xoff = int((minx - gt[0]) / gt[1])
        yoff = int((maxy - gt[3]) / gt[5])
        
        # 读取并合并影像数据
        main_data = np.zeros((num_main_bands, output_size, output_size), dtype=np.float32)
        for band in range(1, num_main_bands + 1):
            if band <= main_raster_ds.RasterCount:
                data = main_raster_ds.GetRasterBand(band).ReadAsArray(xoff, yoff, output_size, output_size)
            else:
                aux_ds = aux_data[band - main_raster_ds.RasterCount - 1]
                data = aux_ds.GetRasterBand(1).ReadAsArray(xoff, yoff, output_size, output_size)
            main_data[band-1, :, :] = data

        # 分割大patch为多个小patch并保存
        num_x_splits = output_size // split_size
        num_y_splits = output_size // split_size
        for y in range(num_y_splits):
            for x in range(num_x_splits):
                split_patch = main_data[:, y*split_size:(y+1)*split_size, x*split_size:(x+1)*split_size]
                np.save(os.path.join(output_path, f"patch_{patch_id}.npy"), split_patch)
                patch_id += 1

    # 关闭数据集
    main_raster_ds = None
    vector_ds = None
    for ds in aux_data:
        ds = None


def extract_geochem_and_split_patches(tif_path, shp_path, output_path, ii, output_size=504, split_size=21):
    """
    提取遥感影像与其他地球观测数据，将大尺寸patch合并后分割成多个小尺寸patch并保存。
    大patch被等分割，按从左至右、从上至下的顺序进行编号和保存。

    参数:
    tif_path: 主影像文件路径。
    shp_path: 矢量文件路径。
    output_path: 输出文件的保存路径。
    ii: 输出文件的起始编号。
    output_size: 输出影像块的大小（像素），默认512。
    split_size: 分割后小影像块的大小（像素），默认21。
    """
    # 打开主影像文件
    main_raster_ds = gdal.Open(tif_path, gdal.GA_ReadOnly)
    vector_ds = ogr.Open(shp_path)
    layer = vector_ds.GetLayer()
    gt = main_raster_ds.GetGeoTransform()

    num_main_bands = main_raster_ds.RasterCount

    patch_id_start = (ii - 1) * ((output_size/split_size)**2) + 1
    patch_id = int(patch_id_start)  # 初始化patch编号

    for feature in layer:
        geom = feature.GetGeometryRef()
        minx, maxx, miny, maxy = geom.GetEnvelope()
        xoff = int((minx - gt[0]) / gt[1])
        yoff = int((maxy - gt[3]) / gt[5])
        
        # 读取并合并影像数据
        main_data = np.zeros((num_main_bands, output_size, output_size), dtype=np.float32)
        for band in range(1, num_main_bands + 1):
            if band <= main_raster_ds.RasterCount:
                data = main_raster_ds.GetRasterBand(band).ReadAsArray(xoff, yoff, output_size, output_size)
            
            main_data[band-1, :, :] = data

        # 分割大patch为多个小patch并保存
        num_x_splits = output_size // split_size
        num_y_splits = output_size // split_size
        for y in range(num_y_splits):
            for x in range(num_x_splits):
                split_patch = main_data[:, y*split_size:(y+1)*split_size, x*split_size:(x+1)*split_size]
                np.save(os.path.join(output_path, f"patch_{patch_id}.npy"), split_patch)
                patch_id += 1

    # 关闭数据集
    main_raster_ds = None
    vector_ds = None


"""4 lmdb dataset"""
def extract_geochem_and_split_patches_to_lmdb(tif_path, shp_path, lmdb_path, ii, output_size=504, split_size=21):
    env = lmdb.open(lmdb_path, map_size=int(1e9))
    
    main_raster_ds = gdal.Open(tif_path, gdal.GA_ReadOnly)
    vector_ds = ogr.Open(shp_path)
    layer = vector_ds.GetLayer()
    gt = main_raster_ds.GetGeoTransform()

    num_main_bands = main_raster_ds.RasterCount

    patch_id_start = (ii - 1) * ((output_size / split_size) ** 2) + 1
    patch_id = int(patch_id_start)

    with env.begin(write=True) as txn:
        for feature in layer:
            geom = feature.GetGeometryRef()
            minx, maxx, miny, maxy = geom.GetEnvelope()
            xoff = int((minx - gt[0]) / gt[1])
            yoff = int((maxy - gt[3]) / gt[5])

            main_data = np.zeros((num_main_bands, output_size, output_size), dtype=np.float32)
            for band in range(1, num_main_bands + 1):
                data = main_raster_ds.GetRasterBand(band).ReadAsArray(xoff, yoff, output_size, output_size)
                main_data[band-1, :, :] = data

            num_x_splits = output_size // split_size
            num_y_splits = output_size // split_size
            for y in range(num_y_splits):
                for x in range(num_x_splits):
                    split_patch = main_data[:, y*split_size:(y+1)*split_size, x*split_size:(x+1)*split_size]
                    txn.put(f"{patch_id}".encode(), np.ascontiguousarray(split_patch))
                    patch_id += 1
    
    # 关闭数据集
    main_raster_ds = None
    vector_ds = None


def extract_merge_and_split_patches_to_lmdb(tif_path, shp_path, lmdb_path, ii, output_size=504, split_size=21):
    # 打开主影像文件
    env = lmdb.open(lmdb_path, map_size=int(6e9))
    
    main_raster_ds = gdal.Open(tif_path, gdal.GA_ReadOnly)
    vector_ds = ogr.Open(shp_path)
    layer = vector_ds.GetLayer()
    gt = main_raster_ds.GetGeoTransform()

    # 准备辅助影像数据
    aux_data = [gdal.Open(path, gdal.GA_ReadOnly) for path in aux_image_paths]
    num_main_bands = main_raster_ds.RasterCount + len(aux_data)  # 更新带有辅助数据的波段计数

    patch_id_start = (ii - 1) * ((output_size/split_size)**2) + 1
    patch_id = int(patch_id_start)  # 初始化patch编号

    with env.begin(write=True) as txn:
        for feature in layer:
            geom = feature.GetGeometryRef()
            minx, maxx, miny, maxy = geom.GetEnvelope()
            xoff = int((minx - gt[0]) / gt[1])
            yoff = int((maxy - gt[3]) / gt[5])
            
            # 读取并合并影像数据
            main_data = np.zeros((num_main_bands, output_size, output_size), dtype=np.float32)
            for band in range(1, num_main_bands + 1):
                if band <= main_raster_ds.RasterCount:
                    data = main_raster_ds.GetRasterBand(band).ReadAsArray(xoff, yoff, output_size, output_size)
                else:
                    aux_ds = aux_data[band - main_raster_ds.RasterCount - 1]
                    data = aux_ds.GetRasterBand(1).ReadAsArray(xoff, yoff, output_size, output_size)
                main_data[band-1, :, :] = data

            # 分割大patch为多个小patch并保存
            num_x_splits = output_size // split_size
            num_y_splits = output_size // split_size
            for y in range(num_y_splits):
                for x in range(num_x_splits):
                    split_patch = main_data[:, y*split_size:(y+1)*split_size, x*split_size:(x+1)*split_size]
                    txn.put(f"{patch_id}".encode(), np.ascontiguousarray(split_patch))
                    patch_id += 1

    # 关闭数据集
    main_raster_ds = None
    vector_ds = None
    for ds in aux_data:
        ds = None


if __name__=="__main__":
    flag=7
    imgsize = 50
    patchsize = 500
    label = "Al2O3"
    labels = ["Al2O3", "Fe2O3", "K2O", "MgO", "Na2O", "SiO2"]
    
    if flag == 1:
        shp_path = "D:\\MyProject\\ETS_data\\shapefile\\1.shp"
        tif_path = "D:\\MyProject\\ETS_data\\TIFF\\1dts.tif"
        
        extract_patche(tif_path, shp_path, output_size=512)
        
    if flag == 2:
        shp_path = "D:\\MyProject\\ETS_data\\shapefile"
        shp_paths = glob.glob(os.path.join(shp_path, "*.shp"))
        # shp_paths = sorted(glob.glob(os.path.join(shp_path, "*.shp")))
        tif_path = "D:\\MyProject\\ETS_data\\TIFF\\1dts.tif"
        aux_image_paths = ["D:\\MyProject\\ETS_data\\TIFF\\2ndvi.tif", "D:\\MyProject\\ETS_data\\TIFF\\3aeromagnetism.tif", "D:\\MyProject\\ETS_data\\TIFF\\4dem_15m.tif"]
        
        output_path = "D:\\MyProject\\ETS_data\\trian\\img"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        for ii, path in tqdm(enumerate(shp_paths), total=len(shp_paths)):
            path = os.path.join(shp_path, f"{ii+1}.shp")
            extract_and_merge_patches(tif_path, path, aux_image_paths, output_path, ii+1, output_size=512)

    if flag == 3:
        shp_path = "D:\\MyProject\\ETS_data\\shapefile"
        shp_paths = glob.glob(os.path.join(shp_path, "*.shp"))
        
        label = "Al2O3"
        tif_path = "D:\\MyProject\\ETS_data\\Geochemistry_TIFF"
        tif_path = os.path.join(tif_path, f"{label}.tif")
        
        output_path = "D:\\MyProject\\ETS_data\\trian\\geo"
        output_path = os.path.join(output_path, label)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        for ii, path in tqdm(enumerate(shp_paths), total=len(shp_paths)):
            path = os.path.join(shp_path, f"{ii+1}.shp")
            extract_geochem_patches(tif_path, path, output_path, ii+1, output_size=512)
    
    if flag == 4:
        shp_path = "D:\\MyProject\\ETS_data\\shapefile"
        shp_paths = glob.glob(os.path.join(shp_path, "*.shp"))
        
        tif_path = "D:\\MyProject\\ETS_data\\TIFF\\1dts.tif"
        aux_image_paths = ["D:\\MyProject\\ETS_data\\TIFF\\2ndvi.tif", "D:\\MyProject\\ETS_data\\TIFF\\3aeromagnetism.tif", "D:\\MyProject\\ETS_data\\TIFF\\4dem_15m.tif"]
        
        output_path = "D:\\MyProject\\ETS_data\\trian\\img\\" + str(imgsize)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        for ii, path in tqdm(enumerate(shp_paths), total=len(shp_paths)):
            path = os.path.join(shp_path, f"{ii+1}.shp")
            extract_merge_and_split_patches(tif_path, path, aux_image_paths, output_path, ii+1, output_size=patchsize, split_size=imgsize)
    
    if flag == 5:
        shp_path = "D:\\MyProject\\ETS_data\\shapefile"
        shp_paths = glob.glob(os.path.join(shp_path, "*.shp"))
        
        
        tif_path = "D:\\MyProject\\ETS_data\\Geochemistry_TIFF"
        tif_path = os.path.join(tif_path, f"{label}.tif")
        
        output_path = "D:\\MyProject\\ETS_data\\trian\\geo\\" + str(imgsize)
        output_path = os.path.join(output_path, label)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        for ii, path in tqdm(enumerate(shp_paths), total=len(shp_paths)):
            path = os.path.join(shp_path, f"{ii+1}.shp")
            extract_geochem_and_split_patches(tif_path, path, output_path, ii+1, output_size=patchsize, split_size=imgsize)
            
    if flag == 6:
        shp_path = "D:\\MyProject\\ETS_data\\shapefile"
        shp_paths = glob.glob(os.path.join(shp_path, "*.shp"))
        
        tif_path = "D:\\MyProject\\ETS_data\\TIFF\\1dts.tif"
        aux_image_paths = ["D:\\MyProject\\ETS_data\\TIFF\\2ndvi.tif", "D:\\MyProject\\ETS_data\\TIFF\\3aeromagnetism.tif", "D:\\MyProject\\ETS_data\\TIFF\\4dem_15m.tif"]
        
        output_path = "D:\\MyProject\\ETS_data\\trian\\imglmdb\\" + str(imgsize)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        for ii, path in tqdm(enumerate(shp_paths), total=len(shp_paths)):
            path = os.path.join(shp_path, f"{ii+1}.shp")
            extract_merge_and_split_patches_to_lmdb(tif_path, path, output_path, ii+1, output_size=patchsize, split_size=imgsize)
    
    if flag == 7:
        shp_path = "D:\\MyProject\\ETS_data\\shapefile"
        shp_paths = glob.glob(os.path.join(shp_path, "*.shp"))
        
        for label in labels:
            tif_path = "D:\\MyProject\\ETS_data\\Geochemistry_TIFF"
            tif_path = os.path.join(tif_path, f"{label}.tif")
            
            output_path = "D:\\MyProject\\ETS_data\\trian\\geolmdb\\" + str(imgsize)
            output_path = os.path.join(output_path, label)
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            
            for ii, path in tqdm(enumerate(shp_paths), total=len(shp_paths)):
                path = os.path.join(shp_path, f"{ii+1}.shp")
                extract_geochem_and_split_patches_to_lmdb(tif_path, path, output_path, ii+1, output_size=patchsize, split_size=imgsize)