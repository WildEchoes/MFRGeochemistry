import os
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from math import sqrt
from tqdm import tqdm
from scipy.stats import pearsonr
import glob


def draw_points(output, label, savepth, name, maxv=None, minv=None):
    if maxv is None and minv is None:
        min_val = min(label)
        max_val = max(label)

        min_val = (min_val // 5) * 5
        max_val = math.ceil(max_val / 5) * 5

        if min_val == max_val:
            min_val = min_val - 5 if min_val - 5 > 0 else 0
            max_val = max_val + 5 if max_val + 5 < 100 else 100
        else:
            min_val = 0 if min_val < 0 else min_val
            max_val = 100 if max_val > 100 else max_val
    else:
        min_val = minv
        max_val = maxv

    # 数据量太大使用直方图计算密度值

    hist , xedges, yedges = np.histogram2d(label, output, bins=40, density=True)
    # 计算每个点的密度值
    xidx = np.searchsorted(xedges, label, side="right") - 1
    yidx = np.searchsorted(yedges, output, side="right") - 1

    # 确保索引不会越界，将100调整为99
    xidx = np.clip(xidx, 0, hist.shape[0]-1)
    yidx = np.clip(yidx, 0, hist.shape[1]-1)

    # 通过直方图密度查找，为每个点找到对应的密度值
    density = hist[xidx, yidx]

    fig, ax = plt.subplots()  # 创建了一个图和坐标轴对象
    
    ax.set_xlim(min_val, max_val)
    ax.set_ylim(min_val, max_val)
    
    plt.xlabel("Measured (%)", fontsize=12)
    plt.ylabel("Predicted (%)", fontsize=12)
    plt.plot([min_val, max_val], [min_val, max_val], "k--")

    scatter = ax.scatter(label, output, c=density, s=2, edgecolor=None, cmap="plasma")  # 创建了一个散点图，其中点的颜色和大小由z控制

    fig.colorbar(scatter, ax=ax, label="Density").set_ticks([])  # 创建了一个颜色条，显示了密度值的范围, 不显示刻度
    # plt.show()

    plt.savefig(os.path.join(savepth, name + ".png"))

    plt.close()

    return min_val, max_val


def train_linear(X_train, X_test, label_trian_list, label_test_list, label, method, savepath1):
    y_train = data_load(label_trian_list).ravel()
    y_test = data_load(label_test_list).ravel()

    # 创建和训练模型
    if method == "Linear":
        model = LinearRegression()
        savepath = savepath1 + "_linear" 
    elif method == "Ridge":
        model = Ridge(alpha=0.5)
        savepath = savepath1 + "_ridge"
    elif method == "Lasso":
        model = Lasso(alpha=0.25)
        savepath = savepath1 + "_lasso"

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    model.fit(X_train, y_train)

    # 获取截距项
    intercept = model.intercept_
    # print("截距项 (Intercept):", intercept)

    # 获取每个变量对应的参数
    coefficients = model.coef_
    # print("系数 (Coefficients):", coefficients)

    # 预测
    predictions_train = model.predict(X_train)
    predictions_test = model.predict(X_test)

    # 计算误差
    mse_train = mean_squared_error(y_train, predictions_train)
    mse_test = mean_squared_error(y_test, predictions_test)
    mae_train = mean_absolute_error(y_train, predictions_train)
    mae_test = mean_absolute_error(y_test, predictions_test)
    r_train = pearsonr(y_train.ravel(), predictions_train.ravel())[0]
    r_test = pearsonr(y_test.ravel(), predictions_test.ravel())[0]
    r2_train = r2_score(y_train, predictions_train)
    r2_test = r2_score(y_test, predictions_test)
    rmse_train = sqrt(mse_train)
    rmse_test = sqrt(mse_test)

    # 画图
    print(f"Drawing {label} scatter plot")
    min, max = draw_points(predictions_train, y_train, savepath, label + "_train")
    draw_points(predictions_test, y_test, savepath, label + "_test", maxv=max, minv=min)

    # 记录结果
    with open(f"{savepath}\\{label}_results.txt", "w", encoding="utf-8") as f:
        f.write(
            f"Train R2: {r2_train:.4f} R: {r_train:.4f} MAE: {mae_train:.4f} RMSE: {rmse_train:.4f} Test R2:{r2_test:.4f} R: {r_test:.4f} MAE: {mae_test:.4f} RMSE: {rmse_test:.4f}\n"
        )
        f.write(f"Intercept: {intercept}\n")
        f.write(f"Coefficients: {coefficients}\n")
        for i, coef in enumerate(coefficients):
            f.write(f"变量 {i+1} 的系数是 {coef}\n")


def data_load(data_list):
    data = []
    for datafile in data_list:
        img = np.load(datafile)
        img = np.transpose(img, (1, 2, 0))
        img = img.reshape(-1, img.shape[-1])
        data.append(img)
    
    data = np.concatenate(data, axis=0)

    return data

if __name__ == "__main__":
    datapath = "D:\\MyProject\\ETS_data\\trian\\img\\50"
    labelpath = "D:\\MyProject\\ETS_data\\trian\\geo\\50"

    assert os.path.exists(datapath), "Data file does not exist"

    savepath = "./compare_method/linear/result"

    labels = ["Al2O3", "Fe2O3", "K2O", "MgO", "Na2O", "SiO2"]
    # methods = ['Linear', 'Ridge', 'Lasso']
    methods = ["Linear"]
    
    data_list = glob.glob(os.path.join(datapath, "*.npy"))

    data_train_list, data_test_list = train_test_split(data_list, test_size=0.3, random_state=3)

    data_train = data_load(data_train_list)
    data_test  = data_load(data_test_list)

    for method in methods:
        for label in tqdm(labels, desc=f"Running {method}"):
            label_files = os.path.join(labelpath, label)
            label_list = glob.glob(os.path.join(label_files, "*.npy"))

            label_train_list, label_test_list = train_test_split(label_list, test_size=0.3, random_state=3)
            
            train_linear(data_train, data_test, label_train_list, label_test_list, label, method, savepath)
