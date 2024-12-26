import numpy as np
from osgeo import gdal
import spectral

# 定义类别和对应的颜色
color_map = {
    'building': [255, 0, 0],     # 红色
    'plant': [0, 255, 0],        # 暗绿色
    'water': [0, 0, 255],        # 蓝色
    'bareland': [255, 255, 0],   # 黄色
    'algae': [0, 128, 0]         # 浅绿色
}

# 定义颜色和类别的映射关系
color_to_class = {
    (0, 0, 0): 0,
    (255, 0, 0): 1,
    (0, 255, 0): 2,
    (0, 0, 255): 3,
    (255, 255, 0): 4,
    (0, 128, 0): 5
}

# 打开分类后的TIF图像
dataset = gdal.Open('colored_predicted_image_2013.tif')
image_data = dataset.ReadAsArray().transpose(1, 2, 0)

# 创建一个空的分类图像
classification_image = np.zeros((image_data.shape[0], image_data.shape[1]), dtype=np.int32)

# 将颜色转换为类别标号
for color, class_id in color_to_class.items():
    mask = np.all(image_data == color, axis=-1)
    classification_image[mask] = class_id

# 获取地理信息
geo_transform = dataset.GetGeoTransform()
projection = dataset.GetProjection()

# 保存为ENVI格式的影像文件，并保留地理信息
spectral.envi.save_image('2013.hdr', classification_image, dtype=np.int32, metadata={'geo_transform': geo_transform, 'projection': projection})

print("分类后的TIF图像已成功转换为ENVI格式的影像文件，并保留了原始的地理信息。")