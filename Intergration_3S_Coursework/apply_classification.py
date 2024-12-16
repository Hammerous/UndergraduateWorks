import pandas as pd
import numpy as np
import joblib
import spectral

year_num = '2024'

fields_to_keep = ["NDWI", "NDBI", "TWI", "NDVI", "IBI", "BSI", "CMI", "FAI"]

# 加载保存的决策树模型
# clf = joblib.load('decision_tree_model.joblib')
clf = joblib.load('tree_best_model.pkl')

# 加载图片并转换为DataFrame（假设图片已经转换为特征向量）
envi_file_path = r'products//{0}//{0}'.format(year_num)

def find_indices(long_list, short_list):
    indices = []
    for element in short_list:
        if element in long_list:
            indices.append(long_list.index(element))
    return indices

# 读取ENVI文件头部信息
envi_image = spectral.envi.open(envi_file_path + '.hdr', envi_file_path)
# 获取波段名称
band_names = envi_image.metadata.get('band names', [])
# 加载数据
envi_image = envi_image.load()
# 显示数据的形状
print(f'数据的形状:{envi_image.shape}; 波段名称： {band_names}')
indice_idx = find_indices(band_names, fields_to_keep)
# 假设图片数据已经转换为DataFrame格式
image_data = envi_image[:,:,indice_idx]
# 找到所有包含NaN的行索引
nan_mask = np.any(np.isnan(image_data), axis=2)
# 找到所有全为0的行索引
zero_mask = np.all(image_data == 0, axis=2)
# 合并两个掩码
combined_mask = np.logical_or(nan_mask, zero_mask)
# 将包含nan的行全行转换为nan
image_data[combined_mask, :] = np.nan

# 将数据重塑为 (n*m, k) 的二维数组
data_reshaped = image_data.reshape(-1, len(fields_to_keep))
# 使用加载的决策树模型进行预测
data_reshaped = clf.predict(data_reshaped)
# 将预测结果重塑为 (n, m, 1) 的形状
predicted_image = data_reshaped.reshape(envi_image.shape[0], envi_image.shape[1])
predicted_image[combined_mask] = np.nan

# 定义类别和对应的颜色
color_map = {
    'algae': [144, 238, 144],    # 浅绿色
    'building': [255, 0, 0],     # 红色
    'water': [0, 0, 255],        # 蓝色
    'bareland': [255, 255, 0],   # 黄色
    'plant': [0, 100, 0]         # 暗绿色
}

# 创建一个新的彩色图像用于保存上色后的结果
colored_image = np.zeros((predicted_image.shape[0], predicted_image.shape[1], 3), dtype=np.uint8)

# 根据预测结果进行上色
for label, color in color_map.items():
    colored_image[predicted_image == label] = color

from PIL import Image
# 保存上色后的图像为PNG文件
colored_image_pil = Image.fromarray(colored_image)
colored_image_pil.save(f'colored_predicted_image_{year_num}.png')

print("上色后的预测图像已保存为 'colored_predicted_image.png' 文件。")