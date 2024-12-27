from osgeo import gdal
from PIL import Image
import numpy as np

# 打开GeoTIFF文件
tiff_file = 'colored_predicted_image_2022_resample.tif'
dataset = gdal.Open(tiff_file)

# 读取数据
bands = [dataset.GetRasterBand(i+1).ReadAsArray() for i in range(3)]
array = np.dstack(bands)

# 将数据转换为8位（0-255）范围
array = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)

# 创建PIL图像
image = Image.fromarray(array, 'RGB')

# 保存为PNG文件
png_file = 'colored_predicted_image_2022_resample.png'
image.save(png_file)

print(f"GeoTIFF文件已转换为PNG格式并保存为 {png_file}")
