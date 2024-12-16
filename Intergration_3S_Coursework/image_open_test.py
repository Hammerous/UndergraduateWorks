import spectral

# 定义ENVI文件路径（无后缀）
envi_file_path = r'products\2023\2023'

# 读取ENVI文件头部信息
envi_image = spectral.envi.open(envi_file_path + '.hdr', envi_file_path)

# 加载数据
data = envi_image.load()

# 显示数据的形状
print('数据的形状:', data.shape)

# 获取波段名称
band_names = envi_image.metadata.get('band names', [])

# 打印所有波段名称
print(band_names)
for i, band_name in enumerate(band_names):
    print(f'波段 {i+1}: {band_name}')