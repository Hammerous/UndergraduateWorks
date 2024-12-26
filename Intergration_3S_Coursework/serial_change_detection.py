import os
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from scipy.ndimage import convolve
from PIL import Image

# 定义类别和对应的颜色
color_map = {
    'algae': [0, 128, 0],    
    'building': [255, 0, 0],   
    'water': [0, 0, 255], 
    'bareland': [255, 255, 0], 
    'plant': [0, 255, 0],
    'nodata': [0, 0, 0]
}

# 创建颜色到类别的映射
color_to_category = {tuple(v): k for k, v in color_map.items()}

# 统计变化

def analyze_pixel_changes(image_folder, output_file, pixel_area=900/1e6):
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')])
    
    if len(image_files) < 2:
        raise ValueError("需要至少两张图片进行分析")

    pixel_changes = []
    years = [os.path.splitext(f)[0] for f in image_files]

    # 加载图片并转换为类别矩阵
    images = []
    for file in image_files:
        img = Image.open(os.path.join(image_folder, file)).convert('RGB')
        img_array = np.array(img)
        categories = np.zeros(img_array.shape[:2], dtype=object)
        
        for color, category in color_to_category.items():
            mask = np.all(img_array == color, axis=-1)
            categories[mask] = category
        
        images.append(categories)

    # 统计每个像素的变化路径
    for i in range(images[0].shape[0]):
        for j in range(images[0].shape[1]):
            change_path = tuple(image[i, j] for image in images)
            if 'nodata' not in change_path:  # 忽略nodata区域
                pixel_changes.append(change_path)

    # 统计变化路径的次数和面积
    change_counter = Counter(pixel_changes)
    changes_data = []
    for change_path, count in change_counter.items():
        area = count * pixel_area
        changes_data.append({**{f"Year {year}": category for year, category in zip(years, change_path)}, "Count": count, "Area (km^2)": area})

    df_changes = pd.DataFrame(changes_data)
    df_changes.to_csv(output_file, index=False)
    return df_changes

# 类型频数统计

def frequency_analysis(image_folder, output_file):
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')])

    frequency_data = []
    for file in image_files:
        img = Image.open(os.path.join(image_folder, file)).convert('RGB')
        img_array = np.array(img)
        categories = np.zeros(img_array.shape[:2], dtype=object)

        for color, category in color_to_category.items():
            mask = np.all(img_array == color, axis=-1)
            categories[mask] = category

        # 统计频数
        category_counts = Counter(categories.flatten())
        category_counts.pop('nodata', None)  # 去掉 nodata 类型
        category_counts['Year'] = os.path.splitext(file)[0]
        frequency_data.append(category_counts)

    df_frequency = pd.DataFrame(frequency_data).fillna(0)
    df_frequency.to_excel(output_file, index=False)
    return df_frequency

# 空间趋势分析
# def analyze_spatial_trends(images, output_file):
#     kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])  # 用于卷积的邻域核
    
#     spatial_correlation = []
#     for t in range(len(images) - 1):
#         current_image = images[t]
#         next_image = images[t + 1]

#         # 遍历所有非nodata类型
#         for target_category in color_map.keys():
#             if target_category == 'nodata':
#                 continue

#             target_mask = (current_image == target_category)
#             influence_mask = convolve(target_mask.astype(int), kernel, mode='constant') > 0

#             for transition_category in color_map.keys():
#                 if transition_category == 'nodata':
#                     continue

#                 transition_mask = (current_image == transition_category) & influence_mask
#                 changed_pixels = np.sum((next_image == transition_category) & transition_mask)

#                 spatial_correlation.append({
#                     'Time Step': f"{t} to {t + 1}",
#                     'Target Category': target_category,
#                     'Transition Category': transition_category,
#                     'Changed Pixels': changed_pixels
#                 })

#     df_spatial = pd.DataFrame(spatial_correlation)
#     df_spatial.to_csv(output_file, index=False)
#     return df_spatial

# 示例用法
if __name__ == "__main__":
    image_folder = "result"

    # 像素变化分析
    changes_output_file = "pixel_changes.csv"
    change_df = analyze_pixel_changes(image_folder, changes_output_file)
    print(f"像素变化统计已保存到 {changes_output_file}")

    # 类型频数统计
    frequency_output_file = "frequency_analysis.xlsx"
    frequency_df = frequency_analysis(image_folder, frequency_output_file)
    print(f"频数统计已保存到 {frequency_output_file}")

    # # 空间趋势分析
    # images = []
    # for file in sorted(os.listdir(image_folder)):
    #     img = Image.open(os.path.join(image_folder, file)).convert('RGB')
    #     img_array = np.array(img)
    #     categories = np.zeros(img_array.shape[:2], dtype=object)

    #     for color, category in color_to_category.items():
    #         mask = np.all(img_array == color, axis=-1)
    #         categories[mask] = category

    #     images.append(categories)

    # trends_output_file = "spatial_trends.csv"
    # trends_df = analyze_spatial_trends(images, trends_output_file)
    # print(f"空间趋势分析已保存到 {trends_output_file}")
