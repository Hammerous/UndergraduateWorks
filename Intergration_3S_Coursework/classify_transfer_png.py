import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# 定义类别和对应的颜色
color_map = {
    'algae': [0, 128, 0],    # 浅绿色
    'building': [255, 0, 0],     # 红色
    'water': [0, 0, 255],        # 蓝色
    'bareland': [255, 255, 0],   # 黄色
    'plant': [0, 255, 0]         # 暗绿色
}

# 反转颜色映射以从RGB值获取类别
reverse_color_map = {tuple(v): k for k, v in color_map.items()}

path1 = 'result//2007.png'
path2 = 'result//2024.png'

# 加载图片
image1 = Image.open(path1)
image2 = Image.open(path2)

# 将图片转换为numpy数组
array1 = np.array(image1)[:,:,:3]
array2 = np.array(image2)[:,:,:3]

# 检查图片的形状是否一致
if array1.shape != array2.shape:
    raise ValueError("两张图片的形状不一致。")

# 初始化转移矩阵和类别计数
categories = list(color_map.keys())
transition_matrix = np.zeros((len(categories), len(categories)), dtype=int)
category_count_1 = np.zeros(len(categories), dtype=int)
category_count_2 = np.zeros(len(categories), dtype=int)

# 初始化变化图像数组，并将[0,0,0]对应的区域应用到change_images中
change_images = {category: np.where((array1 == [0, 0, 0]).all(axis=-1)[:, :, None], [0, 0, 0], [105, 105, 105]).astype(np.uint8) for category in categories}

# 遍历每个像素并更新转移矩阵和类别计数
for i in range(array1.shape[0]):
    for j in range(array1.shape[1]):
        pixel1 = tuple(array1[i, j])
        pixel2 = tuple(array2[i, j])
        if pixel1 == (0, 0, 0) or pixel2 == (0, 0, 0):
            continue
        category1 = reverse_color_map.get(pixel1)
        category2 = reverse_color_map.get(pixel2)
        if category1 and category2:
            transition_matrix[categories.index(category2), categories.index(category1)] += 1
            category_count_1[categories.index(category1)] += 1
            category_count_2[categories.index(category2)] += 1
            
            if category1 == category2:
                change_images[category1][i, j] = [0, 255, 0]  # 未变化的区域，绿色
            else:
                change_images[category2][i, j] = [255, 0, 0]  # 原先并非该分类，变化后成为该分类的区域，红色
                change_images[category1][i, j] = [0, 0, 255]  # 原先为该分类，变化后不再是该分类的区域，浅蓝色

# 保存变化图像
for category in categories:
    Image.fromarray(change_images[category]).save(f"{category}_change-{path1.split('//')[-1].split('.')[0]}-{path2.split('//')[-1].split('.')[0]}.png")

print("变化图像已保存。")

# 绘制转移矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(transition_matrix, annot=True, xticklabels=categories, yticklabels=categories)
plt.xlabel(f"{path1.split('//')[-1].split('.')[0]}")
plt.ylabel(f"{path2.split('//')[-1].split('.')[0]}")
plt.title('Land Use Transition Matrix')
plt.savefig(f"LUTMat-{path1.split('//')[-1].split('.')[0]}-{path2.split('//')[-1].split('.')[0]}.jpg",dpi=300)

import pandas as pd
# 将转移矩阵连同其标签保存到csv中
df_transition_matrix = pd.DataFrame(transition_matrix, index=categories, columns=categories)
df_transition_matrix.to_csv(f"T_mat-{path1.split('//')[-1].split('.')[0]}-{path2.split('//')[-1].split('.')[0]}.csv", index=True, encoding='utf-8-sig')
print("Transition matrix has been saved to 'transition_matrix.csv'.")

# 绘制类别计数的饼状图
plt.figure(figsize=(8, 8))
plt.pie(category_count_1, labels=categories, autopct='%1.1f%%', colors=[np.array(color_map[cat])/255 for cat in categories])
plt.title(f"Category Distribution in {path1.split('//')[-1].split('.')[0]}")
plt.savefig(f"Pie_{path1.split('//')[-1].split('.')[0]}.jpg",dpi=300)

# 绘制类别计数的饼状图
plt.figure(figsize=(8, 8))
plt.pie(category_count_2, labels=categories, autopct='%1.1f%%', colors=[np.array(color_map[cat])/255 for cat in categories])
plt.title(f"Category Distribution in {path2.split('//')[-1].split('.')[0]}")
plt.savefig(f"Pie_{path2.split('//')[-1].split('.')[0]}_Pie.jpg",dpi=300)