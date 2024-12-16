import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import os, joblib
import spectral

root_folder = r'products'
# 指定需要保留的字段
fields_to_keep = ["NDWI", "NDBI", "TWI", "NDVI", "IBI", "BSI", "CMI", "FAI"]

def rename_columns(df, new_names, start_loc):
    # 获取当前列名
    current_columns = df.columns.tolist()
    # 确定需要修改的列数
    num_to_rename = min(len(current_columns) - start_loc, len(new_names))
    # 创建新的列名列表
    new_columns = current_columns[:start_loc] + new_names[:num_to_rename] + current_columns[start_loc + num_to_rename:]
    # 修改列名
    df.columns = new_columns
    return df

data_frames = []
# 遍历根文件夹
for folder_name, subfolders, filenames in os.walk(root_folder):
    # 检查当前文件夹中的每个文件
    for filename in filenames:
        # 如果文件是.csv文件，打印其名称
        if filename.endswith('.csv'):
            print(f"Found CSV file: {filename} in folder: {folder_name}")
            # 定义ENVI文件路径（无后缀）
            envi_file_path = os.path.join(folder_name, os.path.basename(folder_name))
            # 读取ENVI文件头部信息
            envi_image = spectral.envi.open(envi_file_path + '.hdr', envi_file_path)
            # 获取波段名称
            band_names = envi_image.metadata.get('band names', [])
            df = rename_columns(pd.read_csv(os.path.join(folder_name, filename)), band_names, 6)
            df['category'] = os.path.basename(filename).split(".")[0]
            data_frames.extend(df[fields_to_keep+['category']].values.tolist())

data_frames = pd.DataFrame(data_frames, columns=fields_to_keep+['category'])
# 假设目标变量在名为'target'的列中
X = data_frames[fields_to_keep]
y = data_frames['category']

# 初始化决策树分类器
clf = DecisionTreeClassifier(max_depth=3, class_weight={'algae': 1, 'building': 2, 'water': 2, 'bareland': 2, 'plant': 2})

# 拟合决策树模型
clf.fit(X, y)

# 进行10折交叉验证
scores = cross_val_score(clf, X, y, cv=10)

# 打印交叉验证得分
print("Cross-validation scores:", scores)
print("Mean cross-validation score:", scores.mean())

# 保存决策树模型到文件
joblib.dump(clf, 'decision_tree_model.joblib')
print("决策树模型已保存为 'decision_tree_model.joblib' 文件。")