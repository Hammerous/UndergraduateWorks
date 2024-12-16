import pandas as pd
import os
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

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score, accuracy_score
import matplotlib.pyplot as plt
import joblib

# 将数据集分为特征和目标变量
X = data_frames[fields_to_keep]
y = data_frames['category']

# 初始化具有预剪枝参数的决策树分类器
clf = DecisionTreeClassifier(max_depth=5, criterion='entropy',
                            class_weight={'algae': 1, 'building': 1, 'water': 1, 'bareland': 1, 'plant': 1},
                            splitter='best', ccp_alpha=0, min_samples_leaf= max(X.shape[0]//1000, 1))

# # 进行十折交叉验证
# kf = KFold(n_splits=10, shuffle=True)
# cv_scores = cross_val_score(clf, X, y, cv=kf, n_jobs=-1)
# print(cv_scores)
# print(min(cv_scores))

# 使用重复的交叉验证
rkf = RepeatedStratifiedKFold(n_splits=4, n_repeats=20)
best_score = 0
best_model = None

for train_index, test_index in rkf.split(X.values, y):
    X_train, X_test = X.values[train_index], X.values[test_index]
    y_train, y_test = y[train_index], y[test_index]
    clf.fit(X_train, y_train)
    # score = clf.score(X_test, y_test)
    score = cohen_kappa_score(y_test, clf.predict(X_test))      # 在测试数据上进行预测, 计算kappa系数
    print(f"决策树模型的kappa系数是: {score}")
    if score > best_score:
        best_score = score
        best_model = joblib.load('tree_best_model.pkl') if best_model else clf
        joblib.dump(best_model, 'tree_best_model.pkl')

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X.values, y, test_size=0.2)
# 在训练数据上拟合模型
clf.fit(X_train, y_train)
score = cohen_kappa_score(y_test, clf.predict(X_test))
# score = clf.score(X_test, y_test)
if score > best_score:
    best_model = joblib.load('tree_best_model.pkl') if best_model else clf
print(f"Best model saved with score: {best_score}")

# 在测试数据上进行预测
y_pred = best_model.predict(X_test)

# # 计算kappa系数
# kappa = cohen_kappa_score(y_test, y_pred)
# print(f"决策树模型的kappa系数是: {kappa}")

# 绘制混淆矩阵
labels=['algae', 'water', 'plant',  'building', 'bareland']
cm = confusion_matrix(y_test, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.savefig('confusion_mat.png', dpi=300)

# 计算总体分类精度
overall_accuracy = accuracy_score(y_test, y_pred)
# 计算Kappa系数
kappa_coefficient = cohen_kappa_score(y_test, y_pred, labels=labels)
# 计算错分误差（Commission）和漏分误差（Omission）
commission = {}
omission = {}
prod_acc = {}
user_acc = {}

for i, label in enumerate(labels):
    tp = cm[i, i]
    fn = cm[i, :].sum() - tp
    fp = cm[:, i].sum() - tp
    tn = cm.sum() - (tp + fn + fp)
    
    commission[label] = fp / (tp + fp) if (tp + fp) != 0 else 0
    omission[label] = fn / (tp + fn) if (tp + fn) != 0 else 0
    prod_acc[label] = tp / (tp + fn) if (tp + fn) != 0 else 0
    user_acc[label] = tp / (tp + fp) if (tp + fp) != 0 else 0

# 打印结果
print(f"Overall Accuracy: {overall_accuracy}")
print(f"Kappa Coefficient: {kappa_coefficient}")
print("Commission (False Positive Rate):")
for label in labels:
    print(f"  {label}: {commission[label]}")
print("Omission (False Negative Rate):")
for label in labels:
    print(f"  {label}: {omission[label]}")
print("Prod.Acc:")
for label in labels:
    print(f"  {label}: {prod_acc[label]}")
print("User.Acc:")
for label in labels:
    print(f"  {label}: {user_acc[label]}")

# 保存决策树模型
# joblib.dump(clf, 'decision_tree_model.joblib')

# 绘制并保存决策树
plt.figure(figsize=(20,10))
plot_tree(clf, filled=True, feature_names=fields_to_keep, class_names=['algae', 'building', 'water', 'bareland', 'plant'])
plt.savefig('decision_tree.png', dpi=300)