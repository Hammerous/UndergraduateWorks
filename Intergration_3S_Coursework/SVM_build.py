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

from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score, accuracy_score
from sklearn import svm
import matplotlib.pyplot as plt
import joblib

# 将数据集分为特征和目标变量
X = data_frames[fields_to_keep]
y = data_frames['category']

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 创建SVM分类器
clf = svm.SVC(cache_size=1000, verbose=True, kernel='rbf')

# 进行十折交叉验证
kf = KFold(n_splits=10, shuffle=True)
cv_scores = cross_val_score(clf, X, y, cv=kf, n_jobs=-1)
print(cv_scores)

# 训练分类器
clf.fit(X_train, y_train)
# 将训练好的模型保存到文件
joblib.dump(clf, 'svm_model.pkl')
print("模型已训练并保存为 svm_model.pkl")

# 在测试数据上进行预测
y_pred = clf.predict(X_test)

# 绘制混淆矩阵
labels=['algae', 'water', 'plant',  'building', 'bareland']
cm = confusion_matrix(y_test, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.savefig('confusion_mat_SVM.png', dpi=300)

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