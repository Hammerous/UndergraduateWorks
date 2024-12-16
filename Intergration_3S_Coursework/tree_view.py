from joblib import load

# 加载保存的决策树模型
model = load("decision_tree_model.joblib")

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# 检查模型类型
if isinstance(model, (DecisionTreeClassifier, DecisionTreeRegressor)):
    print("Loaded model is a Decision Tree.")
else:
    print("The loaded model is not a Decision Tree.")

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# 绘制决策树
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=["NDWI", "NDBI", "TWI", "NDVI", "IBI", "BSI", "CMI", "FAI"]
          , class_names=['algae', 'building', 'water', 'bareland', 'plant'], filled=True)
plt.savefig('tree_view.jpg', dpi=300)
