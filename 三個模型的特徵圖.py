import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# 讀取資料
df = pd.read_csv("predictive_maintenance.csv")

# 基本前處理：移除非數值欄位
df = df.drop(columns=["UDI", "Product ID", "Failure Type"])
df["Type"] = pd.factorize(df["Type"])[0]

# 定義特徵與標籤
X = df.drop(columns=["Target"])
y = df["Target"]

# 分割訓練與測試資料
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 建立三個模型並訓練
models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)

# 特徵名稱
feature_names = X_train.columns

# 提取特徵重要性
importances = {
    "Decision Tree": models["Decision Tree"].feature_importances_,
    "Random Forest": models["Random Forest"].feature_importances_,
    "Gradient Boosting": models["Gradient Boosting"].feature_importances_
}

# 畫圖
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax, (name, importance) in zip(axes, importances.items()):
    sorted_idx = importance.argsort()[::-1]
    sorted_features = feature_names[sorted_idx]
    sorted_importance = importance[sorted_idx]

    sns.barplot(x=sorted_importance, y=sorted_features, ax=ax, palette="viridis")
    ax.set_title(f"{name} Feature Importance")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("Feature")

plt.tight_layout()
plt.show()
