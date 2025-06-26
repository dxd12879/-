import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 資料路徑（請根據你的實際檔案路徑修改）
file_path = r"C:\Users\dxd12\OneDrive\桌面\pythom\predictive_maintenance.csv"

# 嘗試不同編碼讀取
encodings = ['utf-8', 'big5', 'cp950']
for enc in encodings:
    try:
        df = pd.read_csv(file_path, encoding=enc)
        print(f"成功使用編碼 {enc} 讀取檔案")
        break
    except UnicodeDecodeError:
        continue
else:
    print("檔案讀取失敗，請確認檔案編碼是否正確。")
    exit()

# ----------- 圖表呈現 -----------

# 顯示基本資料摘要（仍以文字形式呈現）
print("\n資料摘要如下：")
print(df.info())

# 1. 故障類型分布圖
if 'Failure Type' in df.columns:
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df, x='Failure Type', order=df['Failure Type'].value_counts().index, palette="Set2")
    plt.title("各類故障類型數量分布", fontsize=14)
    plt.xlabel("故障類型")
    plt.ylabel("數量")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("找不到『Failure Type』欄位，請確認欄位名稱是否正確。")

# 2. 缺失值視覺化
missing = df.isnull().sum()
missing = missing[missing > 0]
if not missing.empty:
    plt.figure(figsize=(8, 6))
    missing.plot(kind='bar', color='salmon')
    plt.title("各欄位缺失值統計", fontsize=14)
    plt.ylabel("缺失值數量")
    plt.xlabel("欄位")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("\n沒有缺失值")

# 3. 數值型欄位分布圖
numeric_cols = df.select_dtypes(include='number').columns
if not numeric_cols.empty:
    df[numeric_cols].hist(figsize=(12, 8), bins=30, color='skyblue', edgecolor='black')
    plt.suptitle("數值欄位分布", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
else:
    print("資料中沒有數值型欄位")
