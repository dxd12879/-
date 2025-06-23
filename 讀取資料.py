import pandas as pd

# 資料路徑（請根據你的電腦實際路徑修改）
file_path = r"C:\Users\dxd12\OneDrive\桌面\pythom\predictive_maintenance.csv"

# 讀取資料
try:
    df = pd.read_csv(file_path)
    print("檔案載入成功，前五筆資料如下：\n")
    print(df.head())
except FileNotFoundError:
    print("檔案找不到，請確認路徑是否正確。")

# 顯示資料基本資訊
print("\n資料摘要如下：")
print(df.info())

# 顯示資料欄位的唯一值（幫助了解分類型欄位）
print("\n故障類型（Failure Type）統計：")
print(df['Failure Type'].value_counts())

# 顯示是否有缺失值
print("\n缺失值統計：")
print(df.isnull().sum())
