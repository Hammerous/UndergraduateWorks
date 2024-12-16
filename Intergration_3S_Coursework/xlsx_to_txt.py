import pandas as pd

# 加载Excel文件
excel_file = r'products\2024\classification_sample_2024.xlsx'
xls = pd.ExcelFile(excel_file)

# 遍历每个工作表并保存为txt文件
for sheet_name in xls.sheet_names:
    df = pd.read_excel(excel_file, sheet_name=sheet_name, engine='openpyxl')
    df.drop(columns=['Point', ' Map X', ' Map Y'], inplace=True)
    txt_file = f"{sheet_name}.txt"
    df.to_csv(txt_file, sep='\t', index=True, encoding='ascii')

print("所有工作表已成功保存为txt文件。")