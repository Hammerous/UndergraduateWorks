import pandas as pd
import csv,os
os.chdir(os.path.dirname(__file__))

# 定义一个函数，接受一个文件夹路径和一个字典作为参数
def in_dict_and_folder(folder_path, img_dict):
  # 导入os模块，用于操作文件系统
  import os
  # 创建一个空列表，用于存储在文件夹中的文件名
  in_folder = []
  # 遍历字典的键
  for item in img_dict.keys():
    # 获取完整的路径
    item_path = os.path.join(folder_path, item+'.jpg')
    # 判断文件名是否在文件夹内
    if os.path.exists(item_path):
      # 如果在，将其添加到列表中
      in_folder.append(item)
  # 返回列表
  return in_folder


# 定义一个函数，接受一个文件夹路径和一个字典作为参数
def not_in_dict_and_folder(folder_path, img_dict):
  # 导入os模块，用于操作文件系统
  import os
  # 创建一个空字典，用于存储不在文件夹中的文件名和对应的值
  not_in_folder = {}
  # 遍历字典的键
  for item in img_dict.keys():
    # 获取完整的路径
    item_path = os.path.join(folder_path, item+'.jpg')
    # 判断文件名是否在文件夹内
    if not os.path.exists(item_path):
      # 如果不在，将其作为键，将字典中对应的值作为值，添加到字典中
      not_in_folder[item] = img_dict[item]
  # 返回字典
  return not_in_folder

def CraterCSV_Extraction(path):
    with open(path, newline='') as f:
        Crater_Info = csv.DictReader(f)
        return [*Crater_Info]
    
def CraterCSV_Saving(CraterCSV_Dict,csv_path):
    df = pd.DataFrame(CraterCSV_Dict)
    df.to_csv(csv_path, index=False, header=True)
    return CraterCSV_Dict

def UpperCSV_Extraction(path):
    Dict = {}
    with open(path, newline='') as f:
        _Info = csv.DictReader(f)
        _Info = [*_Info]
    if(len(_Info)>0):
        main_key = list(_Info[0].keys())[0]
        for each in _Info:
            itemID = each[main_key]
            each.pop(main_key)
            Dict.update({itemID:each})
        return Dict,main_key
    else:
        return False,False

def UpperCSV_Saving(_Dict,mainkey,csv_path):
    Final_dict = []
    for each in _Dict:
        mainkey_dict = {mainkey:each}
        Final_dict.append(dict(mainkey_dict,**_Dict[each]))
    return CraterCSV_Saving(Final_dict,csv_path)

path = 'Processed'

for folder in os.listdir(path):
   folder_path = os.path.join(path, folder)
   if(os.path.splitext(folder_path)[1]==''):
    csv_path = os.path.join(folder_path,'data.csv')
    img_dict,main_key = UpperCSV_Extraction(csv_path)
    if(img_dict):
        not_found_img = not_in_dict_and_folder(folder_path,img_dict)
        #print(not_found_img)
        LargeIMG = in_dict_and_folder(os.path.join(folder_path,'LargeIMG'),not_found_img)
        #print(LargeIMG)
        for each in LargeIMG:
           img_dict[each]['is_AUTOvalidated'] = '1'
           img_dict[each]['is_MANUALvalidated'] = '1'
        UpperCSV_Saving(img_dict,main_key,csv_path)
    print('{0} is done'.format(folder))