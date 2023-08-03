import pandas as pd
import csv,os,numpy
os.chdir(os.path.dirname(__file__))

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
    if(os.path.isfile(path)):
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
    return False,False

def UpperCSV_Saving(_Dict,mainkey,csv_path):
    Final_dict = []
    for each in _Dict:
        mainkey_dict = {mainkey:each}
        Final_dict.append(dict(mainkey_dict,**_Dict[each]))
    return CraterCSV_Saving(Final_dict,csv_path)

#定义一个西数，接受一个字典作为参数
def count_craters(dict):
    #初始化五个范目内的不同标注状态的计数器
    count_1_2 = [0, 0, 0, 0]
    count_2_5 = [0, 0, 0, 0]
    count_5_10 = [0, 0, 0, 0]
    count_10_15 = [0, 0, 0, 0]
    count_15_20 = [0, 0, 0, 0]
    count_20_25 = [0, 0, 0, 0]
    # 历字典中的每个链值对
    for key, value in dict.items():
        #获取 限石坑的直径和标主状态
        if(value['is_cut']=='1'):
            diameter = float(value["Diameter"])
            status = int(value["is_AUTOvalidated"])
            # 判断限石坑的直径属于哪个范围，并在相应的计数器中增加对应的标主状态的数量
            if diameter >= 1 and diameter < 2:
                count_1_2[status] += 1
            elif diameter >= 2 and diameter < 5:
                count_2_5[status] += 1
            elif diameter >= 5 and diameter < 10:
                count_5_10[status] += 1
            elif diameter >= 10 and diameter < 15:
                count_10_15[status] += 1
            elif diameter >= 15 and diameter < 20:
                count_15_20[status] += 1
            elif diameter >= 20 and diameter < 25:
                count_20_25[status] += 1
            #返回五个范围内的不同标注状齐的计数器
    return count_1_2,count_2_5, count_5_10, count_10_15, count_15_20, count_20_25

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def analysis_fig(all_data,img_range):
    if(len(all_data)>0):
        count_1_2,count_2_5, count_5_10, count_10_15, count_15_20, count_20_25 = count_craters(all_data)
        data = [count_1_2,count_2_5, count_5_10, count_10_15, count_15_20, count_20_25]
        mydata = []
        for each in data:
            failed = [each[0],each[3]]
            successed = each[1:3]
            mydata.append(numpy.hstack([successed,failed]))
        print(data)
        print(mydata)
        fig, ax = plt.subplots()
        plt.title('{0} Crater Quality Statics (Diameter < 25KM)'.format(img_range))

        # 添加横轴标注  
        ax.set_xlabel('is_AUTOvalidated')
        ax.set_xticks([0.5, 1.5, 2.5, 3.5])
        ax.set_xticklabels(['Good', 'Verified', 'Fail-0', 'Fail-3'])

        # 添加纵轴标注
        ax.set_ylabel('Diameter')
        ax.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5, 5.5])  
        ax.set_yticklabels(['1-2KM', '2-5KM', '5-10KM', '10-15KM', '15-20KM','20-25KM'])

        for i in range(len(mydata)):
            for j in range(len(mydata[0])):
                color = mcolors.to_rgba('blue', mydata[i][j]/max(map(max, mydata)))
                ax.text(j+0.5, i+0.5, mydata[i][j], ha="center", va="center", color='w' if color[3] > 0.5 else 'k')
                ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color))

        ax.set_xlim(0, len(data[0]))
        ax.set_ylim(0, len(data))
        ax.set_aspect('equal')
        plt.show()
        fig.savefig(img_range+'_Analysis.png', dpi=300, bbox_inches='tight')

path = 'Processed'
all_data = {}
LAT_0_28_data = {}
LAT_28_60_data = {}
LAT_64_88_data = {}
data_mainkey = ''

for folder in os.listdir(path):
   folder_path = os.path.join(path, folder)
   if(os.path.splitext(folder_path)[1]==''):
    lonlat =  [int(each[1:]) for each in folder.split('_')]
    csv_path = os.path.join(folder_path,'data_verify.csv')
    img_dict,main_key = UpperCSV_Extraction(csv_path)
    data_mainkey = main_key
    if(img_dict):
        all_data.update(img_dict)
        if(abs(lonlat[1])>= 0 and abs(lonlat[1]) <= 28):
            LAT_0_28_data.update(img_dict)
        elif(abs(lonlat[1])>= 32 and abs(lonlat[1]) <= 60):
            LAT_28_60_data.update(img_dict)
        elif(abs(lonlat[1])>= 64 and abs(lonlat[1]) <= 88):
            LAT_64_88_data.update(img_dict)
    print('{0} is done'.format(folder))


img_range = 'Global'
analysis_fig(all_data,img_range)
img_range = 'Low_Latitude'
analysis_fig(LAT_0_28_data,img_range)
img_range = 'Middle_Latitude'
analysis_fig(LAT_28_60_data,img_range)
img_range = 'High_Latitude'
analysis_fig(LAT_64_88_data,img_range)
alldata_path = path + '/all_data.csv'
UpperCSV_Saving(all_data,main_key,alldata_path)