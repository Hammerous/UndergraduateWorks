import cv2,os,csv
import pandas as pd

def drawLabel(data,image_path,new_image_path):
#读取图片
    img = cv2.imread(image_path)
    box = [float(each) for each in data['box info'].split()]
    #获取矩形框的左上角和右下角坐标
    x1 = int(float(box[0] - box[2] / 2) * img.shape[0])
    y1 = int(float(box[1] - box[3] / 2)  * img.shape[1])
    x2 = int(float(box[0] + box[2] / 2)  * img.shape[0])
    y2 = int(float(box[1] + box[3] / 2) * img.shape[1])

    #在图片上绘制矩形框，颜色为红色，线宽为2
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    #保存新图片
    cv2.imwrite(new_image_path, img)

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
    if(img_dict):
        data_mainkey = main_key
        for key, value in img_dict.items():
            if(value['is_cut']=='1' and value['is_AUTOvalidated']!='0' and value['is_AUTOvalidated']!='3'):
                jpg_path = os.path.join(folder_path,key+'.jpg')
                if(os.path.exists(jpg_path)):
                        #定义新图片的保存路径
                    new_image_path = os.path.join(folder_path,'labeled'+key+'.jpg')
                    drawLabel(value,jpg_path,new_image_path)

    print('{0} is done'.format(folder))