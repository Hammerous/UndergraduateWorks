# 导入os模块
import os

# 定义一个函数，接收一个文件夹路径作为参数
def count_files_and_folders(path):
    # 初始化一个字典，用于存储各类文件和文件夹的个数
    counts = {}
    # 遍历给定路径下的所有文件和文件夹
    for entry in os.scandir(path):
        # 如果是文件夹，就在字典中增加一个'folder'键，或者给已有的'folder'键加一
        if entry.is_dir():
            counts['folder'] = counts.get('folder', 0) + 1
            # 递归地调用函数，统计子文件夹内的文件类型及数量，并将结果与字典中已有的键值相加
            sub_counts = count_files_and_folders(entry.path)
            for key, value in sub_counts.items():
                counts[key] = counts.get(key, 0) + value
        # 如果是文件，就获取它的扩展名，并在字典中增加一个对应的键，或者给已有的键加一
        else:
            ext = os.path.splitext(entry.name)[1]
            counts[ext] = counts.get(ext, 0) + 1
    # 返回字典
    return counts

# 测试函数
path = 'Processed' # 你可以修改这个路径为你想要统计的文件夹
result = count_files_and_folders(path)
print(f'在{path}下，有以下各类文件和文件夹：')
for key, value in result.items():
    print(f'{key}: {value}')
