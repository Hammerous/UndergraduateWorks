import cupy as cp

# 定义一个函数，接受一个二维数组作为参数
def distance_matrix(points):
    # 获取数组的形状
    n, m = points.shape
    # 沿着第一个轴复制n次
    points_row = cp.repeat(points, n, axis=0)
    # 沿着第二个轴复制n次
    points_col = cp.tile(points, (n, 1))
    # 相减得到差值数组
    diff = points_row - points_col
    # 沿着第三个轴求平方和
    square_sum = cp.sum(diff**2, axis=1)
    # 开根号得到距离矩阵
    dist_mat = cp.sqrt(square_sum).reshape(n, n)
    # 返回距离矩阵
    return dist_mat

# 定义一个函数，接受一个距离矩阵作为参数
def min_distance(dist_mat):
    # 获取矩阵的形状
    n, n = dist_mat.shape
    # 创建一个无穷大值的对角矩阵
    inf_diag = cp.diag([cp.inf] * n)
    # 将距离矩阵与对角矩阵相加，忽略自身与自身的距离
    dist_mat += inf_diag
    # 沿着第二个轴求最小值
    min_dist = []
    for mat_line in dist_mat:
        min_val = float(min(mat_line).get())
        for i in range(len(mat_line)):
            if(mat_line[i]==min_val):
                min_dist.append([i,min_val])
                break
    # 返回最小距离列表
    return min_dist

# 测试数据
# 设置随机数种子
cp.random.seed(40)
# 生成20个随机的三维空间中的点的坐标
points = cp.random.randint(0, 100, size=(500, 3))
#points = cp.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])

# 调用函数计算距离矩阵
dist_mat = distance_matrix(points)
#print("距离矩阵为：")
#print(dist_mat)

# 调用函数计算最小距离列表
min_dist = min_distance(dist_mat)
print("最小距离列表为：")
for each in min_dist:
    print(each)