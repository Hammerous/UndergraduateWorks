import numpy as np
from numba import cuda
from numba import vectorize
cuda.select_device(0)
 
@cuda.jit
def ReducedSum(arr, result):
    i, j = cuda.grid(2)
    cuda.atomic.add(result, 0, arr[i][j])
 
if __name__ == '__main__':
    import time
    np.random.seed(2)
    data_length = 2**10
    arr = np.random.random((data_length,data_length)).astype(np.float32)
    print (arr)
    arr_cuda = cuda.to_device(arr)
    np_time = 0.0
    nb_time = 0.0
    for i in range(100):
        res = np.array([0],dtype=np.float32)
        res_cuda = cuda.to_device(res)
        time0 = time.time()
        ReducedSum[(data_length,data_length),(1,1)](arr_cuda,res_cuda)
        time1 = time.time()
        res = res_cuda.copy_to_host()[0]
        time2 = time.time()
        np_res = np.sum(arr)
        time3 = time.time()
        if i == 0:
            print ('The error rate is: ', abs(np_res-res)/res)
            continue
        np_time += time3 - time2
        nb_time += time1 - time0
    print ('The time cost of numpy is: {}s'.format(np_time))
    print ('The time cost of numba is: {}s'.format(nb_time))