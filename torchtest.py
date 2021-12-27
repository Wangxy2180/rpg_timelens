import numpy as np
import torch


def bool_idx_test():
    idx = [True, True, True, True, True, False, False, False, False]
    val = [1, 2, 3, 4, 5, 6]
    val = np.array(val)
    val = torch.from_numpy(val).float()
    idx = torch.from_numpy(np.array(idx))
    print(val[idx])
    val.index_add_(dim=0, index=torch.Tensor(1), source=val[idx].float())


def flatten_test():
    idx = [True, True, True, True, True, False, False, False, False]
    val = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    val_add = torch.from_numpy(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]))

    val = torch.from_numpy(np.array(val))
    val_flatten = val.flatten()
    val_flatten[0] = 999
    print(val)
    # val_flatten.index_add_(dim=0, index=val_add[idx], source=val_add[idx])
    # idx就是一维上的idx，然后用idx中每个值对应的下标去加上后边val_add的值
    # idx指的是加到val_flatten的哪个idx上
    val_flatten.index_add_(dim=0, index=torch.tensor([0, 1, 2, 3, 3]), source=val_add[idx])
    print(val)


def np_flatten_test():
    # 从输出结果可以看出，numpy不共享内存，torch共享内存
    val = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    val = np.array(val)
    val_flatten = val.flatten()
    val_flatten[0] = 999
    print("np:", val)

    val = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]
    val = torch.from_numpy(np.array(val))
    val_flatten = val.flatten()
    val_flatten[0] = 999
    print("torch:", val)



if __name__ == '__main__':
    # bool_idx_test()
    # flatten_test()
    np_flatten_test()
