import math

import torch as th

from timelens.common import event

def _split_coordinate(c):
    c = c.float()
    left_c = c.floor()
    right_weight = c - left_c
    left_c = left_c.int()
    right_c = left_c + 1
    return left_c, right_c, right_weight

def _to_lin_idx(t, x, y, W, H, B):
    mask = (0 <= x) & (0 <= y) & (0 <= t) & (x <= W-1) & (y <= H-1) & (t <= B-1)
    lin_idx = x.long() + y.long() * W + t.long() * W * H
    return lin_idx, mask

def to_voxel_grid(event_sequence, nb_of_time_bins=5, remapping_maps=None):
    """Returns voxel grid representation of event steam.

    In voxel grid representation, temporal dimension is
    discretized into "nb_of_time_bins" bins. The events fir
    polarities are interpolated between two near-by bins
    using bilinear interpolation and summed up.

    If event stream is empty, voxel grid will be empty.
    """
    voxel_grid = th.zeros(nb_of_time_bins,
                          event_sequence._image_height,
                          event_sequence._image_width,
                          dtype=th.float32,
                          device='cpu')
    # 变成一维的了,但是为啥他会同时填充呢
    voxel_grid_flat = voxel_grid.flatten()

    # Convert timestamps to [0, nb_of_time_bins] range.
    duration = event_sequence.duration()
    start_timestamp = event_sequence.start_time()
    features = th.from_numpy(event_sequence._features)
    x = features[:, event.X_COLUMN]
    y = features[:, event.Y_COLUMN]
    polarity = features[:, event.POLARITY_COLUMN].float()
    # 上边求出的t应该是个比值吧，0~4之间
    t = (features[:, event.TIMESTAMP_COLUMN] - start_timestamp) * (nb_of_time_bins - 1) / duration
    t = t.float()

    if remapping_maps is not None:
        remapping_maps = th.from_numpy(remapping_maps)
        x, y = remapping_maps[:,y,x]
    # 得到每个事件的三维表示，其实就是把像素映射到四个事件层面上了
    left_t, right_t = t.floor(), t.floor()+1
    left_x, right_x = x.floor(), x.floor()+1
    left_y, right_y = y.floor(), y.floor()+1

    for lim_x in [left_x, right_x]:
        for lim_y in [left_y, right_y]:
            for lim_t in [left_t, right_t]:
                mask = (0 <= lim_x) & (0 <= lim_y) & (0 <= lim_t) & (lim_x <= event_sequence._image_width-1) \
                       & (lim_y <= event_sequence._image_height-1) & (lim_t <= nb_of_time_bins-1)

                # we cast to long here otherwise the mask is not computed correctly
                # 这里是在计算他在flatten中的位置啊我的乖乖！！！
                lin_idx = lim_x.long() \
                          + lim_y.long() * event_sequence._image_width \
                          + lim_t.long() * event_sequence._image_width * event_sequence._image_height
                # 下边公式对应“alex zhu的voxel出处”的公式2、3, 这里可以判断(lim_x-x)一定小于1，因为lim_x本就是x取整得到的
                # 这里看着这么像三线性插值呢，计算的是他到(left,right xyt)的权重
                weight = polarity * (1-(lim_x-x).abs()) * (1-(lim_y-y).abs()) * (1-(lim_t-t).abs())
                # 实现指定行列相加,将index加到对应的voxel_grid_flat上
                voxel_grid_flat.index_add_(dim=0, index=lin_idx[mask], source=weight[mask].float())
    # 这个和flatten共享一块内存,本质上还是那个对象
    return voxel_grid
