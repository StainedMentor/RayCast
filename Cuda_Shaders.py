import numpy as np
from numba import cuda


@cuda.jit
def column_wise_matrix_vector_mult_CUDA(matrix, vector, result):
    row, col = cuda.grid(2)
    if row < matrix.shape[0] and col < matrix.shape[1]:
        result[row, col] = matrix[row, col] * vector[row]


@cuda.jit
def subtract_vector_from_matrix_CUDA(matrix, vector):
    row, col = cuda.grid(2)
    if row < matrix.shape[0] and col < matrix.shape[1]:
        matrix[row, col] -= vector[col]


@cuda.jit
def compute_T2_CUDA(t3, t4, t8, t9, result):
    row, col = cuda.grid(2)
    if row < t3.shape[0] and col < t3.shape[1]:

        num = t8[row, col] + t9[row, col]
        denom = t3[row, col] - t4[row, col]

        result[row, col] = num / denom


@cuda.jit
def compute_T1_CUDA(s_px, s_dx, T2, r_px, r_dx, result):
    row, col = cuda.grid(2)
    if row < s_px.shape[0] and col < s_px.shape[1]:

        mult = s_dx[row, col] * T2[row, col]
        num = s_px[row, col] + mult - r_px[row]

        result[row, col] = num / r_dx[row]


@cuda.jit
def compute_intersect_CUDA(r_px, r_dx, T1, result):
    row, col = cuda.grid(2)
    if row < T1.shape[0] and col < T1.shape[1]:

        mult = r_dx[row] * T1[row, col]

        result[row, col] = r_px[row] + mult


def compute_intersect(r_px, r_dx, T1):
    result = np.zeros_like(T1, dtype=np.float32)

    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(T1.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(T1.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    d_T1 = cuda.to_device(T1)
    d_r_px = cuda.to_device(np.ascontiguousarray(r_px))
    d_r_dx = cuda.to_device(np.ascontiguousarray(r_dx))
    d_result = cuda.to_device(result)

    compute_intersect_CUDA[blockspergrid, threadsperblock](d_r_px, d_r_dx, d_T1, d_result)

    result = d_result.copy_to_host()
    return result


def compute_T1(s_px, s_dx, T2, r_px, r_dx):
    result = np.zeros_like(T2, dtype=np.float32)

    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(T2.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(T2.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    d_s_px = cuda.to_device(s_px)
    d_s_dx = cuda.to_device(s_dx)
    d_T2 = cuda.to_device(T2)
    d_r_px = cuda.to_device(np.ascontiguousarray(r_px))
    d_r_dx = cuda.to_device(np.ascontiguousarray(r_dx))
    d_result = cuda.to_device(result)

    compute_T1_CUDA[blockspergrid, threadsperblock](d_s_px, d_s_dx, d_T2, d_r_px, d_r_dx, d_result)

    result = d_result.copy_to_host()
    return result


def compute_T2(t3, t4, t8, t9):
    result = np.zeros_like(t3, dtype=np.float32)

    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(t3.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(t3.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    d_t3 = cuda.to_device(t3)
    d_t4 = cuda.to_device(t4)
    d_t8 = cuda.to_device(t8)
    d_t9 = cuda.to_device(t9)
    d_result = cuda.to_device(result)

    compute_T2_CUDA[blockspergrid, threadsperblock](d_t3, d_t4, d_t8, d_t9, d_result)

    result = d_result.copy_to_host()
    return result


def vector_subtract(a, b):
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(a.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(a.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    d_matrix = cuda.to_device(a)
    d_vector = cuda.to_device(np.ascontiguousarray(b))

    subtract_vector_from_matrix_CUDA[blockspergrid, threadsperblock](d_matrix, d_vector)

    result = d_matrix.copy_to_host()

    return result


def vector_mult(a, b):
    result = np.zeros_like(a, dtype=np.float32)

    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(a.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(a.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    d_matrix = cuda.to_device(a)
    d_vector = cuda.to_device(np.ascontiguousarray(b))
    d_result = cuda.to_device(result)

    column_wise_matrix_vector_mult_CUDA[blockspergrid, threadsperblock](d_matrix, d_vector, d_result)

    result = d_result.copy_to_host()
    return result


def get_light_points(origin, segments, points):
    angles = np.arctan2(points[:, 1] - origin[1], points[:, 0] - origin[0])
    angles = np.flip(np.sort(np.concatenate((angles - 0.00001, angles + 0.00001))), 0)

    num_rays = angles.shape[0]

    rays_px = np.repeat(origin[0], num_rays)
    rays_py = np.repeat(origin[1], num_rays)
    rays_dx = np.cos(angles)
    rays_dy = np.sin(angles)

    segments = np.expand_dims(segments, axis=0)
    segments = np.tile(segments, (num_rays, 1,1,1))

    seg_px = segments[:,:, 0, 0]
    seg_py = segments[:,:, 0, 1]
    seg_dx = segments[:,:, 1, 0] - seg_px
    seg_dy = segments[:,:, 1, 1] - seg_py

    a = vector_subtract(seg_py,rays_py)
    b = -vector_subtract(seg_px,rays_px)
    c = vector_mult(seg_dx,rays_dy)
    d = vector_mult(seg_dy,rays_dx)

    e = vector_mult(a,rays_dx)
    f = vector_mult(b,rays_dy)

    T2 = compute_T2(c,d,e,f)
    T1 = compute_T1(seg_px, seg_dx, T2, rays_px,rays_dx)

    intersect_x = compute_intersect(rays_px,rays_dx, T1)
    intersect_y = compute_intersect(rays_py,rays_dx, T1)

    intersections = np.stack((intersect_x, intersect_y, T1), axis=-1)

    out_of_bounds = np.logical_or((T1 < 0), np.logical_or(T2 < 0, T2 > 1))
    intersections[out_of_bounds, :] = np.nan

    closest = np.nanmin(intersections[:, :, 2], axis=1)
    rays_p = np.stack((rays_px, rays_py), axis=-1)
    rays_d = np.stack((rays_dx, rays_dy), axis=-1)

    result = rays_p + rays_d * np.tile(closest,(2,1)).T
    return result
