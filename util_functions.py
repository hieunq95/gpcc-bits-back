import math
import shutil
import os
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import torch
import constriction
from sklearn.linear_model import Lasso


def custom_draw_geometry_with_rotation(pcd, interactive=True, include_coordinate=True):
    def rotate_view(vis):
        # vis.create_window(width=1920, height=1080)
        ctr = vis.get_view_control()
        parameters = o3d.io.read_pinhole_camera_parameters("ScreenCamera_2024-04-16-16-45-05.json")
        ctr.convert_from_pinhole_camera_parameters(parameter=parameters, allow_arbitrary=True)
        # ctr.rotate(0.0, 0.0)
        return False
    coordinate_mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])
    if not interactive:
        if include_coordinate:
            o3d.visualization.draw_geometries_with_animation_callback([pcd, coordinate_mesh], rotate_view)
        else:
            o3d.visualization.draw_geometries_with_animation_callback([pcd], rotate_view)
    else:
        if include_coordinate:
            o3d.visualization.draw_geometries([pcd, coordinate_mesh])
        else:
            o3d.visualization.draw_geometries([pcd])


def visualize_points(points, interactive=True):
    """
    Visualize `points` as a numpy input `(N,3)`
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    custom_draw_geometry_with_rotation(pcd, interactive=interactive)


def visualize_voxels(voxel_cube):
    """
    Visualize 3D sparse voxel cubes `[height, width, length]`
    :param voxel_cube:
    :return:
    """
    voxel_cube = np.asarray(voxel_cube)
    indices = np.nonzero(voxel_cube)
    points = np.vstack(indices).T.astype(np.float32)
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=1)
    o3d.visualization.draw_geometries([voxel_grid])


def load_mesh_object(file_path, compute_vertex=False, visualize=False):
    mesh = o3d.io.read_triangle_mesh(file_path)
    # print('Mesh with color: {}'.format(mesh.has_vertex_colors()))
    if compute_vertex:
        mesh.compute_vertex_normals()
    if visualize:
        o3d.visualization.draw_geometries([mesh])

    return mesh


def sample_points_from_mesh(file_path, num_points=5000, visualize=False, print_log=False):
    """
    Sample uniformly points from a mesh. Return None if the mesh object fails to be loaded
    """
    # We do a trick here to avoid errors when loading the mesh objects
    new_path = '/Users/hieu/open3d_data/extract/ShapeNet/model_normalized.obj'
    shutil.copyfile(file_path, new_path)
    mesh = o3d.io.read_triangle_mesh(new_path)
    points = None
    if print_log:
        print('mesh object path: {}'.format(file_path))
    try:
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    except:
        pcd = None
    if pcd is not None:
        points = np.asarray(pcd.points)
    if print_log and pcd is not None:
        print('points.shape: {}'.format(points.shape))
        print('Point cloud bounds: {}, {}'.format(pcd.get_min_bound(), pcd.get_max_bound()))
    if visualize and pcd is not None:
        custom_draw_geometry_with_rotation(pcd)
    # We delete the newly create file
    if os.path.isfile(new_path):
        os.remove(new_path)
    return points, pcd


def get_sparse_voxels(points, voxel_size, point_weight, voxel_min_bound, voxel_max_bound, visualize=False):
    """
    Convert a numpy point cloud into sparse voxel representation [Height, Width, Length].
    Voxels that have `k` points inside will receive value `k * point_weight`
    :param points: A numpy point cloud `(N, 3)`
    :param voxel_size: Size of a voxel
    :param point_weight: Weight of a point for counting voxel values
    :param voxel_min_bound: Min bound of the sparse voxels. Shape `(height_min, width_min, length_min)`
    :param voxel_max_bound: Max bound of the sparse voxels. Shape `(height_max, width_max, length_max)`
    :param visualize: Visualize the voxels
    :return: A 3D sparse voxel `[Height, Width, Length]`
    """
    if not isinstance(points, np.ndarray):
        points = points.cpu().data.numpy()
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        input=pcd, voxel_size=voxel_size, min_bound=voxel_min_bound, max_bound=voxel_max_bound)
    # print('voxel_grid: {}'.format(voxel_grid))
    if visualize:
        o3d.visualization.draw_geometries([voxel_grid])

    all_voxels = voxel_grid.get_voxels()
    all_indices = []
    for voxel in all_voxels:
        all_indices.append(voxel.grid_index)

    max_grid_val = voxel_grid.get_voxel(voxel_max_bound)
    grid_patches = torch.zeros(size=(max_grid_val[0], max_grid_val[1], max_grid_val[2]), dtype=torch.float64)

    # grid_patches = np.zeros(
    #     shape=(max_grid_val[0], max_grid_val[1], max_grid_val[2]), dtype=np.float64)

    for v in all_indices:
        grid_patches[v[0], v[1], v[2]] += point_weight

    return grid_patches


def get_sparse_voxels_batch(points_batch, voxel_size, point_weight=1.0, voxel_min_bound=(-1.0, -1.0, -1.0),
                            voxel_max_bound=(1.0, 1.0, 1.0)):
    """
    Return a batch of sparse voxels `[num_batches, heigh, width, length]`
    :param points_batch: Batch of point clouds `[num_batches, N, 3]`
    :param voxel_size: Size of a voxel
    :param point_weight: Weight of a point for counting voxel values
    :param voxel_min_bound: Min bound of the sparse voxels. Shape `(height_min, width_min, length_min)`
    :param voxel_max_bound: Max bound of the sparse voxels. Shape `(height_max, width_max, length_max)`
    :return: A batch of sparse voxels `[num_batches, heigh, width, length]`
    """
    grid_voxels_batches = []
    for points_i in points_batch:
        grid_patches_i = get_sparse_voxels(points_i, voxel_size=voxel_size, point_weight=point_weight,
                                           voxel_min_bound=voxel_min_bound, voxel_max_bound=voxel_max_bound,
                                           visualize=False)
        grid_voxels_batches.append(torch.unsqueeze(grid_patches_i, 0))

    return torch.cat(grid_voxels_batches, 0).float()


def zero_padding(points_3d, axis, val, patch_size):
    """
    Make a sparse representation of point cloud with zero-padding
    :param points_3d: Input point cloud `(N,3)`
    :param axis: Axis we want to divide the 3D space into patches
    :param val: Value to start the patch
    :param patch_size: Size of the patch along the `axis`
    :return: Sparse point cloud `(N,3)`, and number of non-zero coefficients `non_zero_points`
    """
    sparse_points = np.copy(points_3d)
    non_zero_points = 0
    for n in range(sparse_points.shape[0]):
        if not val <= sparse_points[n, axis] <= val + patch_size:
            sparse_points[n, :] = 0.0
        else:
            non_zero_points += 1

    return sparse_points, non_zero_points


def get_zero_padding_patches(points_3d, axis, patch_size):
    """
    Return a set of patches zero padded `[num_patches, N, 3]'
    :param points_3d: 3D point cloud
    :param axis: Axis we want to divide the 3D space into patches
    :param patch_size: Size of the patch along the `axis`
    :return: A set of patches zero padded `[num_patches, N, 3]'
    """
    sparse_points = np.copy(points_3d)
    min_bound = np.min(sparse_points, axis=0)
    max_bound = np.max(sparse_points, axis=0)
    points_range = max_bound - min_bound
    n_iters = int(points_range[axis] / patch_size) + 1

    sparse_points_batch = []

    for i in range(n_iters):
        val_i = min_bound[axis] + i * patch_size
        sparse_points_i, _ = zero_padding(sparse_points, axis=axis, val=val_i, patch_size=patch_size)
        # print('val_i: {}'.format(val_i))
        # print('sparse_points_i: {}'.format(sparse_points_i))
        sparse_points_batch.append(sparse_points_i)

    sparse_points_batch = np.asarray(sparse_points_batch)

    # print('min_bound: {}, max_bound: {}'.format(min_bound, max_bound))
    # print('patch_size: {}'.format(patch_size))
    # print('n_inters: {}'.format(n_iters))
    # print('sparse_points_batch: {}'.format(sparse_points_batch.shape))

    return sparse_points_batch


def calculate_batch_entropy(x, base=2):
    """
    Calculate entropy `H_b(x)` of a batch data `[batch_size, num_features]`. For better accuracy of entropy,
    the `batch_size` should be sufficient large
    :param x: Input data with shape `[batch_size, dim0, dim1]` or `[batch_size, dim0, dim1, dim2]`
    :param base: Base to calculate the log_prob
    :return: Shannon entropy of the batch `x`
    """
    def binary_entropy(p_x):
        if p_x != 0:
            log_px = math.log(1.0/p_x, 2)
        else:
            log_px = 0.0
        return p_x * log_px

    # print('x: {}'.format(x))
    # print('x.shape: {}'.format(x.shape))
    num_dims = len(x.shape)
    # Reshape x so that we have x_flat has the shape [batch_size, num_features]
    if num_dims == 3:
        x_flat = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2]))
    else:
        x_flat = np.reshape(x, (x.shape[0], x.shape[1] * x.shape[2] * x.shape[3]))

    # print('x_flat shape: {}'.format(x_flat.shape))
    x_binary = np.zeros(shape=x_flat.shape)
    for i in range(len(x_flat)):
        for j in range(len(x_flat[i])):
            if x_flat[i][j] != 0:
                x_binary[i][j] = 1
    # print('x_zeros after:\n {}'.format(x_binary))
    # calculate batch entropy on x_binary
    h_x = 0  # entropy of the ensemble X
    batch_size = x_binary.shape[0]
    for j in range(len(x_binary[0])):
        p_1j = np.sum(x_binary[:, j]) / batch_size
        p_0j = 1.0 - p_1j
        # print('p_1j: {}'.format(p_1j))
        h_xj = binary_entropy(p_0j) + binary_entropy(p_1j)
        # print('h_xj: {}'.format(h_xj))
        h_x += h_xj

    return h_x


def tensor_to_ndarray(tensor):
    if type(tensor) is tuple:
        return tuple(tensor_to_ndarray(t) for t in tensor)
    else:
        return tensor.detach().numpy()


def ndarray_to_tensor(arr):
    if type(arr) is tuple:
        return tuple(ndarray_to_tensor(a) for a in arr)
    elif type(arr) is torch.Tensor:
        return arr
    else:
        return torch.from_numpy(np.float32(arr))


def torch_fun_to_numpy_fun(fun):
    def numpy_fun(*args, **kwargs):
        torch_args = ndarray_to_tensor(args)
        return tensor_to_ndarray(fun(*torch_args, **kwargs))
    return numpy_fun


def compress_bernoulli(x, probs, debug=False):
    """
    Compress a sequence of Bernoulli distributed data
    :param x: Input sequence
    :param probs: Probabilities of the sequence (e.g., output of the deep neural nets)
    :return: Compressed binary sequence
    """
    message = np.array(x, dtype=np.int32)
    entropy_model = constriction.stream.model.Bernoulli()
    probabilities = np.array(probs, dtype=np.float64)
    encoder = constriction.stream.stack.AnsCoder()
    encoder.encode_reverse(message, entropy_model, probabilities)
    compressed = encoder.get_compressed()
    if debug:
        # print('Compressed: {}'.format(compressed))
        # print('Compressed_len / message_len: {} / {}'.format(len(compressed), len(message)))
        bin_mess = [bin(word) for word in message]
        bin_mess_compressed = [bin(word) for word in compressed]
        # print('in binary: {}'.format(bin_mess_compressed))
        print('binary_compressed_len: {}'.format(32 * len(bin_mess_compressed)))
        print('binary_message_len: {}'.format(32 * len(bin_mess)))
        print('Bits per voxels: {}'.format(len(bin_mess_compressed) / len(bin_mess)))
    decoder = constriction.stream.stack.AnsCoder(compressed)
    decoded = decoder.decode(entropy_model, probabilities)
    assert np.all(decoded == message)
    return compressed, decoded