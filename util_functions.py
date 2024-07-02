import math
import shutil
import os
import sys
import time
import scipy
import mat73
import re
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import open3d as o3d
import torch


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


def visualize_voxels(voxel_cube, voxel_size):
    """
    Visualize 3D sparse voxel cubes `[height, width, length]`
    :param voxel_cube:
    :return:
    """
    voxel_cube = np.asarray(voxel_cube)
    indices = np.nonzero(voxel_cube)
    points = np.vstack(indices).T.astype(np.float32)
    # point_cloud = o3d.geometry.PointCloud()
    # point_cloud.points = o3d.utility.Vector3dVector(points)
    # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(point_cloud, voxel_size=voxel_size)
    # o3d.visualization.draw_geometries([voxel_grid])
    visualize_points(points)


def load_mesh_object(file_path, compute_vertex=False, visualize=False):
    mesh = o3d.io.read_triangle_mesh(file_path)
    # print('Mesh with color: {}'.format(mesh.has_vertex_colors()))
    if compute_vertex:
        mesh.compute_vertex_normals()
    if visualize:
        o3d.visualization.draw_geometries([mesh])

    return mesh


def sample_points_from_mesh(file_path, num_points=5000, min_bound=(-1.0, -1.0, -1.0), max_bound=(1.0, 1.0, 1.0),
                            visualize=False, print_log=False):
    """
    Sample uniformly points from a mesh. Return None if the mesh object fails to be loaded
    """
    # We do a trick here to avoid errors when loading the mesh objects
    new_path = os.path.expanduser('~/open3d_data/extract/ShapeNet/model_normalized.obj')
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
        points = rescale_points(points, pcd.get_min_bound(), pcd.get_max_bound(), min_bound, max_bound)
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

    for v in all_indices:
        grid_patches[v[0], v[1], v[2]] = point_weight

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


def rescale_points(points, origin_min_bound, origin_max_bound, new_min_bound, new_max_bound):
    """
    Rescale the point cloud for training/testing
    :param points:
    :param origin_min_bound: Original min bounds of the point cloud
    :param origin_max_bound: Original max bounds
    :param new_min_bound: Rescaled min bounds of the point cloud
    :param new_max_bound: Rescaled max bounds
    :return: points: (N, 3)
    """
    desired_scale = np.max(new_max_bound) - np.min(new_min_bound)
    origin_scale = np.max(origin_max_bound) - np.min(origin_min_bound)
    mean_point = np.mean(points, axis=0)
    points = (points - mean_point) * desired_scale / origin_scale
    points = points_remover(points, new_min_bound, new_max_bound)
    return points

def points_remover(points, voxel_min_bound, voxel_max_bound):
    """
    Remove points that are outside the min_bound and max_bound
    :param points: (N, 3)
    :param voxel_min_bound: min bound
    :param voxel_max_bound: max bound
    :return: points: (N, 3)
    """
    for i in range(len(points)):
        if not (voxel_min_bound[0] < points[i][0] < voxel_max_bound[0]
        and voxel_min_bound[1] < points[i][1] < voxel_max_bound[1]
        and voxel_min_bound[2] < points[i][2] < voxel_max_bound[2]):
            points[i][:] = 0
    return points

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

def test_mps_device():
    # Only works for Mac-OS
    time_avg = []
    device = 'mps'
    for i in range(500):
        t0 = time.time()
        a_mps = torch.rand(size=(1000, 1000), device=device)
        b_mps = torch.rand(size=(1000, 500), device=device)
        c_mps = torch.matmul(a_mps, b_mps)
        t1 = time.time()
        time_avg.append(t1 - t0)
    print('Time {} average: {}, std: {}'.format(device, np.mean(time_avg), np.std(time_avg)))

def calculate_psnr(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 1.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_iou(original, decompressed):
    intersection = np.logical_and(original, decompressed).sum()
    union = np.logical_or(original, decompressed).sum()
    iou = intersection / union
    return iou

def calculate_accuracy(original, decompressed):
    correct_voxels = np.sum(original == decompressed)
    total_voxels = original.size
    accuracy = correct_voxels / total_voxels
    return accuracy

def import_dataset(save_img=True):
    def rgb_to_grayscale(rgb_image, width=480):
        # Define the weights for each channel
        weights = np.array([0.2989, 0.5870, 0.1140])

        # Compute the dot product of the image and the weights
        gray_image = np.dot(rgb_image[..., :3], weights)

        # Reshape to add an additional dimension for consistency
        gray_image = gray_image.reshape((width, width, 1))

        return gray_image
    if save_img:
        path = os.path.expanduser('~/Downloads/nyu_depth_v2_labeled.mat')
        save_path = os.path.expanduser('~/open3d_data/extract/processed_nyu/')
        nyu_dataset = mat73.loadmat(path)
        keys = list(nyu_dataset.keys())
        print(keys)
        images = np.asarray(nyu_dataset['images'])
        depths = np.asarray(nyu_dataset['depths'])
        print('images: {}'.format(images.shape))  # [H, W, 3, N]
        print('depths: {}'.format(depths.shape))  # [H, W, N]
        (W, H, N) = depths.shape
        depths = depths[:, :W, :]
        images = images[:, :W, :, :]
        gray_images = np.zeros(shape=(W, W, 1, N), dtype=np.float32)
        for i in range(N):
            gray_images[:, :, :, i] = rgb_to_grayscale(images[:, :, :, i], width=W)

        depths = np.ascontiguousarray(depths).astype(np.float32)
        gray_images = np.ascontiguousarray(gray_images).astype(np.float32) / 255.0
        print('Save NYU dataset to ' + save_path)
        np.save(os.path.expanduser(save_path + 'nyu_depths'), depths)
        np.save(os.path.expanduser(save_path + 'nyu_images'), gray_images)
    else:
        depths = np.load(os.path.expanduser('~/open3d_data/extract/processed_nyu/nyu_depths.npy'))
        images = np.load(os.path.expanduser('~/open3d_data/extract/processed_nyu/nyu_images.npy'))
        print('depths: {}'.format(depths.shape))
        print('images: {}'.format(images.shape))
        for i in range(10):
            dep = np.ascontiguousarray(depths[:, :, -i])
            img = np.ascontiguousarray(images[:, :, :, -i])
            rgbd_depth = o3d.geometry.Image(dep)
            rgbd_color = o3d.geometry.Image(img)
            rgbd_img = o3d.geometry.RGBDImage.create_from_color_and_depth(rgbd_color, rgbd_depth)
            print('rgbd_img: {}'.format(rgbd_img))
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_img,
                o3d.camera.PinholeCameraIntrinsic(
                    o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
            # Flip it, otherwise the pointcloud will be upside down
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            print(pcd)
            points = np.asarray(pcd.points)
            # visualize_points(points)
            # Rescale point cloud
            voxel_min_bound = np.full(3, -1.0)
            voxel_max_bound = np.full(3, 1.0)
            shape = np.full(3, 32, dtype=np.int32)
            points = rescale_points(points, pcd.get_min_bound(), pcd.get_max_bound(), voxel_min_bound, voxel_max_bound)
            visualize_points(points)
            voxel_size = (voxel_max_bound[0] - voxel_min_bound[0]) / shape[0]
            voxels = get_sparse_voxels(points, voxel_min_bound=voxel_min_bound, voxel_max_bound=voxel_max_bound,
                                       voxel_size=voxel_size, point_weight=1.0)
            visualize_voxels(voxels, voxel_size*20)
            ocv = torch.sum(voxels)
            print('voxels.size(): {}'.format(voxels.size()))
            print('ocv: {}'.format(ocv))

            # octree
            voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=0.00005)
            o3d.visualization.draw_geometries([voxel_grid])
            octree = o3d.geometry.Octree(max_depth=7)
            octree.create_from_voxel_grid(voxel_grid)
            o3d.visualization.draw_geometries([octree])
            print('octree: {}'.format(octree))
            print('size of octree: {}'.format(sys.getsizeof(octree)))
            print('size of voxels: {}'.format(sys.getsizeof(voxels)))
            print(octree.root_node)

def obj_to_ply(path=None):
    path = os.path.expanduser('~/open3d_data/extract/ShapeNet/02691156/f8fa93d7b17fe6126bded4fd00661977/models/model_normalized.obj')
    mesh = o3d.io.read_triangle_mesh(path)
    pcd = mesh.sample_points_uniformly(number_of_points=20000)
    print('points: {}'.format(pcd.points))
    points = np.asarray(pcd.points, dtype=np.float32)
    new_pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))

    # print('points.dtype: {}'.format(points.dtype))
    print('new pcd: {}'.format(new_pcd))
    o3d.visualization.draw_geometries([new_pcd])
    new_path = os.path.expanduser('~/open3d_data/extract/test_io.ply')
    # o3d.io.write_triangle_mesh(new_path, mesh, write_vertex_colors=False, write_vertex_normals=False,
    #                            write_triangle_uvs=False, compressed=True)
    o3d.io.write_point_cloud(new_path, new_pcd)


if __name__ == '__main__':
    # import_dataset(save_img=False)
    # obj_to_ply()
    x = np.asarray([[0, 1, 0, 1],
                    [0, 0, 0, 0]])
    y = np.asarray([[0, 1, 0, 0],
                    [0, 0, 0, 0]])
    print(calculate_iou(x, y))
