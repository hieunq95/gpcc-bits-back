import math
import shutil
import os
import sys
import time
import lzma
import dill
import gc
import mat73
import subprocess
import re
import numpy as np
import craystack as cs
import open3d as o3d
import torch
from plyfile import PlyData, PlyElement
from autograd.builtins import tuple as ag_tuple
from craystack import bb_ans

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


def visualize_voxels(voxel_cube, voxel_size=0.001):
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

def save_test_files_ply(file_path, destination='~/', n_points_per_cloud=20000):
    file_path = os.path.expanduser(file_path)
    destination = os.path.expanduser(destination)
    if not os.path.isfile(file_path):
        raise Exception('File not found')
    if not os.path.isdir(destination):
        os.mkdir(destination)
    _, pcd = sample_points_from_mesh(file_path, num_points=n_points_per_cloud)
    points_np = np.asarray(pcd.points, dtype=np.float32)
    vertex = np.array([(point[0], point[1], point[2]) for point in points_np],
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex')

    # Save file
    file_id = re.split('/', file_path)
    dest_fname = file_id[6] + '_' + file_id[7] + '.ply'
    dest_fname = destination + dest_fname
    # print('Save PLY file to: {}'.format(dest_fname))
    # Write to a PLY file
    PlyData([el]).write(dest_fname)

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
    resolution = int((voxel_max_bound[0] - voxel_min_bound[0]) / voxel_size)
    batch_size = points_batch.size()[0]
    grid_voxels_batches = torch.zeros((batch_size, resolution, resolution, resolution))
    for i, points_i in enumerate(points_batch):
        grid_patches_i = get_sparse_voxels(points_i, voxel_size=voxel_size, point_weight=point_weight,
                                           voxel_min_bound=voxel_min_bound, voxel_max_bound=voxel_max_bound,
                                           visualize=False)
        grid_voxels_batches[i] = grid_patches_i

    return grid_voxels_batches.float()


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

def draco_compress(data, out_dir, quantization):
    def compress_ply(input_ply, output_drc, quantz):
        result = subprocess.run(
            ['./draco/draco_encoder', '-i', input_ply, '-o', output_drc, '-qp', '{}'.format(quantz), '-cl', '7'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        output = result.stdout + result.stderr
        # print('output: {}'.format(output))
        encoded_size = None
        for line in output.split('\n'):
            if "Encoded size" in line:
                encoded_size = int(line.split('=')[1].strip().split()[0])
                # print('output ++: encoded_size bytes {}'.format(encoded_size))
                break
        return encoded_size

    out_dir = os.path.expanduser(out_dir)
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    point_clouds = data.detach().numpy()
    batch_size = point_clouds.shape[0]
    encoded_sizes = []
    encoded_fnames = []
    # print('Compress {} point clouds with Draco'.format(batch_size))
    for i in range(batch_size):
        points = point_clouds[i]
        # print('points_i: {}'.format(points.shape))
        input_ply = out_dir + 'temp_{}.ply'.format(i)

        if not os.path.isdir(out_dir + 'Compress/'):
            os.mkdir(out_dir + 'Compress/')
        output_drc = out_dir + 'Compress/' + '{}.drc'.format(i)

        vertex = np.array([(point[0], point[1], point[2]) for point in points],
                          dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        el = PlyElement.describe(vertex, 'vertex')
        # Write to a PLY file
        PlyData([el]).write(input_ply)
        encoded_size = compress_ply(input_ply, output_drc, quantization)
        encoded_sizes.append(encoded_size)
        encoded_fnames.append(output_drc)
        os.remove(input_ply)

    return encoded_sizes, encoded_fnames, point_clouds

def draco_decompress(input_file_names, output_folder):
    def decompress_drc(input_drc, output_ply):
        subprocess.run(
            ['./draco/draco_decoder', '-i', input_drc, '-o', output_ply],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

    output_folder = os.path.expanduser(output_folder)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    decompressed_ply_files = []

    for file_name in input_file_names:
        # print('Decompress {} with Draco'.format(file_name))
        if file_name.endswith('.drc'):
            output_fname = file_name.replace('.drc', '.ply')
            output_fname = output_fname.replace('Compress', 'Decompress')
            output_ply = os.path.join(output_folder, output_fname.replace('.drc', '.ply'))
            decompressed_ply_files.append(output_ply)
            decompress_drc(file_name, output_ply)

    return decompressed_ply_files

# Functions to compress dataset
def draco_ans(point_clouds, voxel_batch, voxel_size, voxel_min_bound, voxel_max_bound, quantz_level):
    draco_results_dir = os.path.expanduser('~/open3d_data/extract/processed_shapenet/Draco_results/')
    data = point_clouds
    t0 = time.time()
    compressed_sizes, compressed_fnames, raw_point_clouds = draco_compress(
        data, draco_results_dir, quantz_level
    )
    t1 = time.time()
    flat_message_len = np.sum(compressed_sizes) * 8  # maybe we also need to calculate overhead of Draco's decoder?
    num_voxels = point_clouds.size()[0] * point_clouds.size()[1]
    bpv_overhead = flat_message_len / num_voxels
    print('--- Draco -- encoded in {} seconds, bpv: {}, bpv_overhead: {}'.format(
        t1 - t0, bpv_overhead, bpv_overhead)
    )
    # Decode
    draco_decode_dir = os.path.expanduser('~/open3d_data/extract/processed_shapenet/Draco_results/Decompress/')
    t0 = time.time()
    decompressed_file_names = draco_decompress(compressed_fnames, draco_decode_dir)
    t1 = time.time()
    # Check compression quality
    iou_arr = []
    for i in range(len(decompressed_file_names)):
        pcd = o3d.io.read_point_cloud(decompressed_file_names[i])
        decoded_points = np.asarray(pcd.points)
        decoded_points = rescale_points(
            decoded_points, pcd.get_min_bound(), pcd.get_max_bound(), voxel_min_bound, voxel_max_bound
        )
        decoded_voxel = get_sparse_voxels(decoded_points, voxel_size, 1.0, voxel_min_bound, voxel_max_bound)
        original_voxel = get_sparse_voxels(data[i], voxel_size, 1.0, voxel_min_bound, voxel_max_bound)
        iou_i = calculate_iou(original_voxel, decoded_voxel)
        iou_arr.append(iou_i)
    print('--- Draco -- decoded in {} seconds, average IoU: {}, std IoU: {}'.format(
        t1 - t0, np.mean(iou_arr), np.std(iou_arr))
    )
    return bpv_overhead

def bernoulli_ans(point_clouds, voxel_batch, voxel_size, voxel_min_bound, voxel_max_bound,
                  model, obs_precision, subset_size=1):
    data_shape = voxel_batch.size()
    num_data = data_shape[0]
    # num_voxels = torch.sum(voxel_batch)
    num_voxels = point_clouds.size()[0] * point_clouds.size()[1]
    obs_shape = (subset_size, data_shape[1], data_shape[2], data_shape[3], data_shape[4])
    obs_size = np.prod(obs_shape)
    latent_size = np.prod((subset_size, model.latent_dim))
    codec = lambda p: cs.Bernoulli(p, obs_precision)

    # Encode data using small batches (preventing forward big batch of data -> crash)
    init_message = cs.base_message(obs_size)
    assert num_data % subset_size == 0
    pop_array = []
    message = init_message
    t0 = time.time()

    for x in voxel_batch:  # small batches
        p = model(x).detach().numpy().flatten()
        push, pop = codec(p)
        pop_array.append(pop)
        message, = push(message, np.asarray(x.detach().numpy().flatten(), dtype=np.uint8))
    flat_message = cs.flatten(message)
    t1 = time.time()
    codec_compressor = lzma.LZMACompressor()
    pop_size = 0
    for pf in pop_array:
        serialized_pop = dill.dumps(pf)
        pfc = codec_compressor.compress(serialized_pop)
        pop_size += len(pfc) * 8
    codec_compressor.flush()
    if data_shape[-1] == 32:
        model_size = 2.1 * 10**6 * 8  # in bits
    elif data_shape[-1] == 64:
        model_size = 8.4 * 10**6 * 8
    else:
        model_size = 8.4 * 10**6 * 8
    pop_size = min(pop_size, model_size)   # compare size of the Codec with size of the deep learning model
    flat_message_len = 32 * len(flat_message)
    bpv_overhead = (pop_size + flat_message_len) / num_voxels
    bpv = flat_message_len / num_voxels
    print('--- NoBB_VAE -- encoded in {} seconds, bpv: {}, bpv_overhead: {}'.format(
        t1 - t0, bpv, bpv_overhead)
    )
    # free up some memory
    del message, codec_compressor
    gc.collect()
    save_dir = os.path.expanduser('~/open3d_data/extract/processed_shapenet/Bernoulli_results/')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    np.save(
        os.path.expanduser(save_dir + 'Bernoulli_shapenet_{}.npy'.format(num_data)),
        flat_message
    )
    # large batch cause memory overflow, we can only decode smaller than 800 point clouds
    if not (data_shape[-1] == 128 and num_data > 800):
        # Decode message
        t0 = time.time()
        message_ = cs.unflatten(flat_message, obs_size)
        # free up some memory
        del flat_message
        gc.collect()
        data_decoded = []
        for i in range(len(pop_array)):
            pop = pop_array[-1 - i]  # reverse order
            message_, data_ = pop(message_, )
            data_decoded.append(np.asarray(data_, dtype=np.uint8))  # cast dtype to prevent out of memory issue
        t1 = time.time()

        data_decoded = reversed(data_decoded)
        # Check quality
        iou_arr = []
        for x, x_, pb in zip(voxel_batch, data_decoded, point_clouds):
            np.testing.assert_equal(x.detach().numpy().flatten(), x_)
            decoded_voxel = np.reshape(np.squeeze(x_), data_shape[2:])
            original_voxel = get_sparse_voxels(
                torch.squeeze(pb), voxel_size, 1.0, voxel_min_bound, voxel_max_bound
            ).detach().numpy()
            iou_i = calculate_iou(original_voxel, decoded_voxel)
            iou_arr.append(iou_i)
        print('--- NoBB_VAE -- decoded in {} seconds, average IoU: {}, std IoU: {}'.format(
            t1 - t0, np.mean(iou_arr), np.std(iou_arr))
        )
        del decoded_voxel, data_decoded
        gc.collect()
    return bpv_overhead, bpv

def bits_back_vae_ans(point_clouds, voxel_batch, voxel_size, voxel_min_bound, voxel_max_bound,
                      gen_net, rec_net, obs_codec, obs_precision, subset_size=1):
    def vae_view(head):
        return ag_tuple((np.reshape(head[:latent_size], latent_shape),
                         np.reshape(head[latent_size:], obs_shape)))

    data_shape = voxel_batch.size()
    num_data = data_shape[0]
    # num_voxels = torch.sum(voxel_batch)
    num_voxels = point_clouds.size()[0] * point_clouds.size()[1]
    assert num_data % subset_size == 0
    num_subsets = num_data // subset_size
    latent_dim = 50
    latent_shape = (subset_size, latent_dim)  # [1, 50]
    latent_size = np.prod(latent_shape)
    obs_shape = (subset_size, data_shape[1], data_shape[2], data_shape[3], data_shape[4])  # [1, 1, 128, 128, 128]
    obs_size = np.prod(obs_shape)

    data = np.split(np.asarray(voxel_batch.detach().numpy(), dtype=np.bool_), num_subsets)
    # Create codec
    vae_append, vae_pop = cs.repeat(cs.substack(
        bb_ans.VAE(gen_net, rec_net, obs_codec, 8, obs_precision),
        vae_view), num_subsets)
    # Encode
    t0 = time.time()
    init_message = cs.base_message(obs_size + latent_size)
    message, = vae_append(init_message, data)
    flat_message = cs.flatten(message)
    flat_message_len = 32 * len(flat_message)
    t1 = time.time()
    # Compress the Codec itself (should be useful for communication of the model and codec)
    codec_compressor = lzma.LZMACompressor()
    compressed_pop = codec_compressor.compress(dill.dumps(vae_pop))
    codec_compressor.flush()
    # Calculate bit rate and save compressed data
    pop_size = len(compressed_pop) * 8
    if data_shape[-1] == 32:
        model_size = 2.1 * 10**6 * 8  # in bits
    elif data_shape[-1] == 64:
        model_size = 8.4 * 10**6 * 8
    else:
        model_size = 8.4 * 10**6 * 8
    pop_size = min(pop_size, model_size)  # compare size of the Codec with size of the deep learning model
    bpv_overhead = (pop_size + flat_message_len) / num_voxels
    print('--- BB_VAE -- encoded in {} seconds, bpv: {}, bpv_overhead: {}'.format(
        t1 - t0, flat_message_len / num_voxels, bpv_overhead)
    )
    # free up some memory
    del message, data, codec_compressor
    gc.collect()

    # save results
    save_dir = os.path.expanduser('~/open3d_data/extract/processed_shapenet/BB_VAE_results/')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    np.save(
        os.path.expanduser(save_dir + 'BB_VAE_shapenet_{}.npy'.format(num_data)),
        flat_message
    )

    # Decode
    t0 = time.time()
    message = cs.unflatten(flat_message, obs_size + latent_size)
    # free up some memory
    del flat_message
    gc.collect()
    message, data_ = vae_pop(message)
    del message
    gc.collect()
    data = np.split(np.asarray(voxel_batch.detach().numpy(), dtype=np.bool_), num_subsets)
    np.testing.assert_equal(data, data_)  # Check lossless compression
    t1 = time.time()
    # Check quality
    iou_arr = []
    for i in range(len(data_)):
        decoded_voxel = np.squeeze(data_[i])
        original_voxel = get_sparse_voxels(
            torch.squeeze(point_clouds[i]), voxel_size, 1.0, voxel_min_bound, voxel_max_bound
        )
        original_voxel = original_voxel.detach().numpy()
        iou_i = calculate_iou(original_voxel, decoded_voxel)
        iou_arr.append(iou_i)
    print('--- BB_VAE -- decoded in {} seconds, average IoU: {}, std IoU: {}'.format(
        t1 - t0, np.mean(iou_arr), np.std(iou_arr))
    )

    del data
    gc.collect()

    return bpv_overhead, data_




