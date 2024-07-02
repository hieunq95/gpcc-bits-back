import os
import argparse
import open3d as o3d
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from util_functions import *


class ShapeNetDataset(Dataset):
    def __init__(self, dataset_path, save_train_test_sets=False, mode='train', device='cpu', resolution=(128, 128, 128),
                 crop_min_bound=(-0.5, -0.5, -0.5), crop_max_bound=(0.5, 0.5, 0.5), n_points_per_cloud=20000,
                 n_mesh_per_class=1000, return_voxel=True):
        self.dataset_path = os.path.expanduser(dataset_path)
        self.mode = mode
        self.device = device
        self.resolution = resolution
        self.crop_min_bound = crop_min_bound
        self.crop_max_bound = crop_max_bound
        self.return_voxel = return_voxel
        self.n_points_per_cloud = n_points_per_cloud
        self.voxel_size = (self.crop_max_bound[0] - self.crop_min_bound[0]) / self.resolution[0]
        # self.shape_classes = ['02691156', '02958343', '03001627', '03636649', '04256520',
        #                       '04379243', '04530566', '04090263', '02924116', '03467517']
        self.shape_classes = ['04379243',  # table 8443
                              '02958343',  # car 7497
                              '03001627',  # chair 6778
                              '02691156',  # airplane 4045
                              '04256520',  # sofa 3173
                              ]

        if save_train_test_sets:
            # save train test datasets
            self.save_train_test_datasets(dir_path=os.path.expanduser('~/open3d_data/extract/processed_shapenet/'),
                                          num_meshes_per_class=n_mesh_per_class, num_points_per_cloud=n_points_per_cloud)
        # load dataset with a fixed slice_window across the shape classes. This is to ensure the equal number of shapes
        if self.mode == 'train':
            slice_window = int(0.79 * n_mesh_per_class)
        else:
            slice_window = int(0.19 * n_mesh_per_class)
        self.dataset = self.load_dataset(slice_window=slice_window)
        print('dataset shape: {}'.format(self.dataset.shape))

        # get 3D bounds
        if not return_voxel:
            self.pc_min_bound, self.pc_max_bound = self.get_pc_bounds()

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.dataset[index]).float().to(self.device)

    def load_mesh_object_paths(self, shape_class='02958343', num_meshes=2000):
        folder_path = self.dataset_path + shape_class
        all_dirs = [x[0] for x in os.walk(folder_path)]
        all_obj_path = []
        mesh_obj_paths = []
        # mesh_objects = []
        idx = 0
        for s in all_dirs:
            if s.find('models') > 0 and s.find('untitled') == -1:
                obj_path = s + '/model_normalized.obj'
                all_obj_path.append(obj_path)

        for path in all_obj_path[:num_meshes]:
            idx += 1
            print('Load mesh object {}: {}'.format(idx, path))
            # mesh_object = o3d.io.read_triangle_mesh(path, print_progress=True)
            # mesh_objects.append(mesh_object)
            if os.path.isfile(path):
                mesh_obj_paths.append(path)
        print('\nFound {} mesh objects'.format(idx))
        return mesh_obj_paths

    def save_train_test_datasets(self, dir_path, split_ratio=0.8, num_meshes_per_class=100, num_points_per_cloud=100):
        for shape_class in self.shape_classes:
            train_stack = []
            test_stack = []
            train_voxel_stack = []
            test_voxel_stack = []
            mesh_obj_paths = self.load_mesh_object_paths(shape_class, num_meshes_per_class)
            dataset_len = len(mesh_obj_paths)
            train_set_len = int(split_ratio * dataset_len)
            train_paths = mesh_obj_paths[:train_set_len]
            test_paths = mesh_obj_paths[train_set_len:]
            print('Save {} class: {} train point clouds, {} test point clouds'.format(
                shape_class, len(train_paths), len(test_paths)))

            for trp in train_paths:
                points, _ = sample_points_from_mesh(trp, num_points=num_points_per_cloud,
                                                    min_bound=self.crop_min_bound, max_bound=self.crop_max_bound)
                voxels = get_sparse_voxels(points, voxel_size=self.voxel_size, point_weight=1.0,
                                           voxel_min_bound=self.crop_min_bound, voxel_max_bound=self.crop_max_bound)
                train_stack.append(points)
                train_voxel_stack.append(voxels)
            train_stack = np.asarray(train_stack)
            train_voxel_stack = np.asarray(train_voxel_stack, dtype=np.uint8)
            save_train_path = dir_path + 'shapenet_train_set_{}.npy'.format(shape_class)
            save_train_voxel_path = dir_path + 'shapenet_voxel_train_set_{}.npy'.format(shape_class)
            print('Save train set: {}'.format(save_train_voxel_path))
            np.save(save_train_path, train_stack)
            np.save(save_train_voxel_path, train_voxel_stack)

            for tep in test_paths:
                points, _ = sample_points_from_mesh(tep, num_points=num_points_per_cloud,
                                                    min_bound=self.crop_min_bound, max_bound=self.crop_max_bound)
                voxels = get_sparse_voxels(points, voxel_size=self.voxel_size, point_weight=1.0,
                                           voxel_min_bound=self.crop_min_bound, voxel_max_bound=self.crop_max_bound)
                test_stack.append(points)
                test_voxel_stack.append(voxels)
            test_stack = np.asarray(test_stack)
            test_voxel_stack = np.asarray(test_voxel_stack, dtype=np.uint8)
            save_test_path = dir_path + 'shapenet_test_set_{}.npy'.format(shape_class)
            save_test_voxel_path = dir_path + 'shapenet_voxel_test_set_{}.npy'.format(shape_class)
            print('Save test set: {}'.format(save_test_voxel_path))
            np.save(save_test_path, test_stack)
            np.save(save_test_voxel_path, test_voxel_stack)

    def load_dataset(self, slice_window=160):
        n_classes = len(self.shape_classes)
        if self.return_voxel:
            dataset = np.zeros(
                shape=(n_classes * slice_window, self.resolution[0], self.resolution[1], self.resolution[2]),
                dtype=np.uint8
            )
        else:
            dataset = np.zeros(
                shape=(n_classes * slice_window, self.n_points_per_cloud, 3),
                dtype=np.float64
            )
        if self.mode == 'train':
            if not self.return_voxel:
                for i, shape_class in enumerate(self.shape_classes):
                    f_path = os.path.expanduser(
                        '~/open3d_data/extract/processed_shapenet/shapenet_train_set_{}.npy'.format(shape_class)
                    )
                    dataset_i = self.load_class_set(f_path)
                    dataset[i*slice_window:(i + 1)*slice_window, :, :] = dataset_i[:slice_window]
            else:
                for i, shape_class in enumerate(self.shape_classes):
                    f_path = os.path.expanduser(
                        '~/open3d_data/extract/processed_shapenet/shapenet_voxel_train_set_{}.npy'.format(shape_class)
                    )
                    dataset_i = self.load_class_set(f_path)
                    dataset[i*slice_window:(i + 1)*slice_window, :, :, :] = dataset_i[:slice_window]
        else:
            if not self.return_voxel:
                for i, shape_class in enumerate(self.shape_classes):
                    f_path = os.path.expanduser(
                        '~/open3d_data/extract/processed_shapenet/shapenet_test_set_{}.npy'.format(shape_class)
                    )
                    dataset_i = self.load_class_set(f_path)
                    dataset[i*slice_window:(i + 1)*slice_window, :, :] = dataset_i[:slice_window]
            else:
                for i, shape_class in enumerate(self.shape_classes):
                    f_path = os.path.expanduser(
                        '~/open3d_data/extract/processed_shapenet/shapenet_voxel_test_set_{}.npy'.format(shape_class)
                    )
                    dataset_i = self.load_class_set(f_path)
                    dataset[i*slice_window:(i + 1)*slice_window, :, :, :] = dataset_i[:slice_window]
        return dataset

    def load_class_set(self, path):
        pc_stack = np.load(path)
        print('Load dataset {}, shape: {}'.format(path, pc_stack.shape))
        return pc_stack

    def get_pc_bounds(self):
        """
        Get min and max bounds of the point cloud dataset. This will be useful for normalization
        :return: [min_height, min_width, min_length], [max_height, max_width, max_length]
        """
        min_bounds = np.min(self.dataset, 0)
        max_bounds = np.max(self.dataset, 0)

        min_bound = np.min(min_bounds, 0)
        max_bound = np.max(max_bounds, 0)

        print('Point cloud dataset bounds: {} - {}'.format(min_bound, max_bound))

        return min_bound, max_bound


if __name__ == '__main__':
    # Create point cloud dataset from ShapeNet mesh objects
    # Let's download and extract ShapeNetCore v2 dataset into ~/open3d_data/extract/ShapeNet/
    # Dataset tree:
    # open3d_data/
    #  > extract/
    #  | |> ShapeNet/
    #  | | |> 02958343/
    #  | | |> 03001627/
    #  | | |> 02691156/
    #  | | | |> 1a04e3eab45ca15dd86060f189eb133/
    # Link to download https://shapenet.org/
    parser = argparse.ArgumentParser(description="Create new datasets from mesh objects")
    parser.add_argument('--make', type=int, default=0,
                        help='Create a new dataset from mesh objects: [0, 1]')
    parser.add_argument('--mpc', type=int, default=2000,
                        help='Number of meshes per class')
    args = parser.parse_args()

    resolution = np.full(3, 128, dtype=np.int32)
    voxel_min_bound = np.full(3, -1.0)
    voxel_max_bound = np.full(3, 1.0)
    voxel_size = (voxel_max_bound[0] - voxel_min_bound[0]) / resolution[0]
    if args.make == 0:
        new_datasets = False
    else:
        new_datasets = True

    voxel_dataset = ShapeNetDataset(dataset_path='~/open3d_data/extract/ShapeNet/', save_train_test_sets=new_datasets,
                                    mode='test', resolution=resolution,
                                    crop_min_bound=voxel_min_bound, crop_max_bound=voxel_max_bound,
                                    n_points_per_cloud=20000, n_mesh_per_class=args.mpc, return_voxel=True)

    points_dataset = ShapeNetDataset(dataset_path='~/open3d_data/extract/ShapeNet/', save_train_test_sets=new_datasets,
                                     mode='test', resolution=resolution,
                                     crop_min_bound=voxel_min_bound, crop_max_bound=voxel_max_bound,
                                     n_points_per_cloud=20000, n_mesh_per_class=args.mpc, return_voxel=False)

    voxel_data_loader = DataLoader(voxel_dataset, batch_size=2, shuffle=True, drop_last=True)
    points_data_loader = DataLoader(points_dataset, batch_size=2, shuffle=True, drop_last=True)

    # Visualize some point clouds with Open3D
    for idx, x in enumerate(voxel_data_loader):
        voxel = x[0]
        visualize_voxels(voxel, voxel_size*40)
        print('Occupied voxels: {}'.format(torch.sum(voxel)))
        if idx == 3:
            break

    for idx, x in enumerate(points_data_loader):
        points = x[0]
        visualize_points(points)
        voxel = get_sparse_voxels(points, voxel_size=voxel_size, point_weight=1.0,
                                  voxel_min_bound=voxel_min_bound, voxel_max_bound=voxel_max_bound)
        print('Num voxels: {}'.format(torch.sum(voxel)))
        visualize_voxels(voxel, voxel_size=voxel_size*20)
        if idx == 3:
            break

