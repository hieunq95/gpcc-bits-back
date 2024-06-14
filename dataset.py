import os
import open3d as o3d
import numpy as np
from torch.utils.data import DataLoader, Dataset
from util_functions import *


class ShapeNetDataset(Dataset):
    def __init__(self, dataset_path, save_train_test_sets=False, mode='train', device='cpu'):
        self.dataset_path = os.path.expanduser(dataset_path)
        self.mode = mode
        self.device = device
        self.shape_classes = ['02691156', '02958343', '03001627', '03636649', '04256520', '04379243']

        if save_train_test_sets:
            # save train test datasets
            self.save_train_test_datasets(dir_path=os.path.expanduser('~/open3d_data/extract/processed_shapenet/'),
                                          num_meshes_per_class=2000, num_points_per_cloud=2000)
        # load dataset
        if self.mode == 'train':
            self.dataset = self.load_dataset(
                os.path.expanduser('~/open3d_data/extract/processed_shapenet/shapenet_train_set.npy'))
        else:
            self.dataset = self.load_dataset(
                os.path.expanduser('~/open3d_data/extract/processed_shapenet/shapenet_test_set.npy'))

        # get 3D bounds
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

    def save_sample_point_cloud_set(self, mesh_obj_paths, num_points_per_cloud, saved_path):
        pc_stack = []
        for path in mesh_obj_paths:
            points, _ = sample_points_from_mesh(path, num_points=num_points_per_cloud)
            pc_stack.append(points)
        pc_stack = np.asarray(pc_stack)
        print('Save point cloud dataset pc_stack.shape: {}'.format(pc_stack.shape))
        np.save(saved_path, pc_stack)

    def save_train_test_datasets(self, dir_path, split_ratio=0.8, num_meshes_per_class=100, num_points_per_cloud=100):
        mesh_train_paths = []
        mesh_test_paths = []
        train_stack = []
        test_stack = []
        for shape_class in self.shape_classes:
            mesh_obj_paths = self.load_mesh_object_paths(shape_class, num_meshes_per_class)
            dataset_len = len(mesh_obj_paths)
            train_set_len = int(split_ratio * dataset_len)
            train_paths = mesh_obj_paths[:train_set_len]
            test_paths = mesh_obj_paths[train_set_len:]
            print('Save {} class: {} train point clouds, {} test point clouds'.format(
                shape_class, len(train_paths), len(test_paths)))

            for trp in train_paths:
                mesh_train_paths.append(trp)
            for tep in test_paths:
                mesh_test_paths.append(tep)

        # test valid mesh objects before saving
        for obj_path in mesh_train_paths:
            points, _ = sample_points_from_mesh(obj_path, num_points=num_points_per_cloud)
            if points is not None:
                train_stack.append(points)

        for obj_path in mesh_test_paths:
            points, _ = sample_points_from_mesh(obj_path, num_points=num_points_per_cloud)
            if points is not None:
                test_stack.append(points)

        train_stack = np.asarray(train_stack)
        test_stack = np.asarray(test_stack)
        print('Train set shape: {}, test set shape: {}'.format(train_stack.shape, test_stack.shape))

        save_train_path = dir_path + 'shapenet_train_set.npy'
        save_test_path = dir_path + 'shapenet_test_set.npy'
        print('Save train set: {}'.format(save_train_path))
        np.save(save_train_path, train_stack)
        print('Save test set: {}'.format(save_test_path))
        np.save(save_test_path, test_stack)

    def load_dataset(self, path):
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
    dataset = ShapeNetDataset(dataset_path='~/open3d_data/extract/ShapeNet/',
                              save_train_test_sets=True, mode='train')
    data_loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)
    # Visualize some point clouds with Open3D
    points = data_loader.dataset[0]
    print('Points shape: {}'.format(points.shape))
    visualize_points(points)