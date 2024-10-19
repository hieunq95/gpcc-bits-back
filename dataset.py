import os
import argparse
import numpy as np
import torch
import open3d as o3d
from torch.utils.data import DataLoader, Dataset
from util_functions import (sample_points_from_mesh, get_sparse_voxels, rescale_points,
                            get_sparse_voxels_batch, visualize_voxels, visualize_points)


class ShapeNetDataset(Dataset):
    def __init__(self, dataset_path, make_new_dataset=False, mode='train', device='cpu', resolution=(128, 128, 128),
                 crop_min_bound=(-1.0, -1.0, -1.0), crop_max_bound=(1.0, 1.0, 1.0),
                 n_points_per_cloud=20000, n_mesh_per_class=2000):
        self.dataset_path = os.path.expanduser(dataset_path)
        self.mode = mode
        self.device = device
        self.resolution = resolution
        self.crop_min_bound = crop_min_bound
        self.crop_max_bound = crop_max_bound
        self.n_points_per_cloud = n_points_per_cloud
        self.voxel_size = (self.crop_max_bound[0] - self.crop_min_bound[0]) / self.resolution[0]
        # Shapenet classes    Name          Num        (https://arxiv.org/pdf/1512.03012)
        self.shape_classes = [
            '04379243',  # table 8443
            '02958343',  # car 7497
            '03001627',  # chair 6778
            '02691156',  # airplane 4045
            '04256520',  # sofa 3173
             ]
        self.split_train_ratio = 0.8
        self.saved_data_dir = os.path.expanduser('~/open3d_data/extract/processed_shapenet/')
        if make_new_dataset:
            # save train test datasets
            self.dataset = self.save_train_test_datasets(
                split_ratio=self.split_train_ratio, dir_path=self.saved_data_dir,
                num_meshes_per_class=n_mesh_per_class, num_points_per_cloud=n_points_per_cloud
            )
        else:
            self.dataset = self.load_dataset()
        print('Dataset shape: {}'.format(self.dataset.shape))

        # get 3D bounds
        if self.mode == 'test':
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
            if os.path.isfile(path):
                mesh_obj_paths.append(path)
        print('\nFound {} mesh objects'.format(idx))
        return mesh_obj_paths

    def save_train_test_datasets(self, dir_path, split_ratio=0.8, num_meshes_per_class=100, num_points_per_cloud=100):
        """
        Save train set (voxelization) and test set (raw point cloud) into numpy array
        :param dir_path: Destination file path
        :param split_ratio: Train/test splitting ratio
        :param num_meshes_per_class: Number of meshes to be sampled from the Shapenet classes
        :param num_points_per_cloud: Number of point cloud to be sampled from each mesh object
        :return:
        """
        if self.mode == 'train':
            dataset_len = int(split_ratio * len(self.shape_classes) * num_meshes_per_class)
            dataset = np.zeros(
                shape=(dataset_len, self.resolution[0], self.resolution[1], self.resolution[2]), dtype=bool
            )
        else:
            dataset_len = int((1 - split_ratio) * len(self.shape_classes) * num_meshes_per_class)
            dataset = np.zeros(
                shape=(dataset_len, num_points_per_cloud, 3), dtype=np.float32
            )

        for i, shape_class in enumerate(self.shape_classes):
            mesh_obj_paths = self.load_mesh_object_paths(shape_class, num_meshes_per_class)
            train_set_len = int(split_ratio * len(mesh_obj_paths))
            test_set_len = int((1 - split_ratio) * len(mesh_obj_paths))
            train_paths = mesh_obj_paths[:train_set_len]
            test_paths = mesh_obj_paths[train_set_len:]
            # Create train set
            if self.mode == 'train':
                for j, trp in enumerate(train_paths):
                    points, _ = sample_points_from_mesh(trp, num_points=num_points_per_cloud,
                                                        min_bound=self.crop_min_bound, max_bound=self.crop_max_bound)
                    voxels = get_sparse_voxels(points, voxel_size=self.voxel_size, point_weight=1.0,
                                               voxel_min_bound=self.crop_min_bound, voxel_max_bound=self.crop_max_bound)
                    dataset[i * train_set_len + j] = voxels.detach().numpy()

            # Create test set
            if self.mode == 'test':
                for j, tep in enumerate(test_paths):
                    points, _ = sample_points_from_mesh(tep, num_points=num_points_per_cloud,
                                                        min_bound=self.crop_min_bound, max_bound=self.crop_max_bound)
                    dataset[i * test_set_len + j] = points

        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        if self.mode == 'train':
            save_path = dir_path + 'shapenet_train_{}.npy'.format(self.resolution[0])
        else:
            save_path = dir_path + 'shapenet_test.npy'
        print('Save dataset: {} ...'.format(save_path))
        np.save(save_path, dataset)
        return dataset

    def load_dataset(self):
        if self.mode == 'train':
            dataset = np.load(self.saved_data_dir + 'shapenet_train_{}.npy'.format(self.resolution[0]))
        else:
            dataset = np.load(self.saved_data_dir + 'shapenet_test.npy')
        return dataset

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


class SunRgbdDataset(Dataset):
    def __init__(self, dataset_path, make_new_dataset=False, mode='train', device='cpu', resolution=(128, 128, 128),
                 crop_min_bound=(-1.0, -1.0, -1.0), crop_max_bound=(1.0, 1.0, 1.0), n_points_per_cloud=20000):
        self.dataset_path = os.path.expanduser(dataset_path)
        self.mode = mode
        self.device = device
        self.resolution = resolution
        self.crop_min_bound = crop_min_bound
        self.crop_max_bound = crop_max_bound
        self.n_points_per_cloud = n_points_per_cloud
        self.saved_data_dir = os.path.expanduser('~/open3d_data/extract/processed_sunrgbd/')
        if make_new_dataset:
            self.dataset = self.save_train_test_datasets(self.resolution, self.crop_min_bound, self.crop_max_bound)
        else:
            self.dataset = self.load_dataset()
        print('Dataset shape: {}'.format(self.dataset.shape))

    def __len__(self):
        return self.dataset.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.dataset[index]).float().to(self.device)

    def load_dataset(self):
        if self.mode == 'train':
            dataset = np.load(self.saved_data_dir + 'sunrgbd_train_{}.npy'.format(self.resolution[0]))
        else:
            dataset = np.load(self.saved_data_dir + 'sunrgbd_test.npy')
        return dataset

    def save_train_test_datasets(self, resolution, min_bound, max_bound):
        print('Create SUN-RGBD {} dataset ...'.format(self.mode))
        depth_paths, image_paths = self.find_depth_image_pairs(self.dataset_path)
        voxel_size = (self.crop_max_bound[0] - self.crop_min_bound[0]) / resolution[0]
        if self.mode == 'train':
            dataset = np.zeros(
                (len(image_paths), resolution[0], resolution[1], resolution[2]), dtype=bool
            )
        else:
            dataset = np.zeros(
                (len(image_paths), self.n_points_per_cloud, 3), dtype=np.float32
            )
        for i in range(len(image_paths)):
            color_raw = o3d.io.read_image(image_paths[i])
            depth_raw = o3d.io.read_image(depth_paths[i])
            rgbd_image = o3d.geometry.RGBDImage.create_from_sun_format(color_raw, depth_raw)
            pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
                rgbd_image,
                o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
            )
            pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
            points = np.asarray(pcd.points)
            points = rescale_points(points, pcd.get_min_bound(), pcd.get_max_bound(), min_bound, max_bound)
            voxels = get_sparse_voxels(points, voxel_size, 1.0, min_bound, max_bound)
            if self.mode == 'train':
                dataset[i] = voxels.detach().numpy()
            else:
                if len(points) > 0:
                    sampled_indices = np.random.choice(len(points), self.n_points_per_cloud, replace=False)
                    dataset[i] = points[sampled_indices]
                else:
                    warning = Warning('Point cloud with len 0!')
                    print(warning)

        if not os.path.isdir(self.saved_data_dir):
            os.mkdir(self.saved_data_dir)
        print('Save {} dataset to {}'.format(self.mode, self.saved_data_dir))
        if self.mode == 'train':
            np.save(self.saved_data_dir + 'sunrgbd_train_{}.npy'.format(self.resolution[0]), dataset)
        else:
            np.save(self.saved_data_dir + 'sunrgbd_test.npy', dataset)
        return dataset

    def find_depth_image_pairs(self, base_dir):
        depth_paths = []
        image_paths = []
        for root, dirs, files in os.walk(base_dir):
            # Check if 'depth' and 'image' directories are present in the current directory
            if 'depth' in dirs and 'image' in dirs:
                depth_path = os.path.join(root, 'depth')
                image_path = os.path.join(root, 'image')
                # Find all image files in 'depth' and 'image' directories
                depth_files = [os.path.join(depth_path, f) for f in os.listdir(depth_path) if
                               os.path.isfile(os.path.join(depth_path, f))]
                image_files = [os.path.join(image_path, f) for f in os.listdir(image_path) if
                               os.path.isfile(os.path.join(image_path, f))]
                # Ensure both directories contain the same number of files
                if len(depth_files) == len(image_files):
                    depth_paths.extend(depth_files)
                    image_paths.extend(image_files)
                else:
                    print(f"Warning: Mismatch in file counts for {depth_path} and {image_path}")
        return depth_paths, image_paths


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
    # For SUN-RGBD dataset: https://rgbd.cs.princeton.edu/challenge.html
    parser = argparse.ArgumentParser(
        description="Create new datasets for training and testing from Shapenet and SUN-RGBD datasets"
    )
    parser.add_argument('--make', type=int, default=0,
                        help='Create a new dataset from mesh objects: [0, 1]')
    parser.add_argument('--mpc', type=int, default=3000,
                        help='Number of meshes per class in Shapenet dataset')
    parser.add_argument('--res', type=int, default=64,
                        help='Resolution of voxels')
    parser.add_argument('--mode', type=str, default='test',
                        help='Train/test set indicator')
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--type', type=str, default='shape',
                        help='Dataset type: [shape, sun]')
    args = parser.parse_args()

    resolution = np.full(3, args.res, dtype=np.int16)
    voxel_min_bound = np.full(3, -1.0)
    voxel_max_bound = np.full(3, 1.0)
    voxel_size = (voxel_max_bound[0] - voxel_min_bound[0]) / resolution[0]

    if args.type == 'shape':
        dataset = ShapeNetDataset(dataset_path='~/open3d_data/extract/ShapeNet/', make_new_dataset=bool(args.make),
                                  mode=args.mode, resolution=resolution, crop_min_bound=voxel_min_bound,
                                  crop_max_bound=voxel_max_bound, n_points_per_cloud=20000, n_mesh_per_class=args.mpc)
    else:
        if args.mode == 'train':
            path = '~/open3d_data/extract/SUNRGBD/'
        else:
            path = '~/open3d_data/extract/SUNRGBDv2Test/'
        dataset = SunRgbdDataset(dataset_path=path, make_new_dataset=bool(args.make),
                                 mode=args.mode, resolution=resolution, crop_min_bound=voxel_min_bound,
                                 crop_max_bound=voxel_max_bound, n_points_per_cloud=20000)

    data_loader = DataLoader(dataset, batch_size=args.batch, shuffle=False, drop_last=True)

    # Visualize some point clouds with Open3D
    for idx, x in enumerate(data_loader):
        if args.mode == 'test':
            x_batch = get_sparse_voxels_batch(x, voxel_size, 1.0, voxel_min_bound, voxel_max_bound)
        else:
            x_batch = x
        x_batch = torch.squeeze(x_batch)
        bit_depth = int(np.log2(resolution[0]))
        print('bit_depth: {}'.format(bit_depth))
        if args.mode == 'train':
            visualize_voxels(x[0])
        else:
            visualize_points(x[0])
            voxel = get_sparse_voxels(x[0], voxel_size=voxel_size, point_weight=1.0,
                                      voxel_min_bound=voxel_min_bound, voxel_max_bound=voxel_max_bound)
            print('Num voxels: {}'.format(torch.sum(voxel)))
            visualize_voxels(voxel)
        if idx == 3:
            break