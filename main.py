import math
import time
import os
import sys
import gc
import argparse
import numpy as np
import torch
import open3d as o3d
import craystack as cs
import matplotlib.pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
from torch.distributions import Bernoulli
from dataset import ShapeNetDataset, SunRgbdDataset
from util_functions import *
from models import ConvoVAE

rng = np.random.RandomState(0)
torch.manual_seed(1234)
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print('Device: {}'.format(device))

def train_convo_vae(train_from_scratch=False, n_epochs=50, learning_rate=0.001, resolution=64, dataset_type='shape'):
    resolution = np.full(3, resolution, dtype=np.int32)
    voxel_min_bound = np.full(3, -1.0)
    voxel_max_bound = np.full(3, 1.0)
    param_name = 'params_{}_res_{}'.format(dataset_type, resolution[0])
    if dataset_type == 'shape':
        train_set = ShapeNetDataset(dataset_path='~/open3d_data/extract/ShapeNet/', make_new_dataset=False,
                                    mode='train', resolution=resolution, device=device,
                                    crop_min_bound=voxel_min_bound, crop_max_bound=voxel_max_bound)
    else:
        train_set = SunRgbdDataset(dataset_path='~/open3d_data/extract/SUNRGBD/', make_new_dataset=False,
                                   mode='train', resolution=resolution, device=device,
                                   crop_min_bound=voxel_min_bound, crop_max_bound=voxel_max_bound,
                                   n_points_per_cloud=20000)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True)

    if not train_from_scratch:
        model = ConvoVAE(in_dim=resolution, h_dim=500, latent_dim=50, out_dim=resolution)
        if os.path.isfile('model_params/' + param_name):
            model.load_state_dict(torch.load('model_params/' + param_name, map_location=device))
            print('Load pre-trained model ...')
    else:
        model = ConvoVAE(in_dim=resolution, h_dim=500, latent_dim=50, out_dim=resolution)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print('Model: {}'.format(model))
    print('Optimizer: {}'.format(optimizer))
    loss_avg = []

    for epoch in range(1, n_epochs + 1):
        ep_loss = []
        for batch_id, data in enumerate(train_loader):
            x_batch = torch.unsqueeze(data, 1)
            optimizer.zero_grad()
            t0 = time.time()
            loss = model.loss(x_batch)
            loss.backward()
            t1 = time.time()
            ep_loss.append(loss.item())
            optimizer.step()
            if batch_id % 20 == 0:
                print('\t--- Ep: {}, batch: {}, ep_loss: {}, time: {}'.
                      format(epoch, batch_id, np.mean(ep_loss), t1 - t0))

        loss_avg.append(np.mean(ep_loss))
        print('Epoch: {}, Avg_Loss: {}'.format(epoch, np.mean(ep_loss)))
        # save model
        torch.save(model.state_dict(), 'model_params/' + param_name)
        # save loss figure
        x_axis = np.arange(len(loss_avg))
        plt.plot(x_axis, np.array(loss_avg), '-b')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('images/train_loss.png')

def test_convo_vae(batch_size=32, generate=True, resolution=64, dataset_type='shape'):
    """
       This function should be used for the illustration purpose only, e.g., it will compress some point clouds
       and visualize the probability density estimation of the VAE model. For compressing large batch of point cloud,
       see other functions `eval_bit_rates` and `eval_bit_depth`
       """
    print('Test model\n')
    resolution = np.full(3, resolution, dtype=np.int32)
    voxel_min_bound = np.full(3, -1.0)
    voxel_max_bound = np.full(3, 1.0)
    voxel_size = (voxel_max_bound[0] - voxel_min_bound[0]) / resolution[0]
    param_name = 'params_{}_res_{}'.format(dataset_type, resolution[0])
    if dataset_type == 'shape':
        test_set = ShapeNetDataset(dataset_path='~/open3d_data/extract/ShapeNet/', make_new_dataset=False,
                                   mode='test', resolution=resolution, device='cpu',
                                   crop_min_bound=voxel_min_bound, crop_max_bound=voxel_max_bound)
    else:
        test_set = SunRgbdDataset(dataset_path='~/open3d_data/extract/SUNRGBDv2Test/', make_new_dataset=False,
                                  mode='test', resolution=resolution, device='cpu',
                                  crop_min_bound=voxel_min_bound, crop_max_bound=voxel_max_bound)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)
    model = ConvoVAE(in_dim=resolution, h_dim=500, latent_dim=50, out_dim=resolution)
    model.load_state_dict(torch.load('model_params/' + param_name, map_location=torch.device('cpu')))
    print('Model: {}'.format(model))
    model.eval()

    rec_net = torch_fun_to_numpy_fun(model.encode)
    gen_net = torch_fun_to_numpy_fun(model.decode)
    obs_codec = lambda p: cs.Bernoulli(p, 27)

    torch.manual_seed(1234)
    for batch_idx, data in enumerate(test_loader):
        x_batch = get_sparse_voxels_batch(
            points_batch=data, voxel_size=voxel_size, point_weight=1.0,
            voxel_min_bound=voxel_min_bound, voxel_max_bound=voxel_max_bound
        )
        x_batch = torch.unsqueeze(x_batch, 1)
        # Handle large batch (over 100). Otherwise, too large size of data will cause out of memory
        if resolution[0] == 128 and x_batch.size()[0] > 100:
            print('Handle large batch x_batch size: {}'.format(x_batch.size()))
            x_small_batches = torch.split(x_batch, 10)
            x_probs = torch.zeros(x_batch.size(), device='cpu')  # final output of the forward pass through the mode
            x_recon = torch.zeros(x_batch.size(), device='cpu')
            for i, x in enumerate(x_small_batches):
                x_len = x.size()[0]
                x_prob_i = model(x).detach()
                x_probs[i * x_len: (i+1) * x_len] = x_prob_i
                x_recon[i * x_len: (i+1) * x_len] = Bernoulli(x_prob_i).sample()
        else:
            x_probs = model(x_batch).detach()
            x_recon = Bernoulli(x_probs).sample()
        print('x_probs: {}'.format(x_probs.size()))
        print('x_recon: {}'.format(x_recon.size()))
        # free up memory
        del x_probs
        gc.collect()
        if generate:
            gen_probs = model.generate(x_batch.size()[0])
            x_gen_batch = Bernoulli(gen_probs).sample()
        # Compress
        bpp_bits_back, decoder_size, model_size, decoded_voxels = bits_back_coding(
            data, x_batch, voxel_size, voxel_min_bound, voxel_max_bound,
            gen_net, rec_net, obs_codec, 25, 1, True
        )

        print('Compress {} batches of voxels with bits-back coding: {} bpp'.format(batch_size, bpp_bits_back))
        if dataset_type == 'shape':
            data_indices_vis = [13, 17, 21]  # slicing indices for visualization
        else:
            data_indices_vis = [4, 10, 22]
        for j in data_indices_vis:
            x_batch_j = torch.squeeze(x_batch[j])
            x_recon_j = torch.squeeze(x_recon[j])
            x_decoded_j = np.squeeze(decoded_voxels[j])
            if not generate:
                # Visualize results
                x_ori_vis = data[j].detach().numpy()
                print('Num points: {}'.format(len(x_ori_vis)))
                try:
                    visualize_points(x_ori_vis)
                except KeyboardInterrupt:
                    sys.exit(130)
                x_vis = x_batch_j.detach().numpy().astype(np.int32)
                print('Num voxels: {}'.format(np.sum(x_vis)))
                try:
                    visualize_voxels(x_vis)
                except KeyboardInterrupt:
                    sys.exit()

                x_rec_vis = x_recon_j.detach().numpy().astype(np.int32)
                try:
                    iou_i = calculate_iou(x_vis, x_rec_vis)
                    acc_i = calculate_accuracy(x_vis, x_rec_vis)
                    print('IoU per voxel: {} / Accuracy per voxel: {}'.format(iou_i, acc_i))
                    visualize_voxels(x_rec_vis)
                except KeyboardInterrupt:
                    sys.exit()

                x_dec_vis = x_decoded_j.astype(np.int32)
                try:
                    visualize_voxels(x_dec_vis)
                except KeyboardInterrupt:
                    sys.exit()
            else:
                x_gen_j = torch.squeeze(x_gen_batch[j])
                x_gen_vis = x_gen_j.detach().numpy().astype(np.int32)
                try:
                    visualize_voxels(x_gen_vis)
                except KeyboardInterrupt:
                    sys.exit()

def eval_bit_rates(batch_values, subset_size=1, obs_precision=25, dataset_type='shape', save_results=True):
    resolution = np.full(3, 64, dtype=np.int32)
    voxel_min_bound = np.full(3, -1.0)
    voxel_max_bound = np.full(3, 1.0)
    voxel_size = (voxel_max_bound[0] - voxel_min_bound[0]) / resolution[0]
    model = ConvoVAE(in_dim=resolution, h_dim=500, latent_dim=50, out_dim=resolution)
    f_name = 'model_params/params_{}_res_{}'.format(dataset_type, resolution[0])
    model.load_state_dict(torch.load(f_name, map_location='cpu'))
    print('Model: {}'.format(model))
    model.eval()

    rec_net = torch_fun_to_numpy_fun(model.encode)
    gen_net = torch_fun_to_numpy_fun(model.decode)
    obs_codec = lambda p: cs.Bernoulli(p, obs_precision)

    if dataset_type == 'shape':
        test_set = ShapeNetDataset(dataset_path='~/open3d_data/extract/ShapeNet/', make_new_dataset=False,
                                   mode='test', resolution=resolution, device='cpu',
                                   crop_min_bound=voxel_min_bound, crop_max_bound=voxel_max_bound)
    else:
        test_set = SunRgbdDataset(dataset_path='~/open3d_data/extract/SUNRGBDv2Test/', make_new_dataset=False,
                                  mode='test', resolution=resolution, device='cpu',
                                  crop_min_bound=voxel_min_bound, crop_max_bound=voxel_max_bound)

    results_bits_back_coding, results_iterative_coding, results_draco = [], [], []
    for batch_size in batch_values:
        print('Evaluate bit rates of compression methods on {} point clouds per batch...'.format(batch_size))
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)
        bpp_bits_back_arr, bpp_draco_arr, bpp_iterative_arr = [], [], []
        # Use different for loops to avoid memory overflow

        for batch_idx, data in enumerate(test_loader):
            print('-/ Batch: {} bits-back coding'.format(batch_idx))
            x_batch = get_sparse_voxels_batch(
                data, voxel_size=voxel_size, voxel_min_bound=voxel_min_bound, voxel_max_bound=voxel_max_bound
            )
            x_batch = torch.unsqueeze(x_batch, 1)
            bpp_bits_back, _, _ = bits_back_coding(
                data, x_batch, voxel_size, voxel_min_bound, voxel_max_bound,
                gen_net, rec_net, obs_codec, obs_precision, subset_size
            )
            bpp_bits_back_arr.append(bpp_bits_back)
        del x_batch, data
        gc.collect()

        for batch_idx, data in enumerate(test_loader):
            print('-/ Batch: {} iterative coding'.format(batch_idx))
            x_batch = get_sparse_voxels_batch(
                data, voxel_size=voxel_size, voxel_min_bound=voxel_min_bound, voxel_max_bound=voxel_max_bound
            )
            x_batch = torch.unsqueeze(x_batch, 1)
            bpp_iterative, _, _ = iterative_coding(
                data, x_batch, voxel_size, voxel_min_bound, voxel_max_bound, model, obs_precision, subset_size
            )
            bpp_iterative_arr.append(bpp_iterative)
        del x_batch, data
        gc.collect()

        for batch_idx, data in enumerate(test_loader):
            print('-/ Batch: {} Draco'.format(batch_idx))
            x_batch = get_sparse_voxels_batch(
                data, voxel_size=voxel_size, voxel_min_bound=voxel_min_bound, voxel_max_bound=voxel_max_bound
            )
            x_batch = torch.unsqueeze(x_batch, 1)
            bpp_draco, _, _ = draco_ans(
                data, x_batch, voxel_size, voxel_min_bound, voxel_max_bound, 6
            )
            bpp_draco_arr.append(bpp_draco)
        del data, x_batch
        gc.collect()

        print('Average results: Bits-back coding: {} / Iterative coding: {} / Draco: {}'.format(
            np.mean(bpp_bits_back_arr), np.mean(bpp_iterative_arr), np.mean(bpp_draco_arr))
        )
        results_bits_back_coding.append(np.mean(bpp_bits_back_arr))
        results_iterative_coding.append(np.mean(bpp_iterative_arr))
        results_draco.append(np.mean(bpp_draco_arr))

    results_bits_back_coding = np.asarray(results_bits_back_coding)
    results_iterative_coding = np.asarray(results_iterative_coding)
    results_draco = np.asarray(results_draco)

    if save_results:
        if dataset_type == 'shape':
            output_dir = os.path.expanduser('~/open3d_data/extract/processed_shapenet/Bit_rate_results/')
        else:
            output_dir = os.path.expanduser('~/open3d_data/extract/processed_sunrgbd/Bit_rate_results/')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        np.save(output_dir + 'bit_rate_vs_batch_size_bits_back_coding.npy', results_bits_back_coding)
        np.save(output_dir + 'bit_rate_vs_batch_size_iterative_coding.npy', results_iterative_coding)
        np.save(output_dir + 'bit_rate_vs_batch_size_draco.npy', results_draco)

    x_axis = np.asarray(batch_values)
    plt.plot(x_axis, results_bits_back_coding, '-^')
    plt.plot(x_axis, results_iterative_coding, '--o')
    plt.plot(x_axis, results_draco, '--d')
    plt.legend(['Bits-back coding (ours)', 'Iterative coding', 'Draco'])
    plt.xlabel('Batch size')
    plt.ylabel('Bit per point')
    plt.grid(linestyle='--')
    plt.show()

def evaluate_bit_depth(depth_values, subset_size=1, batch_size=800, obs_precision=25,
                       dataset_type='shape', save_results=True):
    voxel_min_bound = np.full(3, -1.0)
    voxel_max_bound = np.full(3, 1.0)
    results_bits_back_coding, results_iterative_coding, results_draco = [], [], []
    decoder_size_bits_back_coding, decoder_size_iterative_coding, decoder_size_draco = [], [], []
    for depth in depth_values:
        resolution = np.full(3, depth, dtype=np.int32)
        voxel_size = (voxel_max_bound[0] - voxel_min_bound[0]) / resolution[0]
        # Load dataset
        if dataset_type == 'shape':
            test_set = ShapeNetDataset(dataset_path='~/open3d_data/extract/ShapeNet/', make_new_dataset=False,
                                       mode='test', resolution=resolution, device='cpu',
                                       crop_min_bound=voxel_min_bound, crop_max_bound=voxel_max_bound)
        else:
            test_set = SunRgbdDataset(dataset_path='~/open3d_data/extract/SUNRGBDv2Test/', make_new_dataset=False,
                                      mode='test', resolution=resolution, device='cpu',
                                      crop_min_bound=voxel_min_bound, crop_max_bound=voxel_max_bound)

        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)
        # Load model
        model = ConvoVAE(in_dim=resolution, h_dim=500, latent_dim=50, out_dim=resolution)
        f_name = 'model_params/params_{}_res_{}'.format(dataset_type, resolution[0])
        model.load_state_dict(torch.load(f_name, map_location='cpu'))
        print('Model: {}'.format(model))
        model.eval()

        rec_net = torch_fun_to_numpy_fun(model.encode)
        gen_net = torch_fun_to_numpy_fun(model.decode)
        obs_codec = lambda p: cs.Bernoulli(p, obs_precision)

        bpp_bits_back_arr, bpp_iterative_arr, bpp_draco_arr = [], [], []  # bit-per-point results
        pop_size_bits_back_arr, pop_size_iterative_arr, pop_size_draco_arr = [], [], []  # size of the decoders

        print('Evaluate {} bit-depth of compression methods on {} point clouds per batch...'.format(
            int(np.log2(depth)), batch_size))
        # Use different for loops to avoid memory overflow
        for batch_idx, data in enumerate(test_loader):
            print('-/ Batch: {} iterative coding'.format(batch_idx))

            x_batch = get_sparse_voxels_batch(
                data, voxel_size=voxel_size, voxel_min_bound=voxel_min_bound, voxel_max_bound=voxel_max_bound
            )
            x_batch = torch.unsqueeze(x_batch, 1)
            bpp_iterative, pop_size_iterative, vae_mode_size = iterative_coding(
                data, x_batch, voxel_size, voxel_min_bound, voxel_max_bound, model, obs_precision, subset_size
            )
            bpp_iterative_arr.append(bpp_iterative)
            pop_size_iterative_arr.append(pop_size_iterative)

        del x_batch, data
        gc.collect()

        for batch_idx, data in enumerate(test_loader):
            print('-/ Batch: {} bits-back coding'.format(batch_idx))
            x_batch = get_sparse_voxels_batch(
                data, voxel_size=voxel_size, voxel_min_bound=voxel_min_bound, voxel_max_bound=voxel_max_bound
            )
            x_batch = torch.unsqueeze(x_batch, 1)
            bpp_bits_back, pop_size_bits_back, _ = bits_back_coding(
                data, x_batch, voxel_size, voxel_min_bound, voxel_max_bound,
                gen_net, rec_net, obs_codec, obs_precision, subset_size
            )
            bpp_bits_back_arr.append(bpp_bits_back)
            pop_size_bits_back_arr.append(pop_size_bits_back)

        del x_batch, data
        gc.collect()

        for batch_idx, data in enumerate(test_loader):
            print('-/ Batch: {} Draco'.format(batch_idx))
            x_batch = get_sparse_voxels_batch(
                data, voxel_size=voxel_size, voxel_min_bound=voxel_min_bound, voxel_max_bound=voxel_max_bound
            )
            x_batch = torch.unsqueeze(x_batch, 1)
            bpp_draco, pop_size_draco, _ = draco_ans(
                data, x_batch, voxel_size, voxel_min_bound, voxel_max_bound, int(np.log2(depth))
            )
            bpp_draco_arr.append(bpp_draco)
            pop_size_draco_arr.append(pop_size_draco)

        del x_batch, data
        gc.collect()

        print('Average results: Bits-back: {} / Bernoulli: {} / Draco: {}'.format(
            np.mean(bpp_bits_back_arr), np.mean(bpp_iterative_arr), np.mean(bpp_draco_arr))
        )
        results_bits_back_coding.append(np.mean(bpp_bits_back_arr))
        results_iterative_coding.append(np.mean(bpp_iterative_arr))
        results_draco.append(np.mean(bpp_draco_arr))
        decoder_size_bits_back_coding.append(np.mean(pop_size_bits_back_arr))
        decoder_size_iterative_coding.append(np.mean(pop_size_iterative_arr))
        decoder_size_draco.append(np.mean(pop_size_draco_arr))

    results_bits_back_coding = np.asarray(results_bits_back_coding)
    results_iterative_coding = np.asarray(results_iterative_coding)
    results_draco = np.asarray(results_draco)
    decoder_size_bits_back_coding = np.asarray(decoder_size_bits_back_coding)
    decoder_size_iterative_coding = np.asarray(decoder_size_iterative_coding)
    decoder_size_draco = np.asarray(decoder_size_draco)

    if save_results:
        if dataset_type == 'shape':
            output_dir = os.path.expanduser('~/open3d_data/extract/processed_shapenet/Bit_depth_results/')
        else:
            output_dir = os.path.expanduser('~/open3d_data/extract/processed_sunrgbd/Bit_depth_results/')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        np.save(output_dir + 'bit_rate_vs_bit_depth_bits_back_coding.npy', results_bits_back_coding)
        np.save(output_dir + 'bit_rate_vs_bit_depth_iterative_coding.npy', results_iterative_coding)
        np.save(output_dir + 'bit_rate_vs_bit_depth_draco.npy', results_draco)
        np.save(output_dir + 'decoder_size_vs_bit_depth_bits_back_coding.npy', decoder_size_bits_back_coding)
        np.save(output_dir + 'decoder_size_vs_bit_depth_iterative_coding.npy', decoder_size_iterative_coding)
        np.save(output_dir + 'decoder_size_vs_bit_depth_draco.npy', decoder_size_draco)

    x_axis = np.log2(depth_values)
    # Bit-rate vs bit-depth
    plt.plot(x_axis, results_bits_back_coding, '-^')
    plt.plot(x_axis, results_iterative_coding, '--o')
    plt.plot(x_axis, results_draco, '-d')
    plt.legend(['Bits-back coding (ours)', 'Iterative coding', 'Draco'])
    plt.xlabel('Bit depth')
    plt.ylabel('Bit per point')
    plt.grid(linestyle='--')
    plt.show()
    # Decoder size vs bit-depth
    plt.plot(x_axis, decoder_size_bits_back_coding, '-^')
    plt.plot(x_axis, decoder_size_iterative_coding, '--o')
    plt.plot(x_axis, decoder_size_draco, '-d')
    plt.legend(['Bits-back coding (ours)', 'Iterative coding', 'Draco'])
    plt.xlabel('Bit depth')
    plt.ylabel('Decoder size')
    plt.yscale('log')
    plt.grid(linestyle='--')
    plt.show()


def plot_bit_rates(batch_values, dataset_type='shape', save_fig=False):
    if dataset_type == 'shape':
        output_dir = os.path.expanduser('~/open3d_data/extract/processed_shapenet/Bit_rate_results/')
    else:
        output_dir = os.path.expanduser('~/open3d_data/extract/processed_sunrgbd/Bit_rate_results/')
    results_bits_back_coding = np.load(output_dir + 'bit_rate_vs_batch_size_bits_back_coding.npy')
    results_iterative_coding = np.load(output_dir + 'bit_rate_vs_batch_size_iterative_coding.npy')
    results_draco = np.load(output_dir + 'bit_rate_vs_batch_size_draco.npy')

    x_axis = np.asarray(batch_values)
    plt.plot(x_axis, results_bits_back_coding, '-^', linewidth=2.0)
    plt.plot(x_axis, results_iterative_coding, '--o', linewidth=2.0)
    plt.plot(x_axis, results_draco, '--d', linewidth=2.0)
    plt.legend(['Bits-back coding (ours)', 'Iterative coding', 'Draco'], fontsize=14)
    plt.xlabel('Number of point clouds', fontsize=16)
    plt.ylabel('Bit per point', fontsize=16)
    plt.grid(linestyle='--')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=13)
    if save_fig:
        plt.savefig('images/bit-rate-results.pdf')
    plt.show()

def plot_bit_depth(depth_values, metric='bit_rate', save_fig=True):
    # Load the data
    shapenet_dir = os.path.expanduser('~/open3d_data/extract/processed_shapenet/Bit_depth_results/')
    sunrgbd_dir = os.path.expanduser('~/open3d_data/extract/processed_sunrgbd/Bit_depth_results/')

    results_bits_back_shape = np.load(shapenet_dir + '{}_vs_bit_depth_bits_back_coding.npy'.format(metric))
    results_bits_back_sun = np.load(sunrgbd_dir + '{}_vs_bit_depth_bits_back_coding.npy'.format(metric))
    results_draco_shape = np.load(shapenet_dir + '{}_vs_bit_depth_draco.npy'.format(metric))
    results_draco_sun = np.load(sunrgbd_dir + '{}_vs_bit_depth_draco.npy'.format(metric))
    results_iterative_shape = np.load(shapenet_dir + '{}_vs_bit_depth_iterative_coding.npy'.format(metric))
    results_iterative_sun = np.load(sunrgbd_dir + '{}_vs_bit_depth_iterative_coding.npy'.format(metric))

    # Rescale data
    if metric == 'decoder_size':
        results_bits_back_shape /= 8 * 10**6  # in MB
        results_bits_back_sun /= 8 * 10**6
        results_draco_shape /= 8 * 10**6
        results_draco_sun /= 8 * 10**6
        results_iterative_shape /= 8 * 10**6
        results_iterative_sun /= 8 * 10**6

    # print out
    print('Bits-back {} results on Shapenet: {}'.format(metric, np.round(results_bits_back_shape, 2)))
    print('Bits-back {} results on Sun-RGBD: {}'.format(metric, np.round(results_bits_back_sun, 2)))
    print('No-bits-back {} results on Shapenet: {}'.format(metric, np.round(results_iterative_shape, 2)))
    print('No-bits-back {} results on Sun-RGBD: {}'.format(metric, np.round(results_iterative_sun, 2)))
    print('Draco results {} on Shapenet    : {}'.format(metric, np.round(results_draco_shape, 2)))
    print('Draco results {} on Sun-RGBD    : {}'.format(metric, np.round(results_draco_sun, 2)))

    # Flip and set the x-axis (bit depth) values
    x_axis = np.flip(np.log2(depth_values))

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 3))  # 1 row, 2 columns

    # Plot the first subplot (Shapenet results)
    ax1.plot(x_axis, np.flip(results_bits_back_shape), '-^', label='Bits-back')
    ax1.plot(x_axis, np.flip(results_iterative_shape), '-o', label='No-bits-back')
    ax1.plot(x_axis, np.flip(results_draco_shape), '-d', label='Draco')
    ax1.set_xlabel('Bit depth')
    if metric == 'bit_rate':
        ax1.set_ylabel('Bit per point')
    else:
        ax1.set_ylabel('Decoder size (MB)')
        ax1.set_yscale('log')
    ax1.set_title('Shapenet Results')
    ax1.legend()
    ax1.grid(linestyle='--')

    # Plot the second subplot (SUN-RGBD results)
    ax2.plot(x_axis, np.flip(results_bits_back_sun), '--^', label='Bits-back')
    ax2.plot(x_axis, np.flip(results_iterative_sun), '--o', label='No-bits-back')
    ax2.plot(x_axis, np.flip(results_draco_sun), '--d', label='Draco')
    ax2.set_xlabel('Bit depth')
    if metric == 'bit_rate':
        ax2.set_ylabel('Bit per point')
    else:
        ax2.set_ylabel('Decoder size (MB)')
        ax2.set_yscale('log')
    ax2.set_title('SUN-RGBD Results')
    ax2.legend()
    ax2.grid(linestyle='--')

    # Set the same y-axis limits for both subplots
    y_max = max(np.max(results_bits_back_shape), np.max(results_draco_shape),
                np.max(results_iterative_shape), np.max(results_iterative_sun),
                np.max(results_bits_back_sun), np.max(results_draco_sun)
                )
    if metric == 'bit_rate':
        y_max += 1.0
        ax1.set_ylim([-0.1, y_max])
        ax2.set_ylim([-0.1, y_max])
    else:
        y_max += 100

    # Adjust layout and display the plot
    plt.tight_layout()  # Adjust subplots to fit in figure area.
    if save_fig:
        plt.savefig('images/bit-depth-{}-results.pdf'.format(metric))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Main script for running GPCC-bits-back")
    parser.add_argument('--mode', type=str, default='train',
                        help='Evaluation mode: [train, test, eval_rate, eval_depth, plot_rate, plot_depth]')
    parser.add_argument('--ep', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--init', type=int, default=1,
                        help='Only use this when we train from scratch: [0, 1]')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for the optimizer, e.g., Adam')
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size of test set for compression')
    parser.add_argument('--gen', type=int, default=0,
                        help='Use 1 if we want to generate random samples from the model')
    parser.add_argument('--res', type=int, default=64,
                        help='Resolution of voxels: [32, 64, 128]')
    parser.add_argument('--type', type=str, default='shape',
                        help='Dataset type: [shape, sun]')
    args = parser.parse_args()
    if args.mode == 'train':
        train_convo_vae(train_from_scratch=bool(args.init), n_epochs=args.ep, learning_rate=args.lr,
                        resolution=args.res, dataset_type=args.type)
    elif args.mode == 'test':
        test_convo_vae(batch_size=args.batch, generate=bool(args.gen), resolution=args.res, dataset_type=args.type)
    elif args.mode == 'eval_rate':
        batch_vals = [100 * i for i in [2, 4, 6, 8, 10, 12]]
        eval_bit_rates(batch_values=batch_vals, subset_size=1, dataset_type=args.type, save_results=True)
    elif args.mode == 'eval_depth':
        depth_vals = [128, 64, 32]
        evaluate_bit_depth(depth_vals, subset_size=1, dataset_type=args.type, batch_size=args.batch, save_results=True)
    elif args.mode == 'plot_rate':
        batch_vals = [100 * i for i in [2, 4, 6, 8, 10, 12]]
        plot_bit_rates(batch_vals, args.type, True)
    elif args.mode == 'plot_depth':
        depth_vals = [128, 64, 32]
        for m in ['bit_rate', 'decoder_size']:
            plot_bit_depth(depth_vals, m, True)
    else:
        parser.print_help()