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
                                  mode='test', resolution=resolution, crop_min_bound=voxel_min_bound,
                                  crop_max_bound=voxel_max_bound)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)
    model = ConvoVAE(in_dim=resolution, h_dim=500, latent_dim=50, out_dim=resolution)
    model.load_state_dict(torch.load('model_params/' + param_name, map_location=torch.device('cpu')))
    print('Model: {}'.format(model))
    model.eval()

    rec_net = torch_fun_to_numpy_fun(model.encode)
    gen_net = torch_fun_to_numpy_fun(model.decode)
    obs_codec = lambda p: cs.Bernoulli(p, 27)

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
        bpv_bits_back, decoded_voxels = bits_back_vae_ans(
            data, x_batch, voxel_size, voxel_min_bound, voxel_max_bound,
            gen_net, rec_net, obs_codec, 25, 1
        )

        print('Compress {} batches of voxels with BB_ANS: {} bpv'.format(batch_size, bpv_bits_back))
        for j in range(x_batch.size()[0]):
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
                                  mode='test', resolution=resolution, crop_min_bound=voxel_min_bound,
                                  crop_max_bound=voxel_max_bound)

    results_bitsback, results_bernoulli, results_draco, results_optimal = [], [], [], []
    for batch_size in batch_values:
        print('Evaluate bit rates of compression methods on {} point clouds per batch...'.format(batch_size))
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)
        bpv_bits_back_arr, bpv_bernoulli_arr, bpv_draco_arr, bpv_optimal_arr = [], [], [], []
        for batch_idx, data in enumerate(test_loader):
            # if batch_idx > 12:
            #     break
            print('-/ Batch: {}'.format(batch_idx))
            x_batch = get_sparse_voxels_batch(
                data, voxel_size=voxel_size, voxel_min_bound=voxel_min_bound, voxel_max_bound=voxel_max_bound
            )
            x_batch = torch.unsqueeze(x_batch, 1)

            bpv_bits_back, _ = bits_back_vae_ans(
                data, x_batch, voxel_size, voxel_min_bound, voxel_max_bound,
                gen_net, rec_net, obs_codec, obs_precision, subset_size
            )

            bpv_bernoulli, bpv_optimal = bernoulli_ans(
                data, x_batch, voxel_size, voxel_min_bound, voxel_max_bound, model, obs_precision, subset_size
            )

            bpv_draco = draco_ans(
                data, x_batch, voxel_size, voxel_min_bound, voxel_max_bound, 6
            )

            bpv_bits_back_arr.append(bpv_bits_back)
            bpv_bernoulli_arr.append(bpv_bernoulli)
            bpv_draco_arr.append(bpv_draco)
            bpv_optimal_arr.append(bpv_optimal)

        print('Average results: Bits-back: {} / Bernoulli: {} / Draco: {}'.format(
            np.mean(bpv_bits_back_arr), np.mean(bpv_bernoulli_arr), np.mean(bpv_draco_arr))
        )
        results_bitsback.append(np.mean(bpv_bits_back_arr))
        results_bernoulli.append(np.mean(bpv_bernoulli_arr))
        results_draco.append(np.mean(bpv_draco_arr))
        results_optimal.append(np.mean(bpv_optimal_arr))

    results_bitsback = np.asarray(results_bitsback)
    results_bernoulli = np.asarray(results_bernoulli)
    results_draco = np.asarray(results_draco)
    results_optimal = np.asarray(results_optimal)

    if save_results:
        if dataset_type == 'shape':
            output_dir = os.path.expanduser('~/open3d_data/extract/processed_shapenet/Bit_rate_results/')
        else:
            output_dir = os.path.expanduser('~/open3d_data/extract/processed_sunrgbd/Bit_rate_results/')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        np.save(output_dir + 'bit_rate_vs_batch_size_bitsback.npy', results_bitsback)
        np.save(output_dir + 'bit_rate_vs_batch_size_bernoulli.npy', results_bernoulli)
        np.save(output_dir + 'bit_rate_vs_batch_size_draco.npy', results_draco)
        np.save(output_dir + 'bit_rate_vs_batch_size_optimal.npy', results_optimal)

    x_axis = np.asarray(batch_values)
    plt.plot(x_axis, results_bitsback, '-^')
    plt.plot(x_axis, results_optimal, '--s')
    plt.plot(x_axis, results_bernoulli, '--o')
    plt.plot(x_axis, results_draco, '-d')
    plt.legend(['Bits-back', 'Optimal', 'No-bits-back', 'Draco'])
    plt.xlabel('Batch size')
    plt.ylabel('Bit per point')
    plt.grid(linestyle='--')
    plt.show()

def evaluate_bit_depth(depth_values, subset_size=1, batch_size=800, obs_precision=25,
                       dataset_type='shape', save_results=True):
    voxel_min_bound = np.full(3, -1.0)
    voxel_max_bound = np.full(3, 1.0)
    results_bitsback, results_bernoulli, results_draco, results_optimal = [], [], [], []
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
                                      mode='test', resolution=resolution, crop_min_bound=voxel_min_bound,
                                      crop_max_bound=voxel_max_bound)

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

        bpv_bits_back_arr, bpv_bernoulli_arr, bpv_draco_arr, bpv_optimal_arr = [], [], [], []
        print('Evaluate {} bit-depth of compression methods on {} point clouds per batch...'.format(
            int(np.log2(depth)), batch_size))
        for batch_idx, data in enumerate(test_loader):
            print('-/ Batch: {}'.format(batch_idx))

            x_batch = get_sparse_voxels_batch(
                data, voxel_size=voxel_size, voxel_min_bound=voxel_min_bound, voxel_max_bound=voxel_max_bound
            )
            x_batch = torch.unsqueeze(x_batch, 1)

            bpv_bits_back, _ = bits_back_vae_ans(
                data, x_batch, voxel_size, voxel_min_bound, voxel_max_bound,
                gen_net, rec_net, obs_codec, obs_precision, subset_size
            )

            bpv_draco = draco_ans(
                data, x_batch, voxel_size, voxel_min_bound, voxel_max_bound, int(np.log2(depth))
            )

            bpv_bernoulli, bpv_optimal = bernoulli_ans(
                data, x_batch, voxel_size, voxel_min_bound, voxel_max_bound, model, obs_precision, subset_size
            )

            bpv_bits_back_arr.append(bpv_bits_back)
            bpv_draco_arr.append(bpv_draco)
            bpv_bernoulli_arr.append(bpv_bernoulli)
            bpv_optimal_arr.append(bpv_optimal)

        print('Average results: Bits-back: {} / Bernoulli: {} / Draco: {}'.format(
            np.mean(bpv_bits_back_arr), np.mean(bpv_bernoulli_arr), np.mean(bpv_draco_arr))
        )
        results_bitsback.append(np.mean(bpv_bits_back_arr))
        results_bernoulli.append(np.mean(bpv_bernoulli_arr))
        results_draco.append(np.mean(bpv_draco_arr))
        results_optimal.append(np.mean(bpv_optimal_arr))

    results_bitsback = np.asarray(results_bitsback)
    results_bernoulli = np.asarray(results_bernoulli)
    results_draco = np.asarray(results_draco)
    results_optimal = np.asarray(results_optimal)

    if save_results:
        if dataset_type == 'shape':
            output_dir = os.path.expanduser('~/open3d_data/extract/processed_shapenet/Bit_depth_results/')
        else:
            output_dir = os.path.expanduser('~/open3d_data/extract/processed_sunrgbd/Bit_depth_results/')
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        np.save(output_dir + 'bit_rate_vs_bit_depth_bitsback.npy', results_bitsback)
        np.save(output_dir + 'bit_rate_vs_bit_depth_bernoulli.npy', results_bernoulli)
        np.save(output_dir + 'bit_rate_vs_bit_depth_draco.npy', results_draco)
        np.save(output_dir + 'bit_rate_vs_bit_depth_optimal.npy', results_optimal)

    x_axis = np.log2(depth_values)
    plt.plot(x_axis, results_bitsback, '-^')
    plt.plot(x_axis, results_optimal, '--s')
    plt.plot(x_axis, results_bernoulli, '--o')
    plt.plot(x_axis, results_draco, '-d')
    plt.legend(['Bits-back', 'Optimal', 'No-bits-back', 'Draco'])
    plt.xlabel('Bit depth')
    plt.ylabel('Bit per point')
    plt.grid(linestyle='--')
    plt.show()


def plot_bit_rates(batch_values):
    output_dir = os.path.expanduser('~/open3d_data/extract/processed_shapenet/Bit_rate_results/')
    results_bitsback = np.load(output_dir + 'bit_rate_vs_batch_size_bitsback.npy')
    results_bernoulli = np.load(output_dir + 'bit_rate_vs_batch_size_bernoulli.npy')
    results_draco = np.load(output_dir + 'bit_rate_vs_batch_size_draco.npy')
    results_optimal = np.load(output_dir + 'bit_rate_vs_batch_size_optimal.npy')

    x_axis = np.asarray(batch_values)
    plt.plot(x_axis, results_bitsback, '-^')
    plt.plot(x_axis, results_optimal, '--x')
    plt.plot(x_axis, results_bernoulli, '--o')
    plt.plot(x_axis, results_draco, '-d')
    plt.legend(['Bits-back', 'Optimal', 'No-bits-back', 'Draco'])
    plt.xlabel('Number of point clouds')
    plt.ylabel('Bit per point')
    plt.grid(linestyle='--')
    plt.show()

def plot_bit_depth(depth_values):
    output_dir = os.path.expanduser('~/open3d_data/extract/processed_shapenet/Bit_depth_results/')
    results_bitsback = np.load(output_dir + 'bit_rate_vs_bit_depth_bitsback.npy')
    results_bernoulli = np.load(output_dir + 'bit_rate_vs_bit_depth_bernoulli.npy')
    results_draco = np.load(output_dir + 'bit_rate_vs_bit_depth_draco.npy')
    results_optimal = np.load(output_dir + 'bit_rate_vs_bit_depth_optimal.npy')

    x_axis = np.flip(np.log2(depth_values))
    plt.plot(x_axis, np.flip(results_bitsback), '-^')
    plt.plot(x_axis, np.flip(results_optimal), '--x')
    plt.plot(x_axis, np.flip(results_bernoulli), '--o')
    plt.plot(x_axis, np.flip(results_draco), '-d')
    plt.legend(['Bits-back', 'Optimal', 'No-bits-back', 'Draco'])
    plt.xlabel('Bit depth')
    plt.ylabel('Bit per point')
    plt.grid(linestyle='--')
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
        plot_bit_rates(batch_vals)
    elif args.mode == 'plot_depth':
        depth_vals = [128, 64, 32]
        plot_bit_depth(depth_vals)
    else:
        parser.print_help()