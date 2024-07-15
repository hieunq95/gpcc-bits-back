import time
import os
import sys
import dill
import zlib
import lzma
import argparse
import numpy as np
import torch
import open3d as o3d
import craystack as cs
import matplotlib.pyplot as plt
from craystack import bb_ans
from torch import optim
from torch.utils.data import DataLoader
from torch.distributions import Bernoulli
from autograd.builtins import tuple as ag_tuple
from dataset import ShapeNetDataset
from util_functions import *
from models import ConvoVAE

rng = np.random.RandomState(0)
torch.manual_seed(1234)
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print('Device: {}'.format(device))

def draco_ans(point_clouds, voxel_batch, voxel_size, voxel_min_bound, voxel_max_bound):
    draco_results_dir = os.path.expanduser('~/open3d_data/extract/processed_shapenet/Draco_results/')
    data = torch.squeeze(point_clouds)
    t0 = time.time()
    compressed_sizes, compressed_fnames, raw_point_clouds = draco_compress(
        data, draco_results_dir, 6
    )
    t1 = time.time()
    flat_message_len = np.sum(compressed_sizes) * 8  # maybe we also need to calculate overhead of Draco's decoder?
    num_voxels = torch.sum(voxel_batch)
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
    num_voxels = torch.sum(voxel_batch)
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
    data_tuple = torch.split(voxel_batch, subset_size)
    point_tuple = torch.split(point_clouds, subset_size)

    for x in data_tuple:  # small batches
        p = model(x).detach().numpy().flatten()
        push, pop = codec(p)
        pop_array.append(pop)
        message, = push(message, np.asarray(x.detach().numpy().flatten(), dtype=np.uint8))
    t1 = time.time()
    flat_message = cs.flatten(message)
    codec_compressor = lzma.LZMACompressor()
    pop_size = 0
    for pf in pop_array:
        serialized_pop = dill.dumps(pf)
        pfc = codec_compressor.compress(serialized_pop)
        pop_size += len(pfc) * 8
    codec_compressor.flush()
    pop_size = min(pop_size, 4 * 10**6 * 8)   # compare size of the Codec with size of the deep learning model
    flat_message_len = 32 * len(flat_message)
    bpv_overhead = (pop_size + flat_message_len) / num_voxels
    bpv = flat_message_len / num_voxels
    print('--- NoBB_VAE -- encoded in {} seconds, bpv: {}, bpv_overhead: {}'.format(
        t1 - t0, bpv, bpv_overhead)
    )
    save_dir = os.path.expanduser('~/open3d_data/extract/processed_shapenet/Bernoulli_results/')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    np.save(
        os.path.expanduser(save_dir + 'Bernoulli_shapenet_{}.npy'.format(num_data)),
        flat_message
    )

    # Decode message
    t0 = time.time()
    message_ = cs.unflatten(flat_message, obs_size)
    data_decoded = []
    for i in range(len(pop_array)):
        pop = pop_array[-1 - i]  # reverse order
        message_, data_ = pop(message_, )
        data_decoded.append(np.asarray(data_, dtype=np.uint8))  # cast dtype to prevent out of memory issue
    t1 = time.time()

    data_decoded = reversed(data_decoded)
    # Check quality
    iou_arr = []
    for x, x_, pb in zip(data_tuple, data_decoded, point_tuple):
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
    return bpv_overhead, bpv

def bits_back_vae_ans(point_clouds, voxel_batch, voxel_size, voxel_min_bound, voxel_max_bound,
                      gen_net, rec_net, obs_codec, obs_precision, subset_size=1):
    """
    Compress a batch of voxelized point clouds
    :return: Compression ratio (bit rate including overhead of the Codec), decompressed voxels
    """
    def vae_view(head):
        return ag_tuple((np.reshape(head[:latent_size], latent_shape),
                         np.reshape(head[latent_size:], obs_shape)))

    data_shape = voxel_batch.size()
    num_data = data_shape[0]
    num_voxels = torch.sum(voxel_batch)
    assert num_data % subset_size == 0
    num_subsets = num_data // subset_size
    latent_dim = 50
    latent_shape = (subset_size, latent_dim)
    latent_size = np.prod(latent_shape)
    obs_shape = (subset_size, data_shape[1], data_shape[2], data_shape[3], data_shape[4])
    obs_size = np.prod(obs_shape)

    data = np.split(np.asarray(voxel_batch.detach().numpy(), dtype=np.uint8), num_subsets)

    # Create codec
    vae_append, vae_pop = cs.repeat(cs.substack(
        bb_ans.VAE(gen_net, rec_net, obs_codec, 8, obs_precision - 2),
        vae_view), num_subsets)

    # Encode
    t0 = time.time()
    init_message = cs.base_message(obs_size + latent_size)
    # print('BB_ANS init size: {}'.format(init_message[0].shape))
    message, = vae_append(init_message, data)
    flat_message = cs.flatten(message)
    flat_message_len = 32 * len(flat_message)
    t1 = time.time()
    codec_compressor = lzma.LZMACompressor()
    compressed_pop = codec_compressor.compress(dill.dumps(vae_pop))
    codec_compressor.flush()
    pop_size = len(compressed_pop) * 8
    bpv_overhead = (pop_size + flat_message_len) / num_voxels
    print('--- BB_VAE -- encoded in {} seconds, bpv: {}, bpv_overhead: {}'.format(
        t1 - t0, flat_message_len / num_voxels, bpv_overhead)
    )
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
    message, data_ = vae_pop(message)
    t1 = time.time()

    np.testing.assert_equal(data, data_)  # Check lossless compression
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

    return bpv_overhead, data_

def train_convo_vae(train_from_scratch=False, n_epochs=50, learning_rate=0.001, resolution=64):
    resolution = np.full(3, resolution, dtype=np.int32)
    voxel_min_bound = np.full(3, -1.0)
    voxel_max_bound = np.full(3, 1.0)
    voxel_size = (voxel_max_bound[0] - voxel_min_bound[0]) / resolution[0]
    train_set = ShapeNetDataset(dataset_path='~/open3d_data/extract/ShapeNet/', make_new_dataset=False,
                                mode='train', resolution=resolution, device=device,
                                crop_min_bound=voxel_min_bound, crop_max_bound=voxel_max_bound,
                                n_points_per_cloud=20000, n_mesh_per_class=3000)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True)

    if not train_from_scratch:
        model = ConvoVAE(in_dim=resolution, h_dim=500, latent_dim=50, out_dim=resolution)
        if os.path.isfile('model_params/cvae_params'):
            model.load_state_dict(torch.load('model_params/cvae_params', map_location=device))
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
        if epoch % 20 == 0:
            torch.save(model.state_dict(), 'model_params/cvae_params_{}'.format(epoch))
        torch.save(model.state_dict(), 'model_params/cvae_params')  # final model
        # save loss figure
        x_axis = np.arange(len(loss_avg))
        plt.plot(x_axis, np.array(loss_avg), '-b')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('images/train_loss.png')

def test_convo_vae(batch_size=32, generate=True, epoch_id=0, resolution=64):
    print('Test model\n')
    resolution = np.full(3, resolution, dtype=np.int32)
    voxel_min_bound = np.full(3, -1.0)
    voxel_max_bound = np.full(3, 1.0)
    voxel_size = (voxel_max_bound[0] - voxel_min_bound[0]) / resolution[0]
    test_set = ShapeNetDataset(dataset_path='~/open3d_data/extract/ShapeNet/', make_new_dataset=False,
                               mode='test', resolution=resolution, device='cpu',
                               crop_min_bound=voxel_min_bound, crop_max_bound=voxel_max_bound,
                               n_points_per_cloud=20000, n_mesh_per_class=3000)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)
    if epoch_id != 0:
        f_name = 'model_params/cvae_params_{}'.format(epoch_id)
    else:
        f_name = 'model_params/cvae_params'
    if os.path.isfile(f_name):
        model = ConvoVAE(in_dim=resolution, h_dim=500, latent_dim=50, out_dim=resolution)
        model.load_state_dict(torch.load(f_name, map_location='cpu'))
        print('Model: {}'.format(model))
        model.eval()

    rec_net = torch_fun_to_numpy_fun(model.encode)
    gen_net = torch_fun_to_numpy_fun(model.decode)
    obs_codec = lambda p: cs.Bernoulli(p, 25)

    for batch_idx, data in enumerate(test_loader):
        x_batch = get_sparse_voxels_batch(data, voxel_size=voxel_size,
                                          voxel_min_bound=voxel_min_bound, voxel_max_bound=voxel_max_bound)
        x_batch = torch.unsqueeze(x_batch, 1)
        x_probs = model(x_batch)
        gen_probs = model.generate(x_batch.size()[0])
        x_recon = Bernoulli(x_probs).sample()
        x_gen_batch = Bernoulli(gen_probs).sample()
        # Compress
        bpv_bits_back, decoded_voxels = bits_back_vae_ans(data, x_batch, voxel_size, voxel_min_bound, voxel_max_bound,
                                                          gen_net, rec_net, obs_codec, 25, 1)
        print('Compress {} batches of voxels with BB_ANS: {} bpv'.format(batch_size, bpv_bits_back))
        batch_iou = []
        for j in range(x_batch.size()[0]):
            batch_iou.append(calculate_iou(torch.squeeze(x_batch[j]), torch.squeeze(x_recon[j])))
        print('Batch IoU mean: {} - std: {} / batch size: {}'.format(
            np.mean(batch_iou), np.std(batch_iou), batch_size)
        )

        for j in range(x_batch.size()[0]):
            x_batch_j = torch.squeeze(x_batch[j])
            x_recon_j = torch.squeeze(x_recon[j])
            x_gen_j = torch.squeeze(x_gen_batch[j])
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
                    visualize_voxels(x_vis, voxel_size*40)
                except KeyboardInterrupt:
                    sys.exit()

                x_rec_vis = x_recon_j.detach().numpy().astype(np.int32)
                try:
                    iou_i = calculate_iou(x_vis, x_rec_vis)
                    acc_i = calculate_accuracy(x_vis, x_rec_vis)
                    print('IoU per voxel: {} / Accuracy per voxel: {}'.format(iou_i, acc_i))
                    visualize_voxels(x_rec_vis, voxel_size*40)
                except KeyboardInterrupt:
                    sys.exit()

                x_dec_vis = x_decoded_j.astype(np.int32)
                try:
                    visualize_voxels(x_dec_vis, voxel_size * 40)
                except KeyboardInterrupt:
                    sys.exit()
            else:
                x_gen_vis = x_gen_j.detach().numpy().astype(np.int32)
                try:
                    visualize_voxels(x_gen_vis, voxel_size*40)
                except KeyboardInterrupt:
                    sys.exit()

def eval_bit_rates(batch_values, subset_size=10, epoch_id=200, obs_precision=25, save_results=True):
    resolution = np.full(3, 64, dtype=np.int32)
    voxel_min_bound = np.full(3, -1.0)
    voxel_max_bound = np.full(3, 1.0)
    voxel_size = (voxel_max_bound[0] - voxel_min_bound[0]) / resolution[0]
    model = ConvoVAE(in_dim=resolution, h_dim=500, latent_dim=50, out_dim=resolution)
    if epoch_id != 0:
        f_name = 'model_params/cvae_params_{}'.format(epoch_id)
    else:
        f_name = 'model_params/cvae_params'
    model.load_state_dict(torch.load(f_name, map_location='cpu'))
    print('Model: {}'.format(model))
    model.eval()

    rec_net = torch_fun_to_numpy_fun(model.encode)
    gen_net = torch_fun_to_numpy_fun(model.decode)
    obs_codec = lambda p: cs.Bernoulli(p, obs_precision)

    test_set = ShapeNetDataset(dataset_path='~/open3d_data/extract/ShapeNet/', make_new_dataset=False,
                               mode='test', resolution=resolution, device='cpu',
                               crop_min_bound=voxel_min_bound, crop_max_bound=voxel_max_bound,
                               n_points_per_cloud=20000, n_mesh_per_class=3000)

    results_bitsback, results_bernoulli, results_draco, results_optimal = [], [], [], []
    for batch_size in batch_values:
        print('Evaluate bit rates of compression methods on {} point clouds per batch...'.format(batch_size))
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)
        bpv_bits_back_arr, bpv_bernoulli_arr, bpv_draco_arr, bpv_optimal_arr = [], [], [], []
        for batch_idx, data in enumerate(test_loader):
            if batch_idx > 2:
                break
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
                data, x_batch, voxel_size, voxel_min_bound, voxel_max_bound
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
        output_dir = os.path.expanduser('~/open3d_data/extract/processed_shapenet/Bit_rate_results/')
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
    plt.ylabel('Bit per voxel')
    plt.grid(linestyle='--')
    plt.show()
    # Visualize point clouds

def evaluate_bit_depth():
    pass

def plot_results():
    batch_values = [100 * i for i in [2, 4, 6, 8, 10, 12]]
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
    plt.xlabel('Batch size')
    plt.ylabel('Bit per voxel')
    plt.grid(linestyle='--')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Main script for running GPCC-bits-back")
    parser.add_argument('--mode', type=str, default='train',
                        help='Evaluation mode: [train, test, compress]')
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
    args = parser.parse_args()
    if args.mode == 'train':
        if args.init == 0:
            train_convo_vae(train_from_scratch=False, n_epochs=args.ep, learning_rate=args.lr, resolution=args.res)
        else:
            train_convo_vae(train_from_scratch=True, n_epochs=args.ep, learning_rate=args.lr, resolution=args.res)
    elif args.mode == 'test':
        if args.gen == 1:
            test_convo_vae(batch_size=args.batch, generate=True, epoch_id=args.ep, resolution=args.res)
        else:
            test_convo_vae(batch_size=args.batch, generate=False, epoch_id=args.ep, resolution=args.res)
    elif args.mode == 'compress':
        batch_values = [100 * i for i in [2, 4, 6, 8, 10, 12]]
        eval_bit_rates(batch_values=batch_values, subset_size=1, epoch_id=args.ep, save_results=True)
    elif args.mode == 'plot':
        plot_results()
    else:
        parser.print_help()