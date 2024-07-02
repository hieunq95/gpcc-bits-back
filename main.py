import time
import sys
import argparse
import numpy as np
import torch
import craystack as cs
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


def train_convo_vae(train_from_scratch=False, n_epochs=50, voxelization=True, learning_rate=0.001):
    resolution = np.full(3, 128, dtype=np.int32)
    voxel_min_bound = np.full(3, -1.0)
    voxel_max_bound = np.full(3, 1.0)
    voxel_size = (voxel_max_bound[0] - voxel_min_bound[0]) / resolution[0]
    if voxelization:
        train_set = ShapeNetDataset(dataset_path='~/open3d_data/extract/ShapeNet/', save_train_test_sets=False,
                                    mode='train', resolution=resolution, device=device,
                                    crop_min_bound=voxel_min_bound, crop_max_bound=voxel_max_bound,
                                    n_points_per_cloud=20000, n_mesh_per_class=2000, return_voxel=True)
    else:
        train_set = ShapeNetDataset(dataset_path='~/open3d_data/extract/ShapeNet/', save_train_test_sets=False,
                                    mode='train', resolution=resolution, device=device,
                                    crop_min_bound=voxel_min_bound, crop_max_bound=voxel_max_bound,
                                    n_points_per_cloud=20000, n_mesh_per_class=2000, return_voxel=False)

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
            # Use get_sparse_voxels_batch() if we use raw point cloud as training data
            if not voxelization:
                x_batch = get_sparse_voxels_batch(data, voxel_size=voxel_size, voxel_min_bound=voxel_min_bound,
                                                  voxel_max_bound=voxel_max_bound).to(device)
            else:
                x_batch = data
            x_batch = torch.unsqueeze(x_batch, 1)
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
        if epoch == n_epochs:
            torch.save(model.state_dict(), 'model_params/cvae_params')  # final model
        # save loss figure
        x_axis = np.arange(len(loss_avg))
        plt.plot(x_axis, np.array(loss_avg), '-b')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('images/train_loss.png')


def test_convo_vae(voxelization=True, batch_size=32, generate=True, epoch_id=None):
    print('Test model\n')
    resolution = np.full(3, 128, dtype=np.int32)
    voxel_min_bound = np.full(3, -1.0)
    voxel_max_bound = np.full(3, 1.0)
    voxel_size = (voxel_max_bound[0] - voxel_min_bound[0]) / resolution[0]
    test_set = ShapeNetDataset(dataset_path='~/open3d_data/extract/ShapeNet/', save_train_test_sets=False,
                                mode='test', resolution=resolution, device='cpu',
                                crop_min_bound=voxel_min_bound, crop_max_bound=voxel_max_bound,
                                n_points_per_cloud=20000, n_mesh_per_class=1000, return_voxel=voxelization)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)
    if epoch_id is not None:
        f_name = 'model_params/cvae_params_{}'.format(epoch_id)
    else:
        f_name = 'model_params/cvae_params'
    if os.path.isfile(f_name):
        model = ConvoVAE(in_dim=resolution, h_dim=500, latent_dim=50, out_dim=resolution)
        model.load_state_dict(torch.load(f_name, map_location='cpu'))
        print('Model: {}'.format(model))
        model.eval()
    for batch_idx, data in enumerate(test_loader):
        if not voxelization:
            x_batch = get_sparse_voxels_batch(data, voxel_size=voxel_size,
                                              voxel_min_bound=voxel_min_bound, voxel_max_bound=voxel_max_bound)
        else:
            x_batch = data
        x_batch = torch.unsqueeze(x_batch, 1)
        x_probs = model(x_batch)
        gen_probs = model.generate(x_batch.size()[0])
        x_recon = Bernoulli(x_probs).sample()
        x_gen_batch = Bernoulli(gen_probs).sample()

        batch_iou = calculate_iou(x_batch, x_recon)
        print('Batch IoU: {} / batch size: {}'.format(batch_iou, batch_size))

        for j in range(x_batch.size()[0]):
            x_batch_j = torch.squeeze(x_batch[j])
            x_recon_j = torch.squeeze(x_recon[j])
            x_gen_j = torch.squeeze(x_gen_batch[j])
            if not voxelization:
                # Visualize results
                x_ori_vis = data[j].detach().numpy()
                print('Num points: {}'.format(len(x_ori_vis)))
                try:
                    visualize_points(x_ori_vis)
                except KeyboardInterrupt:
                    sys.exit(130)

            if not generate:
                x_vis = x_batch_j.detach().numpy().astype(np.int32)
                print('Num voxels: {}'.format(np.sum(x_vis)))
                try:
                    visualize_voxels(x_vis, voxel_size*40)
                except KeyboardInterrupt:
                    sys.exit()

                x_rec_vis = x_recon_j.detach().numpy().astype(np.int32)
                try:
                    # psnr_i = calculate_psnr(x_vis, x_rec_vis)
                    iou_i = calculate_iou(x_vis, x_rec_vis)
                    acc_i = calculate_accuracy(x_vis, x_rec_vis)
                    print('IoU per voxel: {} / Accuracy per voxel: {}'.format(iou_i, acc_i))
                    visualize_voxels(x_rec_vis, voxel_size*40)
                except KeyboardInterrupt:
                    sys.exit()
            else:
                x_gen_vis = x_gen_j.detach().numpy().astype(np.int32)
                try:
                    visualize_voxels(x_gen_vis, voxel_size*40)
                except KeyboardInterrupt:
                    sys.exit()


def test_compress_methods(batch_size=100, subset_size=10, epoch_id=200, obs_precision=25):
    resolution = np.full(3, 128, dtype=np.int32)
    voxel_min_bound = np.full(3, -1.0)
    voxel_max_bound = np.full(3, 1.0)
    voxel_size = (voxel_max_bound[0] - voxel_min_bound[0]) / resolution[0]
    model = ConvoVAE(in_dim=resolution, h_dim=500, latent_dim=50, out_dim=resolution)
    model.load_state_dict(torch.load('model_params/cvae_params_{}'.format(epoch_id), map_location='cpu'))
    print('Model: {}'.format(model))
    model.eval()
    test_set = ShapeNetDataset(dataset_path='~/open3d_data/extract/ShapeNet/', save_train_test_sets=False,
                                mode='test', resolution=resolution, device='cpu',
                                crop_min_bound=voxel_min_bound, crop_max_bound=voxel_max_bound,
                                n_points_per_cloud=20000, n_mesh_per_class=1000, return_voxel=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

    rec_net = torch_fun_to_numpy_fun(model.encode)
    gen_net = torch_fun_to_numpy_fun(model.decode)
    obs_codec = lambda p: cs.Bernoulli(p, obs_precision)

    for batch_idx, data in enumerate(test_loader):
        # x_batch = get_sparse_voxels_batch(data, voxel_size=voxel_size,
        #                                   voxel_min_bound=voxel_min_bound, voxel_max_bound=voxel_max_bound)
        x_batch = torch.unsqueeze(data, 1)
        bits_back_vae_ans(x_batch, gen_net, rec_net, obs_codec, obs_precision, subset_size)
        bernoulli_ans(x_batch, model, obs_precision, subset_size)

def bernoulli_ans(data, model, obs_precision, subset_size=1):
    # Entropy coding
    data_shape = data.size()
    num_data = data_shape[0]
    # num_voxels = torch.sum(data)
    num_voxels = np.prod(data_shape)
    obs_shape = (subset_size, data_shape[1], data_shape[2], data_shape[3], data_shape[4])
    obs_size = np.prod(obs_shape)
    latent_size = np.prod((subset_size, model.latent_dim))
    codec = lambda p: cs.Bernoulli(p, obs_precision)

    # Encode data using small batches (preventing forward big batch of data -> crash)
    init_message = cs.base_message(obs_size)
    # print('NoBB_ANS init size: {}'.format(init_message[0].shape))
    # split data for using substack
    assert num_data % subset_size == 0
    pop_array = []
    message = init_message
    t0 = time.time()
    data_tuple = torch.split(data, subset_size)
    for x in data_tuple:  # small batches
        p = model(x).detach().numpy().flatten()
        push, pop = codec(p)
        pop_array.append(pop)
        message, = push(message, np.asarray(x.detach().numpy().flatten(), dtype=np.uint8))
    t1 = time.time()
    flat_message = cs.flatten(message)
    pop_size = 0
    for p in pop_array:
        p_size = sys.getsizeof(p) * 32  # in bits
        pop_size += p_size
    flat_message_len = 32 * len(flat_message)
    print('NoBB_VAE -- encoded in {} seconds, bpv: {}, bpv_overhead: {}'.format(
        t1 - t0, flat_message_len / num_voxels, (pop_size + flat_message_len) / num_voxels)
    )

    # Decode message
    t0 = time.time()
    message_ = cs.unflatten(flat_message, obs_size)
    data_decoded = []
    for i in range(len(pop_array)):
        pop = pop_array[-1-i]  # reverse order
        message_, data_ = pop(message_,)
        data_decoded.append(np.asarray(data_, dtype=np.uint8))  # cast dtype to prevent out of memory issue
        # print('data_ shape: {}'.format(data_.shape))
        # print('decoded iter: {}'.format(i))
    t1 = time.time()

    data_decoded = reversed(data_decoded)
    for x, x_ in zip(data_tuple, data_decoded):
        np.testing.assert_equal(x.detach().numpy().flatten(), x_)
    print('NoBB_VAE -- decoded in {} seconds\n'.format(t1 - t0))


def bits_back_vae_ans(data, gen_net, rec_net, obs_codec, obs_precision, subset_size=1):
    def vae_view(head):
        return ag_tuple((np.reshape(head[:latent_size], latent_shape),
                         np.reshape(head[latent_size:], obs_shape)))

    data_shape = data.size()  # [b, 1, 128, 128, 128]
    num_data = data_shape[0]
    # num_voxels = torch.sum(data)
    num_voxels = np.prod(data_shape)
    # print('num_voxels: {}'.format(num_voxels))
    # num_voxels = num_data * 2000
    assert num_data % subset_size == 0
    num_subsets = num_data // subset_size
    latent_dim = 50
    latent_shape = (subset_size, latent_dim)
    latent_size = np.prod(latent_shape)
    obs_shape = (subset_size, data_shape[1], data_shape[2], data_shape[3], data_shape[4])
    obs_size = np.prod(obs_shape)

    print('Bits-back VAE compress {} point clouds...'.format(num_data))

    data = np.split(np.asarray(data.detach().numpy(), dtype=np.uint8), num_subsets)

    # Create codec
    vae_append, vae_pop = cs.repeat(cs.substack(
        bb_ans.VAE(gen_net, rec_net, obs_codec, 8, obs_precision-2),
        vae_view), num_subsets)

    # Encode
    t0 = time.time()
    init_message = cs.base_message(obs_size + latent_size)
    # print('BB_ANS init size: {}'.format(init_message[0].shape))
    message, = vae_append(init_message, data)
    flat_message = cs.flatten(message)
    flat_message_len = 32 * len(flat_message)
    t1 = time.time()
    pop_size = sys.getsizeof(vae_pop) * 32  # in bits
    print('BB_VAE -- encoded in {} seconds, bpv: {}, bpv_overhead: {}'.format(
        t1 - t0, flat_message_len / num_voxels, (pop_size + flat_message_len) / num_voxels)
    )
    np.save(
        os.path.expanduser('~/open3d_data/extract/compressed_shapenet_{}.npy'.format(num_data)),
        flat_message
    )

    # Decode
    t0 = time.time()
    message = cs.unflatten(flat_message, obs_size + latent_size)
    message, data_ = vae_pop(message)
    t1 = time.time()
    print('BB_VAE -- decoded in {} seconds'.format(t1 - t0))
    np.testing.assert_equal(data, data_)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Main script for running GPCC-Bits-back")
    parser.add_argument('--mode', type=str, default='train',
                        help='Evaluation mode: [train, test, compress]')
    parser.add_argument('--ep', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--init', type=int, default=1,
                        help='Only use this when we train from scratch: [0, 1]')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate for the optimizer, e.g., Adam')
    parser.add_argument('--voxel', type=int, default=1,
                        help='Only use this when we train on raw point clouds: [0, 1]')
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size of test set for compression')
    parser.add_argument('--gen', type=int, default=0,
                        help='Use 1 if we want to generate random samples from the model')
    args = parser.parse_args()
    voxel_input = True
    if args.voxel == 0:
        voxel_input = False
    if args.mode == 'train':
        if args.init == 0:
            train_convo_vae(train_from_scratch=False, n_epochs=args.ep, voxelization=voxel_input, learning_rate=args.lr)
        else:
            train_convo_vae(train_from_scratch=True, n_epochs=args.ep, voxelization=voxel_input, learning_rate=args.lr)
    elif args.mode == 'test':
        if args.gen == 1:
            test_convo_vae(voxelization=voxel_input, batch_size=args.batch, generate=True, epoch_id=args.ep)
        else:
            test_convo_vae(voxelization=voxel_input, batch_size=args.batch, generate=False, epoch_id=args.ep)
    elif args.mode == 'compress':
        test_compress_methods(batch_size=args.batch, subset_size=1, epoch_id=args.ep)
    else:
        parser.print_help()