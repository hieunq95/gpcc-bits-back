import time
import sys
import numpy as np
import torch
import craystack as cs
from craystack import bb_ans
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
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


def train_convo_vae(continual_train=False, n_epochs=50):
    resolution = [128, 128, 128]
    voxel_min_bound = [-0.5, -0.5, -0.5]
    voxel_max_bound = [0.5, 0.5, 0.5]
    voxel_size = (voxel_max_bound[0] - voxel_min_bound[0]) / resolution[0]
    train_set = ShapeNetDataset(dataset_path='~/open3d_data/extract/ShapeNet/', save_train_test_sets=False,
                                mode='train', device=device)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True)

    if continual_train:
        model = ConvoVAE(in_dim=resolution, h_dim=500, latent_dim=50, out_dim=resolution)
        model.load_state_dict(torch.load('model_params/cvae_params', map_location=device))
        print('Load pre-trained model ...')
    else:
        model = ConvoVAE(in_dim=resolution, h_dim=500, latent_dim=50, out_dim=resolution)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print('Model: {}'.format(model))
    loss_avg = []

    for epoch in range(n_epochs):
        ep_loss = []
        for batch_id, data in enumerate(train_loader):
            x_batch = get_sparse_voxels_batch(data, voxel_size=voxel_size, voxel_min_bound=voxel_min_bound,
                                              voxel_max_bound=voxel_max_bound).to(device)
            optimizer.zero_grad()
            x_batch = torch.unsqueeze(x_batch, 1)
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
        torch.save(model.state_dict(), 'model_params/cvae_params')
        # save loss figure
        x_axis = np.arange(len(loss_avg))
        plt.plot(x_axis, np.array(loss_avg), '-b')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('images/train_loss.png')


def test_convo_vae():
    print('Test model\n')
    resolution = [128, 128, 128]
    voxel_min_bound = [-0.5, -0.5, -0.5]
    voxel_max_bound = [0.5, 0.5, 0.5]
    voxel_size = (voxel_max_bound[0] - voxel_min_bound[0]) / resolution[0]
    test_set = ShapeNetDataset(dataset_path='~/open3d_data/extract/ShapeNet/', save_train_test_sets=False,
                               mode='test', device='cpu')
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True, drop_last=True)
    if os.path.isfile('model_params/cvae_params'):
        model = ConvoVAE(in_dim=resolution, h_dim=500, latent_dim=50, out_dim=resolution)
        model.load_state_dict(torch.load('model_params/cvae_params', map_location='cpu'))
        print('Model: {}'.format(model))
        model.eval()
    for batch_idx, data in enumerate(test_loader):
        x_batch = get_sparse_voxels_batch(data, voxel_size=voxel_size,
                                          voxel_min_bound=voxel_min_bound, voxel_max_bound=voxel_max_bound)
        x_batch = torch.unsqueeze(x_batch, 1)
        x_probs = model(x_batch)
        gen_probs = model.generate(x_batch.size()[0])
        x_recon = Bernoulli(x_probs).sample()
        x_gen_batch = Bernoulli(gen_probs).sample()

        for j in range(x_batch.size()[0]):
            x_batch_j = torch.squeeze(x_batch[j])
            x_recon_j = torch.squeeze(x_recon[j])
            x_gen_j = torch.squeeze(x_gen_batch[j])
            # Visualize results
            x_ori_vis = data[j].detach().numpy()
            visualize_points(x_ori_vis)

            x_vis = x_batch_j.detach().numpy()
            visualize_voxels(x_vis)

            x_rec_vis = x_recon_j.detach().numpy()
            visualize_voxels(x_rec_vis)

            x_gen_vis = x_gen_j.detach().numpy()
            visualize_voxels(x_gen_vis)

def test_compress_methods(obs_precision=26):
    resolution = [128, 128, 128]
    voxel_min_bound = [-0.5, -0.5, -0.5]
    voxel_max_bound = [0.5, 0.5, 0.5]
    voxel_size = (voxel_max_bound[0] - voxel_min_bound[0]) / resolution[0]
    model = ConvoVAE(in_dim=resolution, h_dim=500, latent_dim=50, out_dim=resolution)
    model.load_state_dict(torch.load('model_params/cvae_params', map_location='cpu'))
    print('Model: {}'.format(model))
    model.eval()
    test_set = ShapeNetDataset(dataset_path='~/open3d_data/extract/ShapeNet/', save_train_test_sets=False,
                               mode='test', device=device)
    test_loader = DataLoader(test_set, batch_size=200, shuffle=True, drop_last=True)

    rec_net = torch_fun_to_numpy_fun(model.encode)
    gen_net = torch_fun_to_numpy_fun(model.decode)
    obs_codec = lambda p: cs.Bernoulli(p, obs_precision)

    for batch_idx, data in enumerate(test_loader):
        x_batch = get_sparse_voxels_batch(data, voxel_size=voxel_size,
                                          voxel_min_bound=voxel_min_bound, voxel_max_bound=voxel_max_bound)
        x_batch = torch.unsqueeze(x_batch, 1)
        bits_back_vae_ans(x_batch, gen_net, rec_net, obs_codec, obs_precision, 1)
        bernoulli_ans(x_batch, model, obs_precision, 1)

def bernoulli_ans(data, model, obs_precision, subset_size=1):
    # Entropy coding
    data_shape = data.size()
    num_data = data_shape[0]
    num_voxels = np.prod(data_shape)
    obs_shape = (subset_size, data_shape[1], data_shape[2], data_shape[3], data_shape[4])
    obs_size = np.prod(obs_shape)
    codec = lambda p: cs.Bernoulli(p, obs_precision)

    # Encode data using small batches (preventing forward big batch of data -> crash)
    init_message = cs.base_message(obs_shape)
    # print('NoBB_ANS init size: {}'.format(init_message[0].shape))
    # split data for using substack
    assert num_data % subset_size == 0
    pop_array = []
    message = init_message
    t0 = time.time()
    data_tuple = torch.split(data, subset_size)
    for x in data_tuple:  # small batches
        p = model(x).detach().numpy()
        push, pop = codec(p)
        pop_array.append(pop)
        message, = push(message, np.uint64(x.detach().numpy()))
    t1 = time.time()
    flat_message = cs.flatten(message)
    pop_size = 0
    for p in pop_array:
        p_size = sys.getsizeof(p) * 32  # in bits
        pop_size += p_size
    flat_message_len = 32 * len(flat_message)
    print('NoBB_VAE -- encoded in {} seconds, bpv: {}, bpv_overhead: {}'.format(t1 - t0,
        flat_message_len / num_voxels, (pop_size + flat_message_len) / num_voxels))

    # Decode message
    t0 = time.time()
    message_ = cs.unflatten(flat_message, obs_shape)
    data_decoded = []
    for i in range(len(pop_array)):
        pop = pop_array[-1-i]  # reverse order
        message_, data_ = pop(message_,)
        # print('data_ shape: {}'.format(data_.shape))
        data_decoded.append(data_)
    t1 = time.time()
    data_decoded = reversed(data_decoded)
    for x, x_ in zip(data_tuple, data_decoded):
        np.testing.assert_equal(x.detach().numpy(), x_)
    print('NoBB_VAE -- decoded in {} seconds\n'.format(t1 - t0))


def bits_back_vae_ans(data, gen_net, rec_net, obs_codec, obs_precision, subset_size=10):
    def vae_view(head):
        return ag_tuple((np.reshape(head[:latent_size], latent_shape),
                         np.reshape(head[latent_size:], obs_shape)))

    data_shape = data.size()  # [b, 1, 128, 128, 128]
    num_data = data_shape[0]
    num_voxels = num_data * np.prod(data_shape[1:])
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

    data = np.split(data.detach().numpy(), num_subsets)

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
    print('BB_VAE -- encoded in {} seconds, bpv: {}, bpv_overhead: {}'.format(t1 - t0, flat_message_len / num_voxels,
                                                                          (pop_size + flat_message_len) / num_voxels))

    # Decode
    t0 = time.time()
    message = cs.unflatten(flat_message, obs_size + latent_size)
    message, data_ = vae_pop(message)
    t1 = time.time()
    print('BB_VAE -- decoded in {} seconds'.format(t1 - t0))
    np.testing.assert_equal(data, data_)


if __name__ == '__main__':
    train_convo_vae(continual_train=False, n_epochs=50)
    test_convo_vae()
    test_compress_methods()