import time
import torch
import craystack as cs
from torch import optim
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from torch.distributions import Bernoulli
from dataset import ShapeNetDataset
from util_functions import *
from models import ConvoVAE


torch.manual_seed(1234)
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print('Device: {}'.format(device))


def train_convo_vae(continual_train=False, n_epochs=50):
    space_shape = [128, 128, 128]
    voxel_size = 2.0 / space_shape[0]
    train_set = ShapeNetDataset(dataset_path='~/open3d_data/extract/ShapeNet/', save_train_test_sets=False,
                                mode='train', device=device)
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, drop_last=True)

    if continual_train:
        model = ConvoVAE(in_dim=space_shape, h_dim=500, latent_dim=50, out_dim=space_shape)
        model.load_state_dict(torch.load('model_params/cvae_params'))
        print('Load pre-trained model ...')
    else:
        model = ConvoVAE(in_dim=space_shape, h_dim=500, latent_dim=50, out_dim=space_shape)

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print('Model: {}'.format(model))
    loss_avg = []

    for epoch in range(n_epochs):
        ep_loss = []
        for batch_id, data in enumerate(train_loader):
            x_batch = get_sparse_voxels_batch(data, voxel_size=voxel_size).to(device)
            optimizer.zero_grad()
            x_batch = torch.unsqueeze(x_batch, 1)
            t0 = time.time()
            loss = model.loss(x_batch)
            t1 = time.time()
            ep_loss.append(loss.item())
            optimizer.step()
            if batch_id % 20 == 0:
                print('\t--- Ep: {}, batch: {}, ep_loss: {}, time: {}'.
                      format(epoch, batch_id, np.mean(ep_loss), t1 - t0))

        loss_avg.append(np.mean(ep_loss))
        print('Epoch: {}, Avg_Loss: {}'.format(epoch, np.mean(ep_loss)))
        # save model
        torch.save(model.state_dict(), 'model_params')
        # save loss figure
        x_axis = np.arange(len(loss_avg))
        plt.plot(x_axis, np.array(loss_avg), '-b')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('images/train_loss.png')


def test_convo_vae():
    print('Test model\n')
    space_shape = [128, 128, 128]
    voxel_size = 2.0 / space_shape[0]
    test_set = ShapeNetDataset(dataset_path='~/open3d_data/extract/ShapeNet/', save_train_test_sets=False,
                               mode='test', device='cpu')
    test_loader = DataLoader(test_set, batch_size=32, shuffle=True, drop_last=True)
    if os.path.isfile('model_params/cvae_params'):
        model = ConvoVAE(in_dim=space_shape, h_dim=500, latent_dim=50, out_dim=space_shape)
        model.load_state_dict(torch.load('model_params', map_location='cpu'))
        print('Model: {}'.format(model))
        model.eval()
    for batch_idx, data in enumerate(test_loader):
        x_batch = get_sparse_voxels_batch(data, voxel_size=voxel_size)
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


if __name__ == '__main__':
    train_convo_vae(continual_train=False, n_epochs=50)
    test_convo_vae()