import torch
from q2_sampler import svhn_sampler
from q2_model import Critic, Generator
from torch import optim
from torchvision.utils import save_image
from tqdm import tqdm



def lp_reg(x, y, critic, device):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper. Also consider that the norm used is the L2 norm. This is important to consider,
    because we make the assumption that your implementation follows this notation when testing your function. ***

    :param x: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution P.
    :param y: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution Q.
    :param critic: (Module) - torch module that you want to regularize.
    :return: (FloatTensor) - shape: (1,) - Lipschitz penalty
    """
    t = torch.rand(x.size(), device=device)

    x_hat = t*x + (1 - t)*y
    f_x = critic(x_hat)
    grad = torch.autograd.grad(f_x, x_hat, 
                               retain_graph=True, 
                               create_graph=True, 
                               grad_outputs=torch.ones_like(f_x))[0]

    grad_norm = torch.norm(grad, p=2, dim=-1)
    lp = torch.nn.functional.relu(grad_norm - 1) **2
    return lp.mean()


def vf_wasserstein_distance(p, q, critic):
    """
    COMPLETE ME. DONT MODIFY THE PARAMETERS OF THE FUNCTION. Otherwise, tests might fail.

    *** The notation used for the parameters follow the one from Petzka et al: https://arxiv.org/pdf/1709.08894.pdf
    In other word, x are samples from the distribution mu and y are samples from the distribution nu. The critic is the
    equivalent of f in the paper. This is important to consider, because we make the assuption that your implementation
    follows this notation when testing your function. ***

    :param p: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution p.
    :param q: (FloatTensor) - shape: (batchsize x featuresize) - Samples from a distribution q.
    :param critic: (Module) - torch module used to compute the Wasserstein distance
    :return: (FloatTensor) - shape: (1,) - Estimate of the Wasserstein distance
    """
    return critic(p).mean() - critic(q).mean()


if __name__ == '__main__':
    train = False
    test = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Example of usage of the code provided and recommended hyper parameters for training GANs.
    data_root = './'
    n_iter = 50000 # N training iterations
    n_critic_updates = 5 # N critic updates per generator update
    lp_coeff = 10 # Lipschitz penalty coefficient
    train_batch_size = 64
    test_batch_size = 64
    lr = 1e-4
    beta1 = 0.5
    beta2 = 0.9
    z_dim = 100

    if train:
    
        train_loader, valid_loader, test_loader = svhn_sampler(data_root, train_batch_size, test_batch_size)

        generator = Generator(z_dim=z_dim).to(device)
        critic = Critic().to(device)

        optim_critic = optim.Adam(critic.parameters(), lr=lr, betas=(beta1, beta2))
        optim_generator = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))

        # COMPLETE TRAINING PROCEDURE
        train_iter = iter(train_loader)
        valid_iter = iter(valid_loader)
        test_iter = iter(test_loader)
        for i in tqdm(range(n_iter)):
            generator.train()
            critic.train()
            for _ in range(n_critic_updates):
                try:
                    data = next(train_iter)[0].to(device)
                except Exception:
                    train_iter = iter(train_loader)
                    data = next(train_iter)[0].to(device)
                
                z = torch.randn(train_batch_size, z_dim, device=device)
                gen_data = generator(z)

                # Train loss
                optim_critic.zero_grad()
                loss_critic = -vf_wasserstein_distance(data, gen_data, critic) + lp_coeff*lp_reg(data, gen_data, critic, device)
                loss_critic.backward()
                optim_critic.step()

            # Train loss
            optim_generator.zero_grad()
            gen_data = generator(z)
            loss_gen = -torch.mean(critic(gen_data))
            loss_gen.backward()
            optim_generator.step()

            # Save sample images 
            if i % 100 == 0:
                z = torch.randn(64, z_dim, device=device)
                imgs = generator(z)
                save_image(imgs, f'./imgs/train/imgs_{i}.png', normalize=True, value_range=(-1, 1))

        # Save models
        torch.save(generator, './models/generator.pt')
        torch.save(critic, './models/critic.pt')
    
    if test:
        # COMPLETE QUALITATIVE EVALUATION
        generator = torch.load('./models/generator.pt')
        critic = torch.load('./models/critic.pt')

        # Disentangled representation
        z = torch.randn(test_batch_size, z_dim, device=device)
        original = generator(z)
        save_image(original, f'./imgs/disentangled/original.png', normalize=True, value_range=(-1, 1))

        # Generated images
        eps = 1.0
        for i in tqdm(range(z.shape[1])):
            z_gen = z.clone()
            z_gen[:,i] += eps
            imgs = generator(z_gen)
            save_image(imgs, f'./imgs/disentangled/gen_{i}.png', normalize=True, value_range=(-1, 1))
        
        # Interpolation in latent space and pixel space
        z_1 = torch.randn(test_batch_size, z_dim, device=device)
        z_2 = torch.randn(test_batch_size, z_dim, device=device)
        img_1 = generator(z_1)
        img_2 = generator(z_2)
        save_image(img_1, f'./imgs/interpolation/z_1.png', normalize=True, value_range=(-1, 1))
        save_image(img_2, f'./imgs/interpolation/z_2.png', normalize=True, value_range=(-1, 1))

        alphas = torch.linspace(0, 1, 11)
        for a in tqdm(alphas):
            z_interp = (1 - a)*z_1 + a*z_2
            img_interp = (1 - a)*img_1 + img_2*a 
            save_image(generator(z_interp), './imgs/interpolation/z_interp_{:.2f}.png'.format(a), normalize=True, value_range=(-1, 1))
            save_image(img_interp, './imgs/interpolation/img_interp_{:.2f}.png'.format(a), normalize=True, value_range=(-1, 1))