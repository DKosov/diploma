from tqdm import tqdm

import torch
from torch import optim
from torch.autograd import grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from src.datasets import MusicWithFrames
from src.nn import Generator, Discriminator


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def imagefolder_loader(path):
    def loader(transform):
        data = datasets.ImageFolder(path, transform=transform)
        data_loader = DataLoader(data, shuffle=True, batch_size=batch_size,
                                 num_workers=4)
        return data_loader
    return loader


def music_video_loader(path=""):
    def loader(transform):
        data = MusicWithFrames(folder=path, transform=transform, frame_batch=frame_batch)
        data_loader = DataLoader(data, shuffle=True, batch_size=batch_size,
                                 num_workers=4)
        return data_loader
    return loader


def sample_data(dataloader, image_size=4):
    transform = transforms.Compose([
        transforms.Resize(image_size + int(image_size * 0.2) + 1),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    loader = dataloader(transform)
    return loader



def train(generator, discriminator, init_step, loader, total_iter=600000):
    step = init_step
    data_loader = sample_data(loader, 4 * 2 ** step)
    dataset = iter(data_loader)
    total_iter_remain = total_iter - (total_iter//6)*(step-1)
    pbar = tqdm(range(total_iter_remain))

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    alpha = 0
    one = torch.tensor(1).to(device).float()
    mone = torch.tensor(-1).to(device).float()
    iteration = 0
    for i in pbar:
        discriminator.zero_grad()

        alpha = min(1, (2 / (total_iter // 6)) * iteration)

        if iteration > total_iter//6:
            alpha = 0
            iteration = 0
            step += 1

            if step > 6:
                alpha = 1
                step = 6
            data_loader = sample_data(loader, 4 * 2 ** step)
            dataset = iter(data_loader)

        try:
            real_image, label, noise_vector = next(dataset)

        except (OSError, StopIteration):
            dataset = iter(data_loader)
            real_image, label, noise_vector = next(dataset)
        iteration += 1

        ### train Discriminator
        b_size = real_image.size(0)
        real_image = real_image.to(device)
        label = label.to(device).float()
        noise_vector = noise_vector.to(device).float()

        real_predict = discriminator(real_image, step=step, alpha=alpha)
        real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
        real_predict.backward(mone)

        gen_z = torch.cat((label, torch.randn(b_size, 128).to(device)), dim=1) # 
        fake_image = generator(gen_z, step=step, alpha=alpha)
        fake_predict = discriminator(
            fake_image.detach(), step=step, alpha=alpha)
        fake_predict = fake_predict.mean()
        fake_predict.backward(one)

        eps = torch.rand(b_size, 1, 1, 1).to(device)
        x_hat = eps * real_image.data + (1 - eps) * fake_image.detach().data
        x_hat.requires_grad = True
        hat_predict = discriminator(x_hat, step=step, alpha=alpha)
        grad_x_hat = grad(
            outputs=hat_predict.sum(), inputs=x_hat, create_graph=True)[0]
        grad_penalty = ((grad_x_hat.view(grad_x_hat.size(0), -1)
                         .norm(2, dim=1) - 1)**2).mean()
        grad_penalty = 10 * grad_penalty
        grad_penalty.backward()
        grad_loss_val += grad_penalty.item()
        disc_loss_val += (real_predict - fake_predict).item()

        d_optimizer.step()

        ### train Generator
        if (i + 1) % n_critic == 0:
            generator.zero_grad()
            discriminator.zero_grad()
            
            predict = discriminator(fake_image, step=step, alpha=alpha)

            loss = -predict.mean()
            gen_loss_val += loss.item()


            loss.backward()
            g_optimizer.step()
            accumulate(g_running, generator)
 
        if (i + 1) % 10000 == 0 or i==0:
            try:
                torch.save(g_running.state_dict(), './checkpoint/'+str(i + 1).zfill(6)+'_g.model')
                torch.save(discriminator.state_dict(), './checkpoint/'+str(i + 1).zfill(6)+'_d.model')
            except:
                pass

        if (i + 1) % 500 == 0:
            disc_loss_val = 0
            gen_loss_val = 0
            grad_loss_val = 0


if __name__ == '__main__':
    lr = 0.001
    path = './data'
    output_folder = './output'
    input_code_size = 256
    channel = 128
    batch_size = 8
    frame_batch = 1
    n_critic = 1
    init_step = 1
    total_iter = 100000
    pixel_norm = False
    tanh = False

    device = torch.device("cuda:0")

    generator = Generator(in_channel=channel, out_channel= frame_batch*3, input_code_dim=input_code_size, pixel_norm=pixel_norm, tanh=tanh).to(device)
    discriminator = Discriminator(feat_dim=channel, in_channels=frame_batch*3).to(device)
    g_running = Generator(in_channel=channel, out_channel= frame_batch*3, input_code_dim=input_code_size, pixel_norm=pixel_norm, tanh=tanh).to(device)
    g_running.train(False)

    g_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(0.0, 0.99))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.0, 0.99))

    accumulate(g_running, generator, 0)
    loader = music_video_loader(path=path)

    train(generator, discriminator, init_step, loader, total_iter)
