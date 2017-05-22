from __future__ import print_function
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import math
import numpy as np

import models.dcgan as dcgan
import models.ae as ae
# import models.mlp as mlp

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw ')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nc', type=int, default=3, help='input image channels')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--lrD', type=float, default=0.00005, help='learning rate for Critic, default=0.00005')
parser.add_argument('--lrG', type=float, default=0.00005, help='learning rate for Generator, default=0.00005')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda'  , action='store_true', help='enables cuda')
parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--clamp_lower', type=float, default=-0.01)
parser.add_argument('--clamp_upper', type=float, default=0.01)
parser.add_argument('--Diters', type=int, default=5, help='number of D iters per each G iter')
parser.add_argument('--noBN', action='store_true', help='use batchnorm or not (only for DCGAN)')
parser.add_argument('--mlp_G', action='store_true', help='use MLP for G')
parser.add_argument('--mlp_D', action='store_true', help='use MLP for D')
parser.add_argument('--n_extra_layers', type=int, default=0, help='Number of extra layers on gen and disc')
parser.add_argument('--experiment', default=None, help='Where to store samples and models')
parser.add_argument('--adam', action='store_true', help='Whether to use adam (default is rmsprop)')
parser.add_argument('--lrA', type=float, default=0.00005, help='learning rate for autoencoder')
parser.add_argument('--preTrainN', type=int, default=10, help='No of epochs to pretrain autoencoder')
parser.add_argument('--latent', type=int, default=200, help='Latent vector size')
parser.add_argument('--daeWeight', type=float, default=0.01, help='Weight for loss of autoencoder')
parser.add_argument('--savemodel', default='saved_models', help="location to save model")
parser.add_argument('--gen_iter', type=int, default=0, help='Number of generator iterations to start from')
parser.add_argument('--pretr', action='store_true', help='enable pretraining of autoencoder')
parser.add_argument('--tr_gan', action='store_true', help='enable pretraining of GAN')
parser.add_argument('--enb_dae', action='store_true', help='enable simultaneous training of DAE')
parser.add_argument('--netA', default='', help="path to netA (to continue training)")
parser.add_argument('--start', type=int, default=0, help='Number of epoch to start from')
opt = parser.parse_args()
print(opt)

if opt.experiment is None:
    opt.experiment = 'samples'
os.system('mkdir {0}'.format(opt.experiment))
os.system('mkdir {0}'.format(opt.savemodel))

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['classroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
    )
    test_dataset = dset.CIFAR10(root=opt.dataroot, train=False, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])
                           )
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = int(opt.nc)
n_extra_layers = int(opt.n_extra_layers)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


netG = dcgan.DCGAN_G(opt.imageSize, nz, nc, ngf, ngpu, n_extra_layers)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

netD = dcgan.DCGAN_D(opt.imageSize, nz, nc, ndf, ngpu, n_extra_layers)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

netA = ae.AutoEncoder(nc, 32, opt.latent, ngpu)
netA.apply(weights_init)
if opt.netA != '':
    netA.load_state_dict(torch.load(opt.netA))
print(netA)

input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
ae_input = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
ae_output = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
val_sample = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
cnoise = torch.FloatTensor(opt.batchSize, nc, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
one = torch.FloatTensor([1])
mone = one * -1

criterion = nn.MSELoss()

if opt.cuda:
    netD.cuda()
    netG.cuda()
    netA.cuda()
    criterion.cuda()
    input = input.cuda()
    ae_input = ae_input.cuda()
    ae_output = ae_output.cuda()
    val_sample = val_sample.cuda()
    cnoise = cnoise.cuda()
    one, mone = one.cuda(), mone.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

# setup optimizer
if opt.adam:
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lrD, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lrG, betas=(opt.beta1, 0.999))
    optimizerA = optim.Adam(netA.parameters(), lr=opt.lrA, betas=(opt.beta1, 0.999))
else:
    optimizerD = optim.RMSprop(netD.parameters(), lr=opt.lrD)
    optimizerG = optim.RMSprop(netG.parameters(), lr=opt.lrG)
    optimizerA = optim.RMSprop(netA.parameters(), lr=opt.lrA)


def plot(gen_iterations, epoch):
    fake = netG(Variable(fixed_noise, volatile=True))
    fake.data = fake.data.mul(0.5).add(0.5)
    vutils.save_image(fake.data, '{0}/{1}_after{2}geniters.png'.format(opt.experiment, epoch, gen_iterations),
                      nrow=int(math.sqrt(opt.batchSize)))


def plot_dae(gen_iterations, epoch):
    fake = netG(Variable(fixed_noise, volatile=True))
    rec = netA(Variable(fake.data, volatile=True))
    fake.data = fake.data.mul(0.5).add(0.5)
    rec.data = rec.data.mul(0.5).add(0.5)
    ddata = []
    if opt.cuda:
        fake = fake.data.cpu().numpy()
        rec = rec.data.cpu().numpy()
    for i in range(opt.batchSize):
        ddata.append(fake[i])
        ddata.append(rec[i])
    to_plot = np.asarray(ddata)
    to_plot = torch.from_numpy(to_plot)

    vutils.save_image(to_plot, '{0}/dae_{1}_after{2}geniters.png'.format(opt.experiment, epoch, gen_iterations),
                      nrow=int(math.sqrt(opt.batchSize)))


def corrupt(input_data):
    cnoise.resize_(input_data.size()).normal_(0, 1)
    input_data += cnoise
    cnoise.resize_(input_data.size()).uniform_(0, 1)
    input_data[torch.lt(cnoise, 0.4)] = 0


def plot_dae_rec(gen_iterations, epoch):
    crr_sample = val_sample.clone()
    corrupt(crr_sample)
    rec = netA(Variable(crr_sample, volatile=True))
    if opt.cuda:
        test_sample = crr_sample.cpu().numpy()
        rec = rec.data.cpu().numpy()
    ddata = []
    for i in range(opt.batchSize):
        ddata.append(test_sample[i])
        ddata.append(rec[i])
    to_plot = np.asarray(ddata)
    to_plot = torch.from_numpy(to_plot)

    vutils.save_image(to_plot, '{0}/rec_{1}_after{2}geniters.png'.format(opt.experiment, epoch, gen_iterations),
                      nrow=int(math.sqrt(opt.batchSize)))


def train_ae(data):
    for p in netA.parameters():
        p.requires_grad = True
    netA.zero_grad()
    real_cpu, _ = data
    ae_input.resize_(real_cpu.size()).copy_(real_cpu)
    corrupt(ae_input)
    input.resize_(ae_input.size()).copy_(ae_input)
    ae_output.resize_(real_cpu.size()).copy_(real_cpu)

    inputv = Variable(input)
    label = Variable(ae_output)
    output = netA(inputv)
    err_A = criterion(output, label)
    err_A.backward()
    optimizerA.step()
    return err_A


def train_gan(gen_iterations, i, data_iter):
    for p in netD.parameters():
        p.requires_grad = True

    if gen_iterations < 25 or gen_iterations % 500 == 0:
        Diters = 100
    else:
        Diters = opt.Diters
    j = 0
    while j < Diters and i < len(dataloader):
        j += 1

        for p in netD.parameters():
            p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

        data = data_iter.next()
        i += 1

        netD.zero_grad()
        real_cpu, _ = data
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        input.resize_as_(real_cpu).copy_(real_cpu)
        inputv = Variable(input)
        errD_real = netD(inputv)
        errD_real.backward(one)

        noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise, volatile=True)
        fake = Variable(netG(noisev).data)
        errD_fake = netD(fake)
        errD_fake.backward(mone)
        # errD = errD_real - errD_fake
        optimizerD.step()

    for p in netD.parameters():
        p.requires_grad = False

    netG.zero_grad()
    noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
    noisev = Variable(noise)
    fake = netG(noisev)
    errG = netD(fake)
    errG.backward(one)
    optimizerG.step()
    gen_iterations += 1

    return gen_iterations, i, errD_real, errD_fake


def train_dae_gan(gen_iterations, i, data_iter):
    for p in netD.parameters():
        p.requires_grad = True

    if gen_iterations < 25 or gen_iterations % 500 == 0:
        Diters = 100
    else:
        Diters = opt.Diters
    j = 0
    while j < Diters and i < len(dataloader):
        j += 1

        for p in netD.parameters():
            p.data.clamp_(opt.clamp_lower, opt.clamp_upper)

        data = data_iter.next()
        i += 1

        netD.zero_grad()
        real_cpu, _ = data
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        input.resize_as_(real_cpu).copy_(real_cpu)
        inputv = Variable(input)
        errD_real = netD(inputv)
        errD_real.backward(one)

        noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise, volatile=True)
        fake = Variable(netG(noisev).data)
        errD_fake = netD(fake)
        errD_fake.backward(mone)
        # errD = errD_real - errD_fake
        optimizerD.step()

    for p in netD.parameters():
        p.requires_grad = False

    for p in netA.parameters():
        p.requires_grad = False

    netG.zero_grad()
    noise.resize_(opt.batchSize, nz, 1, 1).normal_(0, 1)
    noisev = Variable(noise)
    fake_v = netG(noisev)

    fake = Variable(fake_v.data, requires_grad=True)
    errG = netD(fake)
    rec = netA(fake)
    newd = rec.data - fake.data
    errG.backward()
    fake_v.backward(fake.grad.data + (-opt.daeWeight * newd))
    optimizerG.step()
    gen_iterations += 1

    return gen_iterations, i, errD_real, errD_fake


if opt.pretr:
    for ii in range(opt.preTrainN):
        mse = 0
        for i, data in enumerate(dataloader):
            mse += train_ae(data)
        print("DAE MSE = {0} in {1} iterations".format(mse.data[0], ii))
    torch.save(netA.state_dict(), '{0}/netA_pretr.pth'.format(opt.savemodel))


gen_iterations = opt.gen_iter
train_errG, train_errD = [], []
titer = iter(test_dataloader)
tdata = titer.next()
sample, _ = tdata
if opt.cuda:
    sample = sample.cuda()
val_sample.resize_as_(sample).copy_(sample)
for epoch in range(opt.start, opt.niter):
    data_iter = iter(dataloader)
    i = 0
    err_D, err_G = 0, 0
    mse = 0
    giter = 0
    while i < len(dataloader):
        if epoch < 10 and opt.tr_gan:
            gen_iterations, i, errD, errG = train_gan(gen_iterations, i, data_iter)
        else:
            gen_iterations, i, errD, errG = train_dae_gan(gen_iterations, i, data_iter)
        err_D += errD
        err_G += errG
        giter += 1
    if epoch < 10 or opt.enb_dae:
        for mse_iter, data in enumerate(dataloader):
            mse += train_ae(data)

    print('[%d/%d][%d] Loss_D: %f Loss_G: %f MSE_Loss: %f'
        % (epoch, opt.niter, gen_iterations, err_D.data[0]/giter, err_G.data[0]/giter, mse.data[0]/mse_iter))
    train_errG.append(err_G.data[0] / giter)
    train_errD.append(err_D.data[0] / giter)

    plot(gen_iterations, epoch)
    plot_dae(gen_iterations, epoch)
    plot_dae_rec(gen_iterations, epoch)

    torch.save(netG.state_dict(), '{0}/netG_epoch_{1}.pth'.format(opt.savemodel, epoch))
    torch.save(netD.state_dict(), '{0}/netD_epoch_{1}.pth'.format(opt.savemodel, epoch))
    torch.save(netA.state_dict(), '{0}/netA_epoch_{1}.pth'.format(opt.savemodel, epoch))
