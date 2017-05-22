import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, nc, naf, latent, ngpu):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        enc = nn.Sequential()
        enc.add_module('conv_layer_{0}_{1}'.format(nc, naf),
                       nn.Conv2d(nc, naf, 4, 2, 1, bias=False))
        enc.add_module('batch_norm_{0}'.format(naf), nn.BatchNorm2d(naf))
        enc.add_module('leaky_relu_{0}'.format(naf), nn.LeakyReLU(0.2, inplace=True))
        enc.add_module('conv_layer_{0}_{1}'.format(naf, naf*2),
                       nn.Conv2d(naf, naf*2, 4, 2, 1, bias=False))
        enc.add_module('batch_norm_{0}'.format(naf*2), nn.BatchNorm2d(naf*2))
        enc.add_module('leaky_relu_{0}'.format(naf*2), nn.LeakyReLU(0.2, inplace=True))
        enc.add_module('conv_layer_{0}_{1}'.format(naf*2, naf*4),
                       nn.Conv2d(naf*2, naf*4, 4, 2, 1, bias=False))
        enc.add_module('batch_norm_{0}'.format(naf*4), nn.BatchNorm2d(naf*4))
        enc.add_module('leaky_relu_{0}'.format(naf*4), nn.LeakyReLU(0.2, inplace=True))
        enc.add_module('conv_layer_{0}_{1}'.format(naf*4, latent),
                       nn.Conv2d(naf*4, latent, 4, 2, 1, bias=False))
        enc.add_module('batch_norm_{0}'.format(latent), nn.BatchNorm2d(latent))
        enc.add_module('leaky_relu_{0}'.format(latent), nn.LeakyReLU(0.2, inplace=True))

        self.enc = enc

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.enc, input, range(self.ngpu))
        else:
            output = self.enc(input)

        return output

class Decoder(nn.Module):
    def __init__(self, nc, naf, latent, ngpu):
        super(Decoder, self).__init__()
        self.ngpu = ngpu

        dec = nn.Sequential()
        dec.add_module('convt_layer_{0}_{1}'.format(latent, naf*4),
                       nn.ConvTranspose2d(latent, naf*4, 4, 2, 1, bias=False))
        dec.add_module('batch_norm_{0}'.format(naf*4), nn.BatchNorm2d(naf*4))
        dec.add_module('leaky_relu_{0}'.format(naf*4), nn.LeakyReLU(0.2, inplace=True))
        dec.add_module('convt_layer_{0}_{1}'.format(naf*4, naf*2),
                       nn.ConvTranspose2d(naf*4, naf*2, 4, 2, 1, bias=False))
        dec.add_module('batch_norm_{0}'.format(naf*2), nn.BatchNorm2d(naf*2))
        dec.add_module('leaky_relu_{0}'.format(naf*2), nn.LeakyReLU(0.2, inplace=True))
        dec.add_module('convt_layer_{0}_{1}'.format(naf*2, naf),
                       nn.ConvTranspose2d(naf*2, naf, 4, 2, 1, bias=False))
        dec.add_module('batch_norm_{0}'.format(naf), nn.BatchNorm2d(naf))
        dec.add_module('leaky_relu_{0}'.format(naf), nn.LeakyReLU(0.2, inplace=True))
        dec.add_module('convt_layer_{0}_{1}'.format(naf, nc),
                       nn.ConvTranspose2d(naf, nc, 4, 2, 1, bias=False))

        self.dec = dec

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.dec, input, range(self.ngpu))
        else:
            output = self.dec(input)

        return output

class AutoEncoder(nn.Module):
    def __init__(self, nc, naf, latent, ngpu):
        super(AutoEncoder, self).__init__()
        self.ngpu = ngpu

        ae = nn.Sequential()
        ae.add_module('Encoder', Encoder(nc, naf, latent, ngpu))
        ae.add_module('Decoder', Decoder(nc, naf, latent, ngpu))

        self.ae = ae

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.ae, input, range(self.ngpu))
        else:
            output = self.ae(input)

        return output
