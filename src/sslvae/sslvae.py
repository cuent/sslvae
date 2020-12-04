"""
This file implements SSLVAE-M2 as described in:

Kingma, D. P., Mohamed, S., Jimenez Rezende, D., & Welling, M. (2014).
Semi-supervised learning with deep generative models.
Advances in neural information processing systems, 27, 3581-3589.
https://arxiv.org/pdf/1406.5298.pdf
"""
import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
import pytorch_lightning as pl
from .utils import make_mnist_semi_super
from pytorch_lightning.metrics import Accuracy


class Encoder(nn.Module):
    def __init__(self, inputsz, numclass, hiddensz, embeddingsz):
        '''
        Encoder for q(z|x,y)

        :param inputsz: input size
        :param numclass: number of classes
        :param hiddensz: hidden size
        :param embeddingsz: embedding size for z
        '''
        super().__init__()

        base = nn.Sequential(
            nn.Linear(inputsz + numclass, hiddensz),
            nn.Softplus()
        )
        self.mean = nn.Sequential(
            base,
            nn.Linear(hiddensz, embeddingsz)
        )

        self.logvar = nn.Sequential(
            base,
            nn.Linear(hiddensz, embeddingsz),
        )

    def forward(self, x, y):
        xy = torch.cat((x, y), dim=1)
        mean = self.mean(xy)
        logvar = self.logvar(xy)
        return mean, logvar


class Decoder(nn.Module):
    def __init__(self, inputsz, numclass, hiddensz, embeddingsz, decoder_type):
        '''
        Decoder for p(x|z,y)

        :param inputsz: input size of x
        :param numclass: number of classes
        :param hiddensz: hidden units size
        :param embeddingsz: output size
        :param decoder_type: 'bernoulli'
        '''
        super(Decoder, self).__init__()

        self.decoder_type = decoder_type
        if decoder_type == 'bernoulli':
            self.dec = nn.Sequential(
                nn.Linear(numclass + embeddingsz, hiddensz),
                nn.Softplus(),
                nn.Linear(hiddensz, inputsz),
                nn.Sigmoid()
            )

    def forward(self, z, y):
        zy = torch.cat((z, y), dim=1)
        return self.dec(zy)

class M2(nn.Module):
    def __init__(self, inputsz, numclass, hiddensz, embeddingsz, decoder_type, alpha):
        '''
        SSLVAE module.

        :param inputsz: input size of x
        :param numclass: number of classes
        :param hiddensz: size of hidden units
        :param embeddingsz: dimension of latent variable z
        :param decoder_type: 'bernoulli'
        :param alpha: rate to control generative and discriminative learning
        '''
        super().__init__()

        self.numclass = numclass
        self.loss_fn = nn.CrossEntropyLoss
        self.alpha = alpha

        self.encoder = Encoder(inputsz, numclass, hiddensz, embeddingsz)
        self.decoder = Decoder(inputsz, numclass, hiddensz, embeddingsz, decoder_type)
        self.clf = nn.Sequential(
            nn.Linear(inputsz, hiddensz),
            nn.Softplus(),
            nn.Linear(hiddensz, numclass),
            nn.Softmax()
        )

        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, y):
        mean, logvar = self.encoder(x, y)  # q(z|x,y)
        z = self.reparameterize(mean, logvar)  # sample z
        p = self.decoder(z, y)  # p(x|z,y)
        # pi = self.clf(x) # q(y|x)
        return mean, logvar, p

    def labelled_loss(self, x, xc, mu, logvar):
        if self.decoder.decoder_type == 'bernoulli':
            log_px = -F.binary_cross_entropy(xc, x, reduction='none').sum(1, keepdims=True)
        log_py = torch.log(torch.zeros((x.shape[0], 1)) + 1 / self.numclass) # log p(y)
        kl_qz_pz = 0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), 1, keepdims=True)  # - KL(q(z|x,y) || p(z))

        return -(log_px + log_py + kl_qz_pz)

    def unlabelled_loss(self, x, logits_u):
        loss_u = torch.zeros((x.shape[0], 1))
        for i in range(self.numclass):
            y = torch.zeros(x.shape[0], self.numclass)
            y[:, i] = 1

            mean, logvar, p = self.forward(x, y)
            loss_l = self.labelled_loss(x, p, mean, logvar)

            q_y = logits_u.gather(1, torch.argmax(y, axis=1, keepdim=True))
            loss_y = q_y * loss_l
            loss_u += loss_y + q_y * torch.log(q_y+1e-10) #- torch.distributions.Categorical(q_y).entropy()

        return loss_u

    def loss(self, xl, yl, p_l, logits_l, mu_l, logvar_l, xu, logits_u):
        ll = self.labelled_loss(xl, p_l, mu_l, logvar_l)
        ul = self.unlabelled_loss(xu, logits_u)
        l = ll + ul

        l_alpha = l + (self.alpha * xl.shape[0]*2) * F.cross_entropy(logits_l, torch.argmax(yl, 1), reduction='mean')
        return l_alpha.mean(), ll.mean(), ul.mean()



class SSLVAE(pl.LightningModule):
    def __init__(self, params):
        super(SSLVAE, self).__init__()

        self.model = M2(**params.network)
        self.accuracy = Accuracy()
        self.params = params

    def forward(self, xl, yl, xu):
        mu_l, logvar_l, p_l = self.model(xl, yl)
        pi_l = self.model.clf(xl)
        pi_u = self.model.clf(xu)
        return mu_l, logvar_l, p_l, pi_l, pi_u

    def training_step(self, batch, batch_idx):
        xl, yl, xu, yu= batch
        mu_l, logvar_l, p_l, pi_l, pi_u = self(xl, yl, xu)
        loss,l_loss, u_loss = self.model.loss(xl, yl, p_l, pi_l, mu_l, logvar_l, xu, pi_u)

        self.log('train loss', loss)
        self.log('train s-loss', l_loss)
        self.log('train u-loss', u_loss)

        return {
            'loss': loss,
        }

    def validation_step(self, batch, batch_idx):
        xl, yl, xu, yu = batch
        mu_l, logvar_l, p_l, pi_l, pi_u = self.forward(xl, yl, xu)
        loss, l_loss, u_loss = self.model.loss(xl, yl, p_l, pi_l, mu_l, logvar_l, xu, pi_u)

        self.log('val loss', loss)
        self.log('val s-loss', l_loss)
        self.log('val u-loss', u_loss)
        self.log('val accuracy step', self.accuracy(pi_u, yu), prog_bar=True)

        return {'val_loss': loss,
                'lbl': f"true: {yl[0].argmax().item()} - pred: {pi_l[0].argmax().item()}",
                'img_sample': torch.distributions.binomial.Binomial(total_count=1, probs=p_l[0]).sample().reshape(-1,28,28),
                'img_p': p_l[0].reshape(-1, 28, 28),
                }

    def validation_step_end(self, outputs):
        label = outputs['lbl']
        img1 = outputs['img_sample']
        img2 = outputs['img_p']

        self.log('val accuracy epoc', self.accuracy.compute())
        self.logger.experiment.add_image(f"{label}_sample", img1)
        self.logger.experiment.add_image(f"{label}_prob", img2)

        return outputs['val_loss']

    def test_step(self, batch, batch_idx):
        xl, yl, xu = batch
        mu_l, logvar_l, p_l, pi_l, pi_u = self.forward(xl, yl, xu)
        loss, l_loss, u_loss = self.model.loss(xl, yl, p_l, pi_l, mu_l, logvar_l, xu, pi_u)

        self.log('test loss', loss)
        self.log('test supervised loss', l_loss)
        self.log('test unsupervised loss', u_loss)

        return {'test_loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.params.train.learning_rate)

    def prepare_data(self):
        MNIST(self.params.dataset.mnist.path, download=True)

    def train_dataloader(self):
        train_data = MNIST(self.params.dataset.mnist.path, train=True, download=False,
                                    transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        lambda x: torch.distributions.binomial.Binomial(total_count=1,probs=x).sample(),
                                        lambda x: x.view(-1),
                                    ]))
        train_dataset = make_mnist_semi_super(train_data, self.params.dataset.mnist.train_pct, train=True)
        return DataLoader(train_dataset, batch_size=self.params.train.batch_size, shuffle=True)

    def val_dataloader(self):
        val_data = MNIST(self.params.dataset.mnist.path, train=True, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           lambda x: torch.distributions.binomial.Binomial(total_count=1,probs=x).sample(),
                           lambda x: x.view(-1),
                       ]))
        val_dataset = make_mnist_semi_super(val_data, self.params.dataset.mnist.val_pct, val=True)
        return DataLoader(val_dataset, batch_size=self.params.train.batch_size, shuffle=True)

    def test_dataloader(self):
        test_data = MNIST(self.params.dataset.mnist.path, train=False, download=False,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           lambda x: torch.distributions.binomial.Binomial(total_count=1,probs=x).sample(),
                           lambda x: x.view(-1),
                       ]))
        test_dataset = make_mnist_semi_super(test_data)
        return DataLoader(test_dataset, batch_size=self.params.train.batch_size, shuffle=True)
