import torch
import torch.nn as nn


class BaseLoss(nn.Module):
    def __init__(self, output_key, target_key, nickname=None, weight=1):
        super().__init__()
        self.output_key = output_key
        self.target_key = target_key
        self.weight = weight
        self.nickname = self.__class__.__name__ if nickname is None else nickname

    def _preproces(self, data_dict, output_dict):
        return data_dict, output_dict

    def _postprocess(self, output, target):
        return output, target

    def forward(self, data_dict, output_dict):
        data_dict, output_dict = self._preproces(data_dict, output_dict)
        output = output_dict[self.output_key]
        target = data_dict[self.target_key]
        output, target = self._postprocess(output, target)
        return self.loss_fn(output, target)


class CrossEntropyLoss(BaseLoss):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.loss_fn = nn.CrossEntropyLoss()


class GANLoss(BaseLoss):
    def __init__(
        self, network,
        type='lsgan',
        target_real_label=1.0, target_fake_label=0.0,
        *args, **kargs
    ):
        super().__init__(output_key=None, target_key=None, *args, **kargs)
        if type == 'nsgan':
            self.loss_fn = nn.BCELoss()

        elif type == 'lsgan':
            self.loss_fn = nn.MSELoss()

        elif type == 'hinge':
            self.loss_fn = nn.ReLU()

        elif type == 'l1':
            self.loss_fn = nn.L1Loss()

        else:
            raise NotImplementedError()

        self.network = network
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

    def forward(self, data_dict, output_dict):
        if self.network == 'generator':
            outputs = output_dict['D_G_z']
            targets = self.real_label.expand_as(outputs).to(outputs.device)
            loss = self.loss_fn(outputs, targets)

        elif self.network == 'discriminator':
            fake_outputs = output_dict['D_G_z']
            fake_targets = self.fake_label.expand_as(fake_outputs).to(fake_outputs.device)
            loss_d_fake = self.loss_fn(fake_outputs, fake_targets)

            real_outputs = output_dict['D_x']
            real_targets = self.real_label.expand_as(real_outputs).to(real_outputs.device)
            loss_d_real = self.loss_fn(real_outputs, real_targets)
            loss = (loss_d_fake + loss_d_real) / 2

        else:
            raise NotImplementedError(f"Wrong network '{self.network}' for GANMSELoss")

        return loss


# Formulation reference: https://arxiv.org/pdf/1802.05957.pdf (eq. 17)
class HingeLossG(nn.Module):
    def __init__(self, nickname=None, weight=1):
        super().__init__()
        self.weight = weight
        self.nickname = self.__class__.__name__ if nickname is None else nickname

    def forward(self, data_dict, output_dict):
        D_G_z = output_dict['D_G_z']
        return (-D_G_z).mean()


# Formulation reference: https://arxiv.org/pdf/1802.05957.pdf (eq. 16)
class HingeLossD(nn.Module):
    def __init__(self, nickname=None, weight=1):
        super().__init__()
        self.weight = weight
        self.nickname = self.__class__.__name__ if nickname is None else nickname
        self.relu = nn.ReLU()

    def forward(self, data_dict, output_dict):
        D_G_z = output_dict['D_G_z']
        D_x = output_dict['D_x']
        loss_real = self.relu(1 - D_x).mean()
        loss_fake = self.relu(1 + D_G_z).mean()
        return loss_real + loss_fake


# Based on https://github.com/knazeri/edge-connect/blob/master/src/loss.py
class AdversarialLoss(nn.Module):
    r"""
    Adversarial loss
    https://arxiv.org/abs/1711.10337
    """

    def __init__(self, type='lsgan', target_real_label=1.0, target_fake_label=0.0):
        r"""
        type = nsgan | lsgan | hinge | l1
        """
        super(AdversarialLoss, self).__init__()

        self.type = type
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

        if type == 'nsgan':
            self.criterion = nn.BCELoss()

        elif type == 'lsgan':
            self.criterion = nn.MSELoss()

        elif type == 'hinge':
            self.criterion = nn.ReLU()

        elif type == 'l1':
            self.criterion = nn.L1Loss()

    def __call__(self, outputs, is_real, is_disc=None):
        if self.type == 'hinge':
            if is_disc:
                if is_real:
                    outputs = -outputs
                return self.criterion(1 + outputs).mean()
            else:
                return (-outputs).mean()

        else:
            labels = (self.real_label if is_real else self.fake_label).expand_as(outputs)
            loss = self.criterion(outputs, labels)

            return loss
