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


class SiameseCrossEntropyLoss(BaseLoss):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.loss_fn = nn.CrossEntropyLoss()

    def _preproces(self, data_dict, output_dict):
        data_dict[self.target_key] = torch.cat(data_dict[self.target_key])
        return data_dict, output_dict


class SiameseMSELoss(BaseLoss):
    def __init__(self, *args, **kargs):
        super().__init__(*args, **kargs)
        self.loss_fn = nn.MSELoss()

    def _preproces(self, data_dict, output_dict):
        data_dict[self.target_key] = output_dict[self.target_key]
        return data_dict, output_dict


class GANLoss(BaseLoss):
    def __init__(
        self, network,
        type_='lsgan',
        target_real_label=1.0, target_fake_label=0.0,
        *args, **kargs
    ):
        super().__init__(output_key=None, target_key=None, *args, **kargs)
        assert network in ['generator', 'discriminator']
        if type_ == 'nsgan':
            self.loss_fn = nn.BCELoss()
        elif type_ == 'lsgan':
            self.loss_fn = nn.MSELoss()
        elif type_ == 'l1':
            self.loss_fn = nn.L1Loss()
        elif type_ == 'hinge':
            self.hinge_loss = HingeLossG() if network == 'generator' else HingeLossD()
        else:
            raise NotImplementedError()
        self.type_ = type_

        self.network = network
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))

    def forward(self, data_dict, output_dict):
        if self.type_ == 'hinge':
            return self.hinge_loss(data_dict, output_dict)

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
    def __init__(self):
        super().__init__()

    def forward(self, data_dict, output_dict):
        D_G_z = output_dict['D_G_z']
        return (-D_G_z).mean()


# Formulation reference: https://arxiv.org/pdf/1802.05957.pdf (eq. 16)
class HingeLossD(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()

    def forward(self, data_dict, output_dict):
        D_G_z = output_dict['D_G_z']
        D_x = output_dict['D_x']
        loss_real = self.relu(1 - D_x).mean()
        loss_fake = self.relu(1 + D_G_z).mean()
        return loss_real + loss_fake
