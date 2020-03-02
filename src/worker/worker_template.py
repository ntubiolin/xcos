import time
from abc import ABC, abstractmethod

import torch
from torchvision.utils import make_grid

from data_loader.base_data_loader import BaseDataLoader
from pipeline.base_pipeline import BasePipeline
from utils.global_config import global_config
from utils.util import batch_visualize_xcos


class WorkerTemplate(ABC):
    """ Worker template, base class for trainer, validator and tester.

    Child class need to implement at least the _run_and_optimize_model() method
    that deals with the main optimization & model inference.
    """
    def __init__(
        self, pipeline: BasePipeline, data_loader: BaseDataLoader, step: int
    ):
        # Attributes listed below are shared from pipeline among all different workers.
        for attr_name in ['device', 'model', 'evaluation_metrics', 'writer', 'optimize_strategy']:
            setattr(self, attr_name, getattr(pipeline, attr_name))

        self.data_loader = data_loader
        self.step = step  # Tensorboard log step

    # ============ Implement the following functions ==============
    @property
    @abstractmethod
    def enable_grad(self):
        pass

    @abstractmethod
    def _run_and_optimize_model(self, data):
        """ Put data into model and optimize the model"""
        return {}, None

    def _print_log(self, epoch, batch_idx, batch_start_time, loss):
        """ Print messages on terminal. """
        pass

    @abstractmethod
    def _setup_model(self):
        """ Set random seed and self.model.eval() or self.model.train() """
        pass

    @abstractmethod
    def _init_output(self):
        pass

    @abstractmethod
    def _update_output(self, output: dict, products: dict):
        return output

    @abstractmethod
    def _finalize_output(self, epoch_output) -> dict:
        """ The final output of worker.run() will be processed by this
            function, whose responsibility is to create a dictionary contraining
            log messages and/or saved inference outputs. """
        pass

    # ============ Implement the above functions ==============
    def _update_all_metrics(self, data_input, model_output, write=True):
        for metric in self.evaluation_metrics:
            with torch.no_grad():
                value = metric.update(data_input, model_output)
                # some metrics do not have per-batch evaluation (e.g. FID), then value would be None
                if write and value is not None:
                    self.writer.add_scalar(metric.nickname, value)

    # Generally, the following function should not be changed.
    def _write_data_to_tensorboard(self, data, model_output):
        """ Write images to Tensorboard """
        img_tensors = data["data_input"]
        if not isinstance(img_tensors, torch.Tensor):
            img_tensors = torch.cat(img_tensors)
        if global_config.arch.type == "xCosModel":
            img1s, img2s = data['data_input']
            img1s = img1s.cpu().numpy()
            img2s = img2s.cpu().numpy()
            grid_cos_maps = model_output['grid_cos_maps'].squeeze().detach().cpu().numpy()
            attention_maps = model_output['attention_maps'].squeeze().detach().cpu().numpy()
            visualizations = batch_visualize_xcos(img1s, img2s, grid_cos_maps, attention_maps)
            if len(visualizations) > 10:
                visualizations = visualizations[:10]
            self.writer.add_image("xcos_visualization", make_grid(torch.cat(visualizations), nrow=1))

        if self.optimize_strategy == 'GAN':
            self.writer.add_image("G_z", make_grid(model_output["G_z"], nrow=4, normalize=True))
            self.writer.add_histogram("dist_G_z", model_output["G_z"])
            self.writer.add_histogram("dist_x", img_tensors)

    def _setup_writer(self):
        """ Setup Tensorboard writer for each iteration """
        self.writer.set_step(self.step, self.data_loader.name)
        self.step += 1

    def _get_and_write_losses(self, data, model_output):
        """ Calculate losses and write them to Tensorboard
        Losses (dict: nickname -> loss tensor) and total loss (tensor) will be returned.
        """
        losses = {}
        for loss_function in self.loss_functions:
            if loss_function.weight <= 0.0:
                continue
            loss = loss_function(data, model_output) * loss_function.weight
            losses[loss_function.nickname] = loss
            self.writer.add_scalar(f'{loss_function.nickname}', loss.item())
        if len(self.loss_functions) == 0:
            total_loss = torch.zeros([1])
        else:
            total_loss = torch.stack(list(losses.values()), dim=0).sum(dim=0)
        self.writer.add_scalar('total_loss', total_loss.item())
        return losses, total_loss

    def _data_to_device(self, data):
        """ Put data into CPU/GPU """
        for key in data.keys():
            # Dataloader yeilds something that's not tensor, e.g data['video_id']
            if torch.is_tensor(data[key]):
                data[key] = data[key].to(self.device)
            elif isinstance(data[key], list):
                for i, elem in enumerate(data[key]):
                    data[key][i] = elem.to(self.device)
        return data

    def _iter_data(self, epoch):
        """
        Iterate through the dataset and do inference.
        Output of this worker will be init and updated(after a batch) here using
        `self._output_init` and `self._output_update`.
        """
        output = self._init_output()
        for batch_idx, data in enumerate(self.data_loader):
            batch_start_time = time.time()
            self._setup_writer()
            data = self._data_to_device(data)
            data['batch_idx'] = batch_idx
            model_output, loss = self._run_and_optimize_model(data)

            products = {
                'data': data,
                'model_output': model_output,
                'loss': loss,
            }

            if batch_idx % global_config.log_step == 0:
                self._write_data_to_tensorboard(data, model_output)
                if global_config.verbosity >= 2:
                    self._print_log(epoch, batch_idx, batch_start_time, loss)

            output = self._update_output(output, products)
        return output

    def run(self, epoch):
        self._setup_model()
        with torch.set_grad_enabled(self.enable_grad):
            epoch_output = self._iter_data(epoch)
        output = self._finalize_output(epoch_output)
        return output
