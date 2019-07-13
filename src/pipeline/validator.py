import numpy as np
import torch
from torchvision.utils import make_grid

from .worker_template import WorkerTemplate


class Validator(WorkerTemplate):
    """
    Validator class

    Note:
        Inherited from WorkerTemplate.
    """
    def _run_and_optimize_model(self, data):
        model_output = self.model(data)
        loss = self._get_and_write_loss(data, model_output)

        metrics = self._get_and_write_metrics(data, model_output)
        return model_output, loss, metrics

    def _to_log(self, epoch, epoch_time, avg_loss, avg_metrics):
        log = {
            'valid_epoch_time': epoch_time,
            'valid_avg_loss': avg_loss,
            # 'valid_avg_metrics': avg_metrics
        }
        for i, item in enumerate(self.config['metrics']):
            key = item["args"]["nickname"]
            log[f"valid_avg_{key}"] = avg_metrics[i]

        return log

    def _valid_epoch(self, epoch, valid_loader_idx):
        """
        Validate after training an epoch

        :return: A log that contains information about validation

        Note:
            The validation metrics in log must have the key 'val_metrics'.
        """
        self.model.eval()
        total_val_loss = 0
        total_val_metrics = np.zeros(len(self.metrics))
        loader = self.valid_data_loaders[valid_loader_idx]
        with torch.no_grad():
            for batch_idx, data_input in enumerate(loader):
                self.writer.set_step(self.valid_iteration_counts[valid_loader_idx], loader.name)
                self.valid_iteration_counts[valid_loader_idx] += 1
                for key in data_input.keys():
                    # Dataloader yeilds something that's not tensor, e.g data_input['video_id']
                    if torch.is_tensor(data_input[key]):
                        data_input[key] = data_input[key].to(self.device)
                model_output = self.model(data_input)

                loss = self._get_loss(data_input, model_output)

                self.writer.add_scalar('total_loss', loss.item())
                total_val_loss += loss.item()
                total_val_metrics += self._eval_metrics(data_input, model_output)
                if self.verbosity >= 2 and batch_idx % self.log_step == 0:
                    self._write_images(data_input, model_output)

        return {
            f'{loader.name}_loss': total_val_loss / len(loader),
            f'{loader.name}_metrics': (total_val_metrics / len(loader)).tolist()
        }
        self.writer.add_image("data_input", make_grid(data_input["data_input"], nrow=4, normalize=True))
