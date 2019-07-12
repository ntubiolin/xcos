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
    def _eval_metrics(self, data_input, model_output):
        acc_metrics = np.zeros(len(self.metrics))
        for i, metric in enumerate(self.metrics):
            acc_metrics[i] += metric(data_input, model_output)
            self.writer.add_scalar(f'{metric.__name__}', acc_metrics[i])
        return acc_metrics

    def _get_loss(self, data_input, model_output):
        losses = []
        for loss_name, (loss_instance, loss_weight) in self.losses.items():
            if loss_weight <= 0.0:
                continue
            loss = loss_instance(data_input, model_output) * loss_weight
            losses.append(loss)
            self.writer.add_scalar(f'{loss_name}', loss.item())
        loss = sum(losses)
        return loss

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

    def _write_images(self, data_input, model_output):

        self.writer.add_image("data_input", make_grid(data_input["data_input"], nrow=4, normalize=True))

    def inference(self, data_loader, saved_keys=['verb_logits', 'noun_logits', 'uid']):
        self.model.eval()
        self.logger.info(f'Inferencing with following keys to save: {saved_keys}')
        self.logger.info(f'Number of examples is around {data_loader.batch_size * len(data_loader)}')
        saved_results = {k: [] for k in saved_keys}

        with torch.no_grad():
            for batch_idx, data_input in enumerate(data_loader):
                self.writer.set_step(batch_idx, 'inference')
                for key in data_input.keys():
                    value = data_input[key]
                    if key in saved_keys:
                        saved_value = value.numpy() if torch.is_tensor(value) else value
                        saved_results[key].extend([v for v in saved_value])
                    data_input[key] = value.to(self.device) if torch.is_tensor(value) else value

                model_output = self.model(data_input)
                for key in model_output.keys():
                    if key in saved_keys:
                        saved_results[key].extend([v for v in model_output[key].cpu().numpy()])

                if batch_idx == 0:
                    non_exists_keys = [key for key in saved_keys if len(saved_results[key]) == 0]
                    if len(non_exists_keys) > 0:
                        self.logger.warning(f'Keys {non_exists_keys} not exists')
                if batch_idx % 10 == 0:
                    self.logger.info(f'Entry {batch_idx * data_loader.batch_size} done.')

        return saved_results
