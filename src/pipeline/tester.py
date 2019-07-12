import torch

from .worker_template import WorkerTemplate


class Tester(WorkerTemplate):
    """
    Tester class

    Note:
        Inherited from WorkerTemplate.
    """
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
