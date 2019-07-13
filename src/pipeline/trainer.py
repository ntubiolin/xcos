import time

from .worker_template import WorkerTemplate
from utils.logging_config import logger


class Trainer(WorkerTemplate):
    """
    Trainer class

    Note:
        Inherited from WorkerTemplate.
    """
    def _print_log(self, epoch, batch_idx, batch_start_time, loss, metrics):
        logger.info(
            f'Epoch: {epoch} [{batch_idx * self.data_loader.batch_size}/{self.data_loader.n_samples} '
            f' ({100.0 * batch_idx / len(self.data_loader):.0f}%)] '
            f'loss_total: {loss.item():.6f}, '
            f'BT: {time.time() - batch_start_time:.2f}s'
        )
