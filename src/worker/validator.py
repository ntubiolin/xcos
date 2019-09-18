from .training_worker import TrainingWorker
from pipeline.base_pipeline import BasePipeline


class Validator(TrainingWorker):
    """
    Validator class

    Note:
        Inherited from WorkerTemplate.
    """
    def __init__(self, pipeline: BasePipeline, *args):
        super().__init__(pipeline, *args)
        # Some shared attributes are validator exclusive and therefore is initialized here
        for attr_name in ['loss_functions', 'optimize_strategy']:
            setattr(self, attr_name, getattr(pipeline, attr_name))
        if self.optimize_strategy == 'GAN':
            attr_name = 'gan_loss_functions'
            setattr(self, attr_name, getattr(pipeline, attr_name))

    @property
    def enable_grad(self):
        return False

    def _run_and_optimize_model(self, data):
        if self.optimize_strategy == 'normal':
            model_output = self.model(data)
            losses, total_loss = self._get_and_write_losses(data, model_output)
        elif self.optimize_strategy == 'GAN':
            model_output = self.model(data, scenario='generator_only')
            losses, total_loss = self._get_and_write_losses(data, model_output)

        return model_output, total_loss

    def _setup_model(self):
        self.model.eval()
