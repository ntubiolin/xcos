import json

from attrdict import AttrDict

from .logging_config import logger


class SingleGlobalConfig(AttrDict):
    """ The global config object for all module configuration.

    It needs to be setup first by main.py

    One could use either global_config.some_attribute or global_config['some_attribute']
    to access the config
    """

    def set_config(self, config):
        for k, v in config:
            self[k] = v

    def extend_configs(self, config_filenames: list):
        # load config files, the overlapped entries will be overwriten
        for config_filename in config_filenames:
            with open(config_filename) as fin:
                config = json.load(fin)
                self._extend_config(config)

    def _extend_config(self, added_config):
        for key, value in added_config.items():
            if key in self.keys():
                if key == 'name':
                    value = f"{self[key]}_{value}"
                else:
                    logger.warning(f"Overriding '{key}' in config")
                del self[key]
            self[key] = value


# Initialize this global_config first
# and then for all modules, import this config
global_config = SingleGlobalConfig()
