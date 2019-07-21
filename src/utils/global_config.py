import json

from logging_config import logger


class SingleGlobalConfig:
    def __init__(self):
        self.config = {}

    def set_config(self, config):
        self.config = config

    def extend_configs(self, config_filenames: list):
        # load config files, the overlapped entries will be overwriten
        for config_filename in config_filenames:
            with open(config_filename) as fin:
                config = json.load(fin)
                self.extend_config(config)

    def extend_config(self, added_config):
        for key, value in added_config.items():
            if key in self.config.keys():
                if key == 'name':
                    value = f"{self.config[key]}_{value}"
                else:
                    logger.warning(f"Overriding '{key}' in config")
                del self.config[key]
            self.config[key] = value
        return self.config


global_config = SingleGlobalConfig()
