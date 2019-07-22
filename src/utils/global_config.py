from copy import deepcopy

import json
from attrdict import AttrDict

from .logging_config import logger


def iter_attr_dict(attr_dict: AttrDict, root_path: str, key_value_pair_dict: AttrDict):
    for k, v in attr_dict.items():
        current_path = f"{root_path}/{k}" if root_path != "" else k

        if type(v) != dict:
            key_value_pair_dict[current_path] = v
        else:
            key_value_pair_dict = iter_attr_dict(v, current_path, key_value_pair_dict)
    return key_value_pair_dict


def get_value_in_nested_dict(nested_dict: dict, keys: list):
    temp = nested_dict
    for i, k in enumerate(keys):
        temp = temp[k]
        if i == len(keys) - 1:
            return temp


def get_changed_and_added_config(template_config, specified_config):
    changed_config = {}
    added_config = {}
    flatten_template_config = iter_attr_dict(template_config, "", {})
    flatten_specified_config = iter_attr_dict(specified_config, "", {})
    for k, v in flatten_specified_config.items():
        if k == 'name':
            changed_config['name'] = f"{template_config['name']}+{specified_config['name']}"
        elif k in flatten_template_config:
            if v != flatten_template_config[k]:
                changed_config[k] = v
        else:
            changed_config[k] = v
            added_config[k] = v
    return changed_config, added_config


def merge_template_and_changed_config(template_config, changed_config):
    merged_config = deepcopy(template_config)
    for k, v in changed_config.items():
        keys = k.split('/')
        temp = merged_config
        for i, k in enumerate(keys):
            if i == len(keys) - 1:
                temp[k] = v
            else:
                if k not in temp:
                    temp[k] = {}
                temp = temp[k]
    return merged_config


class SingleGlobalConfig(AttrDict):
    """ The global config object for all module configuration.

    It needs to be setup first by main.py

    One could use either global_config.some_attribute or global_config['some_attribute']
    to access the config
    """
    def setup(self, template_config_filename, specified_config_filenames):
        self._load_template_conifg(template_config_filename)
        self._load_specified_configs(specified_config_filenames)
        self._get_changed_and_merged_config()
        self.set_config(self.merged_config)

    def set_config(self, config):
        for k, v in config.items():
            self[k] = v

    def __print__(self):
        for k, v in self.items():
            logger.info(f"{k}: {v}")

    def print_changed(self):
        for k, v in self.changed_config.items():
            if k in self.added_config:
                logger.info(f"Added key: {k} ({v})")
            else:
                original_value = get_value_in_nested_dict(self.template_config, k.split('/'))
                breakpoint()
                logger.warning(f"Changed key: {k} ({original_value} -> {v})")

    def _load_template_conifg(self, config_filename):
        # Note that for some reason this is not mutable
        with open(config_filename) as fin:
            self.template_config = json.load(fin)

    def _load_specified_configs(self, config_filenames):
        # Note that for some reason this is not mutable
        self.specified_config = self._extend_configs({}, config_filenames)

    def _get_changed_and_merged_config(self):
        self.changed_config, self.added_config = \
            get_changed_and_added_config(self.template_config, self.specified_config)
        self.merged_config = merge_template_and_changed_config(self.template_config, self.changed_config)

    def _extend_configs(self, config, config_filenames: list):
        # load config files, the overlapped entries will be overwriten
        for config_filename in config_filenames:
            with open(config_filename) as fin:
                added_config = json.load(fin)
                config = self._extend_config(config, added_config)
        return config

    def _extend_config(self, config, added_config):
        for key, value in added_config.items():
            if key in config.keys():
                if key == 'name':
                    value = f"{config[key]}_{value}"
                else:
                    logger.warning(f"Overriding '{key}' in config")
                del config[key]
            config[key] = value
        return config


# Initialize this global_config first
# and then for all modules, import this config
global_config = SingleGlobalConfig()
