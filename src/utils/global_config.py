'''
global_config.py

Global configuration module with a global var "global_config" for other modules to access
all configuration.
'''
from copy import deepcopy

import json
from attrdict import AttrDict

from .logging_config import logger


def flatten_nested_dict(nested_dict: dict, root_path: str, flatten_dict: dict = {}):
    """ Recursively iterate all values in a nested dictionary and return a flatten one.

    Args:
        nested_dict (dict): the nested dictionary to be flatten
        root_path (str): node path, viewing nested_dict as a tree
        flatten_dict (dict): recorded flatten_dict for recursive call

    Returns:
        flatten_dict (dict): flatten dictionary with flatten_key (root_node/leavenode/...) as path

    """
    for k, v in nested_dict.items():
        current_path = f"{root_path}/{k}" if root_path != "" else k

        if type(v) != dict:
            flatten_dict[current_path] = v
        else:
            flatten_dict = flatten_nested_dict(v, current_path, flatten_dict)
    return flatten_dict


def get_value_in_nested_dict(nested_dict: dict, keys: list):
    """ Get a value in a nested dictionary given a sequential key list. """
    temp = nested_dict
    for i, k in enumerate(keys):
        temp = temp[k]
        if i == len(keys) - 1:
            return temp


def get_changed_and_added_config(template_config: dict, specified_config: dict):
    """ Compare the difference between template config and specified config,
    and return changed (include added) and added config.
    """
    changed_config = {}
    added_config = {}
    # Flatten nested dictionaries
    flatten_template_config = flatten_nested_dict(template_config, "")
    flatten_specified_config = flatten_nested_dict(specified_config, "")

    # Check each value in specified_config to see if it is different from the template
    for k, v in flatten_specified_config.items():
        # Concatenate if it is name
        if k == 'name':
            changed_config['name'] = f"{template_config['name']}+{specified_config['name']}"

        # Added to changed_config only if it is different from the tempalte
        elif k in flatten_template_config:
            if v != flatten_template_config[k]:
                changed_config[k] = v

        # Added to both added_config changed_config if it is new
        else:
            changed_config[k] = v
            added_config[k] = v
    return changed_config, added_config


def merge_template_and_changed_config(template_config, changed_config):
    """ Merge the template and changed config as a global_config. """
    merged_config = deepcopy(template_config)
    for k, v in changed_config.items():
        keys = k.split('/')

        # Trace the path by the key and current_dict
        current_dict = merged_config
        for i, k in enumerate(keys):
            if i == len(keys) - 1:
                current_dict[k] = v
            else:
                # If it is added, create a new dictionary for it
                if k not in current_dict:
                    current_dict[k] = {}
                current_dict = current_dict[k]
    return merged_config


class SingleGlobalConfig(AttrDict):
    """ The global config object for all module configuration.

    It needs to be setup first by main.py

    It is a AttrDict with additional functions for template/specified config settings.

    One could use either global_config.some_attribute or global_config['some_attribute']
    to access the config
    """
    def setup(self, template_config_filename: list, specified_config_filenames: list):
        """ Setup the global_config. """
        # NOTE: this function needs to be called by main.py before imported by modules unless it is resumed
        self._load_template_conifg(template_config_filename)
        self._load_specified_configs(specified_config_filenames)
        self._get_changed_and_merged_config()
        self.set_config(self.merged_config)

    def set_config(self, config: list):
        """ Set the config. """
        for k, v in config.items():
            self[k] = v

    def __print__(self):
        """ Print all key & value pairs. """
        for k, v in self.items():
            logger.info(f"{k}: {v}")

    def print_changed(self):
        """ Print all changed/added values. """
        for k, v in self.changed_config.items():
            if k in self.added_config:
                logger.info(f"Added key: {k} ({v})")
            else:
                original_value = get_value_in_nested_dict(self.template_config, k.split('/'))
                logger.warning(f"Changed key: {k} ({original_value} -> {v})")

    def _load_template_conifg(self, config_filename: str):
        """ Load the template config. """
        # Note that for some reason this is not mutable?
        with open(config_filename) as fin:
            self.template_config = json.load(fin)

    def _load_specified_configs(self, config_filenames: list):
        """ Load specified config(s). """
        # Note that for some reason this is not mutable?
        self.specified_config = self._extend_configs({}, config_filenames)

    def _get_changed_and_merged_config(self):
        """ Compare specified_config and template_config to get changed_config/merged_config. """
        self.changed_config, self.added_config = \
            get_changed_and_added_config(self.template_config, self.specified_config)
        self.merged_config = merge_template_and_changed_config(self.template_config, self.changed_config)

    def _extend_configs(self, config: dict, config_filenames: list):
        """ Extend a dict config with several config files. """
        # load config files, the overlapped entries will be overwriten
        for config_filename in config_filenames:
            with open(config_filename) as fin:
                added_config = json.load(fin)
                config = self._extend_config(config, added_config)
        return config

    def _extend_config(self, config: dict, added_config: dict):
        """ Extend a dict config with an dict added_config"""
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
