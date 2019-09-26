'''
global_config.py

Global configuration module with a global var "global_config" for other modules to access
all configuration.
'''
from copy import deepcopy

import json
from attrdict import AttrDict

from .logging_config import logger


def flatten_nested_dict(nested_dict: dict, root_path: str, flattened_dict: dict):
    """ Recursively iterate all values in a nested dictionary and return a flatten one.

    Args:
        nested_dict (dict): the nested dictionary to be flatten
        root_path (str): node path, viewing nested_dict as a tree
        flattened_dict (dict): recorded flattened_dict for recursive call

    Returns:
        flattened_dict (dict): flatten dictionary with flatten_key (root_node/leavenode/...) as path

    """
    for k, v in nested_dict.items():
        current_path = f"{root_path}/{k}" if root_path != "" else k

        if type(v) != dict:
            flattened_dict[current_path] = v
        else:
            flattened_dict = flatten_nested_dict(v, current_path, flattened_dict)
    return flattened_dict


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
    flattened_changed_config = {}
    added_config = {}
    # flattened nested dictionaries
    flattened_template_config = flatten_nested_dict(template_config, "", dict())
    flattened_specified_config = flatten_nested_dict(specified_config, "", dict())

    # Check each value in specified_config to see if it is different from the template
    for k, v in flattened_specified_config.items():
        # Concatenate if it is name
        if k == 'name' and 'name' in specified_config:
            flattened_changed_config['name'] = f"{template_config['name']}+{specified_config['name']}"

        # Added to flattened_changed_config only if it is different from the tempalte
        elif k in flattened_template_config:
            if v != flattened_template_config[k]:
                flattened_changed_config[k] = v

        # Added to both added_config flattened_changed_config if it is new
        else:
            flattened_changed_config[k] = v
            added_config[k] = v
    return flattened_changed_config, added_config


def merge_template_and_flattened_changed_config(template_config, flattened_changed_config):
    """ Merge the template and changed config as a global_config. """
    merged_config = deepcopy(template_config)
    for k, v in flattened_changed_config.items():
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

    Note that since this class is inherited from AttrDict, attributes without starting with a _ will
    be put into the dictionary. Attrbutes starting with a _ would be viewed as invalid.
    To setup template_config and specified_config, here the '_allow_invalid_attributes' is set to
    be True first to allow this invalid self attributes and avoid putting them into the self dict.
    """
    def setup(self, template_config_filename: list, specified_config_filenames: list, resumed_checkpoint: dict = None):
        """ Setup the global_config. """
        # NOTE: this function needs to be called by main.py before imported by modules unless it is resumed

        # This is to set self._template_config, self._specified_config etc as they are classified as invalid attributes
        # to be put into self. See https://github.com/bcj/AttrDict/blob/9f672997bf/attrdict/mixins.py#L169
        self._setattr('_allow_invalid_attributes', True)

        if resumed_checkpoint is not None:
            self._template_config = resumed_checkpoint['config']
        else:
            self._template_config = self._load_template_config(template_config_filename)
        self._specified_config = self._load_specified_configs(specified_config_filenames)

        # Compare specified_config and template_config to get changed_config/merged_config
        self._flattened_changed_config, self._added_config = \
            get_changed_and_added_config(self._template_config, self._specified_config)
        self._merged_config = merge_template_and_flattened_changed_config(
            self._template_config, self._flattened_changed_config)

        self._setattr('_allow_invalid_attributes', False)
        self.set_config(self._merged_config)

    def set_config(self, config: dict):
        """ Set the config. """
        for k, v in config.items():
            self[k] = v

    def __print__(self):
        """ Print all key & value pairs. """
        for k, v in self.items():
            logger.info(f"{k}: {v}")

    def print_changed(self):
        """ Print all changed/added values. """
        for k, v in self._flattened_changed_config.items():
            if k in self._added_config:
                logger.info(f"Added key: {k} ({v})")
            else:
                original_value = get_value_in_nested_dict(self._template_config, k.split('/'))
                logger.warning(f"Changed key: {k} ({original_value} -> {v})")

    def _load_template_config(self, config_filename: str):
        """ Load the template config. """
        # Note that since this class is inherited from AttrDict, attributes without starting with a _ will
        # be put into the dictionary.
        with open(config_filename) as fin:
            logger.info(f"===== Using {config_filename} as template config =====")
            return json.load(fin)

    def _load_specified_configs(self, config_filenames: list):
        """ Load specified config(s). """
        # Note that since this class is inherited from AttrDict, attributes without starting with a _ will
        # be put into the dictionary.
        return self._extend_configs({}, config_filenames)

    def _extend_configs(self, config: dict, config_filenames: list):
        """ Extend a dict config with several config files. """
        if config_filenames is None:
            return {}
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
