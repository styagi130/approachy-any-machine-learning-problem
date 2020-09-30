import yaml
import pathlib

def load_config(config_filepath: pathlib.Path):
    """
        Function to load config file
        :param config_filepath: Path to configuration file
        :return : SimpleNamespace config object
    """
    with config_filepath.open() as config_file:
        config = yaml.full_load(config_file)

    all_config_params = vars(config)
    for config_param, config_param_value in all_config_params.items():
        if "path" in config_param:
            setattr(config, config_param, pathlib.Path(config_param_value))
    return config