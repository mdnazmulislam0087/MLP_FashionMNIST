import yaml

def read_config(config_path):
    """Reads the configuration

    Args:
        config_path (string): File path of config.yaml

    Returns:
        dictionary: Returns the dictionary containing the config from the config yaml file.
    """
    
    with open (config_path) as config_file:
        content = yaml.safe_load(config_file)
    return content
