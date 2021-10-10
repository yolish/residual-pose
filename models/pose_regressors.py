from models.posenet.ResidualPoseNet import ResidualPoseNet

def get_model(model_name, config):
    """
    Get the instance of the request model
    :param model_name: (str) model name
    :param config: (dict) config file
    :return: instance of the model (nn.Module)
    """
    if model_name == 'posenet':
        return ResidualPoseNet(config)
    else:
        raise "{} not supported".format(model_name)