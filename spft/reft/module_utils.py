import torch


def get_reft_dict(state_dict):
    """
    Extracts the REFT dictionary from the state dictionary.
    
    Args:
        state_dict (dict): The state dictionary containing model parameters.
        
    Returns:
        dict: A dictionary containing the REFT parameters.
    """
    reft_dict = {}
    for key, value in state_dict.items():
        if "reft" in key:
            reft_dict[key] = value
    return reft_dict
