''' Loading and training ML model '''

from typing import Callable, Dict


def get_pretrained_model(selected_model: str = 'liquid') -> Callable[[str], Dict]:
    ''' Loads a pre-trained model '''
    if selected_model == 'liquid':
        from Modules.Models.liquid import liquid_desc_to_json
        pre_trained_model = liquid_desc_to_json

    return pre_trained_model


def train_model() -> None:
    ''' Trains the model '''
    pass
