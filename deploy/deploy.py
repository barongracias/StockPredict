# imports
import os
from typing import Optional

import fire
from cerebrium import deploy, model_type

# project imports
from src.paths import MODELS_DIR
from src.logger import get_console_logger

# log run
logger = get_console_logger(name='model_deployment')

try:
    CEREBRIUM_API_KEY = os.environ['CEREBRIUM_API_KEY']
except KeyError:
    logger.error('Cerebrium API key environment variable not set')
    raise

def deploy(
    local_pickle: Optional[str] = None,
    from_model_registry: bool = False
):
    logger.info('Deploying model...')
    
    if from_model_registry:
        logger.info('Loading model from model registry...')
        raise NotImplementedError('To Do')
    elif local_pickle:
        logger.info('Deploying model from local pickle...')
        model_pickle_file = MODELS_DIR / local_pickle
        endpoint = deploy((model_type.SKLEARN, model_pickle_file), "sk-test-model", CEREBRIUM_API_KEY)
    else:
        raise ValueError('Must specify either --local-pickle or --from-model-registry')
    
    logger.info('Model deployed')

if __name__ == '__main__':
    fire.Fire(deploy)
    