import os
from typing import Optional

import fire
import cerebrium

from src.paths import MODELS_DIR
from src.logger import get_console_logger

logger = get_console_logger(name='model_deployment')

try:
    CEREBRIUM_API_KEY = os.environ['CEREBRIUM_API_KEY']
except KeyError:
    logger.error('CEREBRIUM_API_KEY environment variable not set')
    raise


def deploy_model(
    local_pickle: Optional[str] = None,
    from_model_registry: bool = False,
) -> None:
    """Deploy a trained sklearn pipeline to Cerebrium."""
    logger.info('Deploying model...')

    if from_model_registry:
        raise NotImplementedError('Deployment from model registry is not yet implemented')
    elif local_pickle:
        model_pickle_file = MODELS_DIR / local_pickle
        if not model_pickle_file.exists():
            raise FileNotFoundError(f'Model file not found: {model_pickle_file}')
        logger.info(f'Deploying from local pickle: {model_pickle_file}')
        endpoint = cerebrium.deploy(
            (cerebrium.model_type.SKLEARN, model_pickle_file),
            "stock-predict-model",
            CEREBRIUM_API_KEY,
        )
        logger.info(f'Model deployed. Endpoint: {endpoint}')
    else:
        raise ValueError('Must specify either --local-pickle or --from-model-registry')


if __name__ == '__main__':
    fire.Fire(deploy_model)
