# imports
from pydantic import BaseModel

# project imports
from src.paths import MODELS_DIR
from src.logger import get_console_logger

# log run
logger = get_console_logger(name='model_deployment')