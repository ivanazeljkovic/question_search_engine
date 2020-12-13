import os
import logging
from constants import LOGS_DIR_PATH


if not os.path.exists(LOGS_DIR_PATH):
    os.mkdir(LOGS_DIR_PATH)

logging.basicConfig(filename=os.path.join(LOGS_DIR_PATH, 'question_search_engine.log'),
                    format='%(asctime)s.%(msecs)03d %(levelname)-8s %(message)s',
                    datefmt='%d %b %Y %H:%M:%S')
logger = logging.getLogger('question_search_engine')
logger.setLevel(logging.INFO)
