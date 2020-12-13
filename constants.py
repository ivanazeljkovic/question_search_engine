import os

# Global directories
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
LOGS_DIR_PATH = os.path.join(ROOT_PATH, 'logs')
MODEL_DIR_PATH = os.path.join(ROOT_PATH, 'model')
DATA_DIR_PATH = os.path.join(ROOT_PATH, 'data')
RAW_DATA_DIR_PATH = os.path.join(DATA_DIR_PATH, 'raw')
CACHE_DIR_PATH = os.path.join(DATA_DIR_PATH, 'cache')

# Raw data
RAW_DATA_EXTENSION = 'json'
RAW_DATA_FILE_PATH = os.path.join(RAW_DATA_DIR_PATH, f'questions.{RAW_DATA_EXTENSION}')
QUESTION_ID_KEY = 'id'
QUESTION_CONTENT_KEY = 'question'

# Data cache
TF_IDF_CACHE_PATH = os.path.join(CACHE_DIR_PATH, 'tf-idf_cache.pkl')

# Tf-Idf
VOCABULARY_SIZE = 3000
VOCABULARY_PATH = os.path.join(MODEL_DIR_PATH, 'vocabulary.pkl')
IDF_VECTOR_PATH = os.path.join(MODEL_DIR_PATH, 'idf_vector.pkl')
