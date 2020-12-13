from ast import literal_eval

from utils import *
from constants import *
from search_engine.question_search_engine import QuestionSearchEngine


def load_data(path: str) -> Dict[int, str]:
    """Load question corpus stored in JSON file. Duplicated questions are ignored.

    Args:
        path: path to the corpus

    Returns:
         Question corpus in dictionary form, where question content is the key and question ID is the value
    """
    data = {}
    with open(path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            row_data = literal_eval(line)
            data[row_data[QUESTION_CONTENT_KEY]] = row_data[QUESTION_ID_KEY]

    return data


if __name__ == '__main__':
    data = load_data(RAW_DATA_FILE_PATH)
    search_engine = QuestionSearchEngine(list(data.keys()), fit_vectorizer=True)

    while True:
        query = input('>>> ')
        result = search_engine.most_similar(query)
        for similarity_score, question in result:
            print(similarity_score, data[question], question)
