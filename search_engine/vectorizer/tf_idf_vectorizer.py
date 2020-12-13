import numpy as np
from collections import Counter

from utils import *
from settings import *
from constants import *
from search_engine.vectorizer.preprocessor import QuestionPreprocessor


class TfIdfVectorizer:
    """Tf-Idf vectorizer for question embedding.

    Attributes:
        _use_cache (bool): flag that indicates should vectorized question corpus be serialized or not
        _cache_path (str): path to the file where vectorized question corpus will be serialized
        _vocabulary_path (str): path to the file where vocabulary for Bag-Of-Words model will be serialized
        _idf_vector_path (str): path to the file where vector with IDF scores for all words from vocabulary
                                will be stored
        _vocabulary_size (int): size of vocabulary for Bag-Of-Words model
        _vocabulary (dict): vocabulary for Bag-Of-Words model
        _idf_vector (np.ndarray): vector with IDF scores for all words from vocabulary
        _preprocessor (QuestionPreprocessor): singleton preprocessor for question content
        questions (np.ndarray): vectorized question corpus
    """

    def __init__(self, use_cache: bool = True, cache_path: str = TF_IDF_CACHE_PATH) -> None:
        """Initialize vectorizer that uses Tf-Idf approach (document level embedding).

        Args:
            use_cache: flag that indicates should vectorized question corpus be serialized or not
            cache_path: path to the file where vectorized question corpus will be serialized

        Returns:
            no value
        """
        self._use_cache = use_cache
        self._cache_path = cache_path

        self._vocabulary_path = VOCABULARY_PATH
        self._idf_vector_path = IDF_VECTOR_PATH
        self._vocabulary_size = VOCABULARY_SIZE
        self._vocabulary = None
        self._idf_vector = None

        self._preprocessor = QuestionPreprocessor()
        self.questions = None

    def fit(self, questions: Sequence[str]) -> None:
        """Fit vectorizer with sequence of raw questions.

        Args:
            questions: sequence of raw question corpus

        Returns:
            no value
        """
        print('----> Fitting Tf-Idf vectorizer\n\n')

        questions = self._preprocessor.preprocess(questions)
        self._build_vocabulary(questions)
        self.questions = self._vectorize_questions(questions)

        # serialize vocabulary and idf vector
        check_does_dir_exist(path=MODEL_DIR_PATH, create_dir=True)
        serialize_data(data=self._vocabulary, path=self._vocabulary_path)
        serialize_data(data=self._idf_vector, path=self._idf_vector_path)

        # serialize vectorized questions
        if self._use_cache:
            self._save()

        logger.info(f'Fitting Tf-Idf vectorizer on corpus with {len(questions)} questions finished')

    def transform(self, questions: Sequence[str]) -> np.ndarray:
        """Transform sequence of raw questions into vector representation, with Tf-Idf scores.

        Args:
            questions: sequence of raw question corpus

        Returns:
            Sequence of vectorized questions as numpy array of (N, D) shape where
            N is number of questions in given corpus and D is vocabulary size.
        """
        questions = self._preprocessor.preprocess(questions)

        # deserialize vocabulary and idf vector
        if self._vocabulary is None:
            self._vocabulary = deserialize_data(path=self._vocabulary_path)
            self._vocabulary_size = len(self._vocabulary)
        if self._idf_vector is None:
            self._idf_vector = deserialize_data(path=self._idf_vector_path)
        if self.questions is None:
            self._load()

        return self._vectorize_questions(questions)

    def _build_vocabulary(self, questions: Sequence[List[str]]) -> None:
        """Build vocabulary used in Bag-Of-Words model using sequence of question tokens.

        Args:
            questions: sequence of tokens for question corpus

        Returns:
            no value
        """
        self._vocabulary = Counter()
        idf_map = Counter()
        for question_tokens in questions:
            self._vocabulary.update(question_tokens)
            unique_tokens = set(question_tokens)
            idf_map.update(unique_tokens)

        # take N most frequent words
        self._vocabulary = dict(self._vocabulary.most_common(self._vocabulary_size))
        # update vocabulary_size if there is less words in vocabulary
        vocabulary_size = len(self._vocabulary)
        if vocabulary_size < self._vocabulary_size:
            self._vocabulary_size = vocabulary_size

        # transform vocabulary in form of word-index pairs
        self._vocabulary = dict(zip(sorted(self._vocabulary.keys()), range(self._vocabulary_size)))

        num_of_questions = len(questions)
        idf_func = lambda value: np.round(np.log((num_of_questions + 1)/(value + 1)) + 1, decimals=8)
        # calculate idf value for each word in vocabulary
        self._idf_vector = np.asarray([idf_func(idf_map[word]) for word in self._vocabulary.keys()])

        logger.info('Building vocabulary and IDF vector finished')

    def _vectorize_questions(self, questions: Sequence[List[str]]) -> np.ndarray:
        """Transform sequence of question tokens into vector representation, using calculated Tf-Idf scores.

        Args:
            questions: sequence of tokens for question corpus

        Returns:
            Sequence of vectorized questions as numpy array of (N, D) shape where
            N is number of questions in sequence and D is vocabulary size.
        """
        vectorized_questions = []

        for question_tokens in questions:
            # zero question vector
            question_vector = np.zeros(self._vocabulary_size)
            nonzero_array = False

            token_occurences = Counter(question_tokens)
            num_of_tokens = len(question_tokens)
            for token in set(question_tokens):
                if token in self._vocabulary:
                    token_index = self._vocabulary[token]
                    nonzero_array = True
                    # calculate tf value
                    tf_value = token_occurences[token]/num_of_tokens
                    # calculate tf-idf value
                    question_vector[token_index] = self._idf_vector[token_index] * tf_value

            # normalize vector
            if nonzero_array:
                question_vector /= np.sqrt(np.sum(question_vector ** 2))

            vectorized_questions.append(question_vector)

        return np.asarray(vectorized_questions)

    def _load(self) -> None:
        """Deserialize vectorized question corpus.

        Returns:
            no value
        """
        print('----> Deserialization of vectorized corpus\n\n')
        self.questions = deserialize_data(self._cache_path)

        logger.info('Deserialization of vectorized corpus finished')

    def _save(self) -> None:
        """Serialize vectorized question corpus.

        Returns:
            no value
        """
        print(f'----> Serialization of vectorized corpus {self.questions.shape}\n\n')
        serialize_data(self.questions, self._cache_path)

        logger.info('Serialization of vectorized corpus finished')
