import unittest
import numpy as np

from utils import *
from search_engine.vectorizer.preprocessor import QuestionPreprocessor
from search_engine.vectorizer.tf_idf_vectorizer import TfIdfVectorizer


class TestTfIdfVectorizer(unittest.TestCase):

    def setUp(self):
        self.corpus = [
            'This is the first document.',
            'This document is the second document.',
            'And this is the third one.',
            'Is this the first document?'
        ]
        self.cache_path = 'cache.pkl'
        self.vocabulary_path = 'vocabulary.pkl'
        self.idf_vector_path = 'idf_vector.pkl'

    def tearDown(self):
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)
        if os.path.exists(self.vocabulary_path):
            os.remove(self.vocabulary_path)
        if os.path.exists(self.idf_vector_path):
            os.remove(self.idf_vector_path)

    def test_init(self):
        vectorizer = TfIdfVectorizer(cache_path=self.cache_path)

        # check attribute types and values after initialization
        self.assertIsInstance(vectorizer._use_cache, bool)
        self.assertEqual(vectorizer._use_cache, True)

        self.assertIsInstance(vectorizer._cache_path, str)
        self.assertEqual(vectorizer._cache_path, 'cache.pkl')

        self.assertIsInstance(vectorizer._vocabulary_path, str)
        self.assertIsInstance(vectorizer._idf_vector_path, str)
        self.assertIsInstance(vectorizer._vocabulary_size, int)
        self.assertIsInstance(vectorizer._preprocessor, QuestionPreprocessor)
        self.assertEqual(vectorizer.questions, None)

    def test_fit(self):
        # test for methods: __init__, fit, _build_vocabulary, _vectorize_questions and _save
        vectorizer = TfIdfVectorizer(cache_path=self.cache_path)
        vectorizer._vocabulary_path = self.vocabulary_path
        vectorizer._idf_vector_path = self.idf_vector_path

        # check return value of fit method
        self.assertEqual(vectorizer.fit(self.corpus), None)

        vocabulary = {
            'and': 0,
            'document': 1,
            'first': 2,
            'is': 3,
            'one': 4,
            'second': 5,
            'the': 6,
            'third': 7,
            'this': 8
        }
        # check created vocabulary
        self.assertIsInstance(vectorizer._vocabulary, dict)
        self.assertEqual(vectorizer._vocabulary, vocabulary)
        self.assertEqual(vectorizer._vocabulary_size, len(vocabulary))

        idf_vector = [1.91629073, 1.22314355, 1.51082562, 1., 1.91629073, 1.91629073, 1., 1.91629073, 1.]
        # check idf vector
        self.assertIsInstance(vectorizer._idf_vector, np.ndarray)
        self.assertEqual(np.array_equal(vectorizer._idf_vector, idf_vector), True)

        # check vocabulary serialization
        vocabulary_deserialized = deserialize_data(self.vocabulary_path)
        self.assertIsInstance(vocabulary_deserialized, dict)
        self.assertEqual(vectorizer._vocabulary, vocabulary_deserialized)
        # check idf vector serialization
        idf_vector_deserialized = deserialize_data(self.idf_vector_path)
        self.assertIsInstance(idf_vector_deserialized, np.ndarray)
        self.assertEqual(np.array_equal(vectorizer._idf_vector, idf_vector_deserialized), True)

        vectorized_corpus = np.asarray([
            np.asarray([0., 0.46979139, 0.58028582, 0.38408524, 0., 0., 0.38408524, 0., 0.38408524]),
            np.asarray([0., 0.6876236, 0., 0.28108867, 0., 0.53864762, 0.28108867, 0., 0.28108867]),
            np.asarray([0.51184851, 0., 0., 0.26710379, 0.51184851, 0., 0.26710379, 0.51184851, 0.26710379]),
            np.asarray([0., 0.46979139, 0.58028582, 0.38408524, 0., 0., 0.38408524, 0., 0.38408524])
        ])
        # check vectorized corpus - type and dimensions
        self.assertIsInstance(vectorizer.questions, np.ndarray)
        self.assertEqual(len(vectorizer.questions), len(self.corpus))

        # check type, dimensions, number of nonzero elements (Tf-Idf scores) and Tf-Idf embedding for each question from corpus
        for i, tf_idf_embedding in enumerate(vectorizer.questions):
            # type - numpy array
            self.assertIsInstance(tf_idf_embedding, np.ndarray)
            # tf-idf embedding size
            self.assertEqual(len(tf_idf_embedding), vectorizer._vocabulary_size)
            # number of nonzero elements in tf-idf vector
            preprocessed_question_tokens = vectorizer._preprocessor.preprocess([self.corpus[i]])[0]
            words_in_vocabulary = [word for word in preprocessed_question_tokens if word in vectorizer._vocabulary]
            self.assertEqual(np.count_nonzero(tf_idf_embedding), len(set(words_in_vocabulary)))
            # tf-idf embedding
            self.assertEqual(np.array_equal(np.round(tf_idf_embedding, 8), vectorized_corpus[i]), True)

    def test_transform(self):
        # test for methods: transform, _vectorize_questions and _load

        query_question = 'Is this first or second document?'

        # transform with previous vectorizer.fit()
        vectorizer = TfIdfVectorizer(cache_path=self.cache_path)
        vectorizer._vocabulary_path = self.vocabulary_path
        vectorizer._idf_vector_path = self.idf_vector_path
        vectorizer.fit(self.corpus)

        vectorized_query = np.asarray([0., 0.39787085, 0.49144966, 0.32528549, 0., 0.62334157, 0., 0., 0.32528549])
        # check type
        tranfsormed_query = vectorizer.transform([query_question])[0]
        self.assertIsInstance(tranfsormed_query, np.ndarray)
        # check dimensions of Tf-Idf embedding
        self.assertEqual(len(tranfsormed_query), vectorizer._vocabulary_size)
        # check number of nonzero elements (Tf-Idf scores)
        preprocessed_question_tokens = vectorizer._preprocessor.preprocess([query_question])[0]
        words_in_vocabulary = [word for word in preprocessed_question_tokens if word in vectorizer._vocabulary]
        self.assertEqual(np.count_nonzero(tranfsormed_query), len(set(words_in_vocabulary)))
        # check Tf-Idf embedding
        self.assertEqual(np.array_equal(np.round(tranfsormed_query, 8), vectorized_query), True)

        # transform without previous vectorizer.fit()
        vectorizer = TfIdfVectorizer(cache_path=self.cache_path)
        vectorizer._vocabulary_path = self.vocabulary_path
        vectorizer._idf_vector_path = self.idf_vector_path

        # vocabulary, idf vector and questions are None
        self.assertEqual(vectorizer._vocabulary, None)
        self.assertEqual(vectorizer._idf_vector, None)
        self.assertEqual(vectorizer.questions, None)

        tranfsormed_query = vectorizer.transform([query_question])[0]

        # check type
        self.assertIsInstance(tranfsormed_query, np.ndarray)
        # check dimensions of Tf-Idf embedding
        self.assertEqual(len(tranfsormed_query), vectorizer._vocabulary_size)
        # check number of nonzero elements (Tf-Idf scores)
        preprocessed_question_tokens = vectorizer._preprocessor.preprocess([query_question])[0]
        words_in_vocabulary = [word for word in preprocessed_question_tokens if word in vectorizer._vocabulary]
        self.assertEqual(np.count_nonzero(tranfsormed_query), len(set(words_in_vocabulary)))
        # check Tf-Idf embedding
        self.assertEqual(np.array_equal(np.round(tranfsormed_query, 8), vectorized_query), True)


if __name__ == '__main__':
    unittest.main()
