import os
import unittest
from search_engine.question_search_engine import QuestionSearchEngine


class TestQuestionSearchEngine(unittest.TestCase):

    def setUp(self):
        self.corpus = [
            'How do I use Error handling in Java?',
            'Error Handling in Swift 3',
            'Java BufferedReader error',
            'If block error handling in bash',
            'java ATM program simulation with exception handling - no error neither full output?'
        ]
        self.question_search_engine = QuestionSearchEngine(self.corpus, fit_vectorizer=True)
        self.vocabulary_path = 'vocabulary.pkl'
        self.idf_vector_path = 'idf_vector.pkl'
        self.cache_path = 'cache.pkl'
        self.question_search_engine._tf_idf_vectorizer._vocabulary_path = self.vocabulary_path
        self.question_search_engine._tf_idf_vectorizer._idf_vector_path = self.idf_vector_path
        self.question_search_engine._tf_idf_vectorizer._cache_path = self.cache_path

    def tearDown(self):
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)
        if os.path.exists(self.vocabulary_path):
            os.remove(self.vocabulary_path)
        if os.path.exists(self.idf_vector_path):
            os.remove(self.idf_vector_path)

    def test_most_similar_small_corpus(self):
        # test number of results when top N is greater than corpus size
        query = 'Error handling in Java?'
        result = self.question_search_engine.most_similar(query, n=10)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 5)

    def test_most_similar_(self):
        query = 'Rukovanje greskama u Javi?'
        result = self.question_search_engine.most_similar(query, n=2)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_most_similar_no_results(self):
        query = 'Rukovanje greskama u Javi?'
        result = self.question_search_engine.most_similar(query, n=2)

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)


if __name__ == '__main__':
    unittest.main()
