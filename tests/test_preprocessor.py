import string
import unittest

from search_engine.vectorizer.preprocessor import QuestionPreprocessor


class TestPreprocessor(unittest.TestCase):

    def setUp(self):
        self.preprocessor_1 = QuestionPreprocessor()

    def test_is_singleton(self):
        preprocessor_2 = QuestionPreprocessor()
        self.assertEqual(id(self.preprocessor_1), id(preprocessor_2))

    def test_remove_non_word_chars(self):
        text_1 = 'Some text 28 and speci@al character!s!'
        self.assertIsInstance(self.preprocessor_1._remove_non_word_chars(text_1), str)
        self.assertEqual(self.preprocessor_1._remove_non_word_chars(text_1), 'Some text  and special characters')

        text_2 = ''
        self.assertIsInstance(self.preprocessor_1._remove_non_word_chars(text_2), str)
        self.assertEqual(self.preprocessor_1._remove_non_word_chars(text_2), '')

        text_3 = string.digits + string.punctuation
        self.assertIsInstance(self.preprocessor_1._remove_non_word_chars(text_3), str)
        self.assertEqual(self.preprocessor_1._remove_non_word_chars(text_3), '')

        text_4 = 'Some text'
        self.assertIsInstance(self.preprocessor_1._remove_non_word_chars(text_4), str)
        self.assertEqual(self.preprocessor_1._remove_non_word_chars(text_4), 'Some text')

    def test_normalize(self):
        text_1 = 'Some TexT'
        self.assertIsInstance(self.preprocessor_1._normalize(text_1), str)
        self.assertEqual(self.preprocessor_1._normalize(text_1), 'some text')

        text_2 = 'SOME TEXT'
        self.assertIsInstance(self.preprocessor_1._normalize(text_2), str)
        self.assertEqual(self.preprocessor_1._normalize(text_2), 'some text')

        text_3 = 'some text'
        self.assertIsInstance(self.preprocessor_1._normalize(text_3), str)
        self.assertEqual(self.preprocessor_1._normalize(text_3), 'some text')

        text_4 = ''
        self.assertIsInstance(self.preprocessor_1._normalize(text_4), str)
        self.assertEqual(self.preprocessor_1._normalize(text_4), '')

    def test_tokenize(self):
        text_1 = 'Some text with several tokens'
        self.assertIsInstance(self.preprocessor_1._tokenize(text_1), list)
        self.assertEqual(len(self.preprocessor_1._tokenize(text_1)), 5)
        self.assertEqual(self.preprocessor_1._tokenize(text_1), ['Some', 'text', 'with', 'several', 'tokens'])
        for token in self.preprocessor_1._tokenize(text_1):
            self.assertIsInstance(token, str)

        text_2 = 'some     text'
        self.assertIsInstance(self.preprocessor_1._tokenize(text_2), list)
        self.assertEqual(len(self.preprocessor_1._tokenize(text_2)), 2)
        self.assertEqual(self.preprocessor_1._tokenize(text_2), ['some', 'text'])
        for token in self.preprocessor_1._tokenize(text_2):
            self.assertIsInstance(token, str)

        text_3 = '    '
        self.assertIsInstance(self.preprocessor_1._tokenize(text_3), list)
        self.assertEqual(len(self.preprocessor_1._tokenize(text_3)), 0)

    def text_preprocess(self):
        text_1 = 'First sentence   in corpus!'
        text_2 = '\'Second senten@ce here?!'
        texts = [text_1, text_2]

        self.assertIsInstance(self.preprocessor_1.preprocess(texts), list)
        self.assertIsInstance(len(self.preprocessor_1.preprocess(texts)), 2)

        sequences = self.preprocessor_1.preprocess(texts)
        for sequence in sequences:
            self.assertIsInstance(sequence, list)
            for token in sequence:
                self.assertIsInstance(token, str)

        self.assertEqual(len(sequences[0]), 4)
        self.assertEqual(sequences[0], ['first', 'sentence', 'in', 'corpus'])
        self.assertEqual(len(sequences[1]), 3)
        self.assertEqual(sequences[1], ['second', 'sentence', 'here'])


if __name__ == '__main__':
    unittest.main()
