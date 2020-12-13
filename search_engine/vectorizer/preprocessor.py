import re
from typing import *


class Singleton(type):
    """Metaclass for classes that use Singleton pattern"""

    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class QuestionPreprocessor(metaclass=Singleton):
    """Preprocessor for question content."""

    def _remove_non_word_chars(self, text: str) -> Optional[str]:
        """Remove all non word characters from input text.

        Args:
            text: text (question content)

        Returns:
            Cleaned text
        """
        regex = re.compile('[^a-zA-Z ]+')
        return regex.sub('', text)

    def _normalize(self, text: str) -> str:
        """Normalize input text, converting all letters into lowercase.

        Args:
            text: text (question content)

        Returns:
            Lowercase text
        """
        return text.lower()

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize input text.

        Args:
            text: text (question content)

        Returns:
            List of textual tokens.
        """
        return text.split()

    def preprocess(self, questions: Sequence[str]) -> Sequence[List[str]]:
        """Preprocess sequence of raw questions.

        Args:
            questions: sequence of raw question corpus

        Returns:
            Sequence of preprocessed questions, where each question is represented
            as list of its textual tokens.
        """
        preprocessed_questions = []
        for question in questions:
            question = self._remove_non_word_chars(question)
            question = self._normalize(question)
            tokens = self._tokenize(question)
            tokens = [token for token in tokens if len(token) > 1 or len(token) == 1 and token == 'c']
            preprocessed_questions.append(tokens)

        return preprocessed_questions
