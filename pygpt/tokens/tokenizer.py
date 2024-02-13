from abc import ABC, abstractmethod


class TextTokenizer(ABC):
    @abstractmethod
    def encode(self, text):
        """
        Tokenizes the given text into tokens.

        Parameters:
            text (str): The text to tokenize.

        Returns:
            List[str]: A list of tokens extracted from the text.
        """
        pass

    @abstractmethod
    def decode(self, tokens):
        """
        Reconstructs the text from the tokens.

        Parameters:
            tokens (List[str]): The list of tokens to detokenize.

        Returns:
            str: The reconstructed text from the tokens.
        """
        pass

    @abstractmethod
    def __len__(self):
        """Return the number of items in the dataset."""
        pass

