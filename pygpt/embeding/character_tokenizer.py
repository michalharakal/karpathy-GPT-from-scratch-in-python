from typing import List

from pygpt.embeding.tokenizer import TextTokenizer


class CharTokenizer(TextTokenizer):
    def __len__(self):
        pass

    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, text: str) -> List[int]:
        """
        Tokenizes the given text into tokens.

        Parameters:
            text (str): The text to tokenize.

        Returns:
            List[str]: A list of tokens extracted from the text.
        """

        return [self.stoi[c] for c in text]  # encoder: take a string, output a list of integers

    def decode(self, tokens: List[int]) -> str:
        """
        Reconstructs the text from the tokens.

        Parameters:
            tokens (List[str]): The list of tokens to detokenize.

        Returns:
            str: The reconstructed text from the tokens.
        """
        return ''.join([self.itos[i] for i in tokens])  # decoder: take a list
