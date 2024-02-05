from unittest import TestCase

from pygpt.embeding.character_tokenizer import CharTokenizer
from pygpt.test.test_text_loader import INPUT_TXT_SAMPLE


class TestCharTokenizer(TestCase):
    def test_encode(self):
        tokenizer = CharTokenizer(INPUT_TXT_SAMPLE)
        encoded = tokenizer.encode("hii there")
        assert encoded == [17, 18, 18, 1, 27, 17, 15, 25, 15], "decode text should be a vector"

    def test_decode(self):
        tokenizer = CharTokenizer(INPUT_TXT_SAMPLE)
        decoded = tokenizer.decode([17, 18, 18, 1, 27, 17, 15, 25, 15])
        assert decoded == "hii there", "decode text should be a text"
