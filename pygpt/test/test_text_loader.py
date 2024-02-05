from unittest import TestCase

from pygpt.dataset.text_loader import TextFileDataset

INPUT_TXT_SAMPLE = """First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

First Citizen:
You are all resolved rather to die than to famish?
"""


class TestTextFileDataset(TestCase):
    def test_load_first_148_chars(self):
        dl = TextFileDataset("../../nanogpt-lecture/input.txt")
        self.assertIs(len(dl.data[:148]), len(INPUT_TXT_SAMPLE))

    def test_load_has_over_million_characters(self):
        dl = TextFileDataset("../../nanogpt-lecture/input.txt")
        assert len(dl) == 1115394, " has over million chars"
