from pygpt.dataset.loader import Dataset


class TextFileDataset(Dataset):
    def __init__(self, file_path):
        """Initialize the dataset with the path to a text file."""
        self.data = self._load_data(file_path)

    @staticmethod
    def _load_data(file_path):
        """Private method to load data from a text file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            data = file.read()
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self.data):
            raise IndexError("Index out of bounds")
        return self.data[idx]

    def __iter__(self):
        for item in self.data:
            yield item
