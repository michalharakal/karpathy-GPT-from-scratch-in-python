from abc import ABC, abstractmethod


class Dataset(ABC):
    @abstractmethod
    def __len__(self):
        """Return the number of items in the dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """Return a single item by index."""
        pass

    @abstractmethod
    def __iter__(self):
        """Return an iterator over the items in the dataset."""
        pass
