## Torch proposes a function to do this

import numpy as np
from . import CompositionDataset

def split_subdatasets_random(parent_dataset, fractions):
    """
    Split a parent dataset into non-overlapping subdatasets with random indices,
    based on specified fractions.

    Parameters:
    - parent_dataset: The dataset to be split.
    - fractions: A list of fractions indicating the size of each subdataset relative to the parent dataset.

    Returns:
    - A list of subdatasets, each being a portion of the parent dataset with random indices.
    """
    assert sum(fractions) <= 1, "The sum of the fractions must not exceed 1."

    total_length = len(parent_dataset)
    shuffled_indices = np.random.permutation(total_length)
    
    # Calculate the number of elements for each fraction
    counts = [int(frac * total_length) for frac in fractions]
    
    # Calculate the start index for each subdataset
    starts = np.cumsum([0] + counts)
    ends = starts[1:]

    subdatasets = []
    for start, end in zip(starts, ends):
        # Extract the indices for this subdataset
        sub_indices = shuffled_indices[start:end]
        
        # Define a subdataset class that uses these indices
        class SubDataset(CompositionDataset):
            def __init__(self, parent, indices):
                super().__init__(parent, copy_parent=False)
                self.indices = indices
            
            def __len__(self):
                return len(self.indices)
            
            def __getitem__(self, idx):
                if idx < 0 or idx >= len(self):
                    raise IndexError("Index out of range")
                real_idx = self.indices[idx]
                return self.parent_dataset.__getitem__(real_idx)
        
        # Create a new subdataset instance for the current subset of indices
        subdatasets.append(SubDataset(parent_dataset, sub_indices))
    
    return subdatasets