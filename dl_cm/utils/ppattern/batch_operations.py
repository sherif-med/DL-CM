import torch
import itertools

def merge_multiple_batches(batches):
    """
    Merges multiple batches into a single batch.

    This function takes a list of batches, where each batch is a dictionary mapping keys to values,
    and merges them into a single batch. For keys mapping to torch.Tensor objects, it concatenates
    them along the 0th dimension. For other types of values, it combines them into a single list.

    Parameters:
    - batches (list of dict): A list where each element is a batch represented as a dictionary.
      Each dictionary maps keys to values, where values can be of any type, but torch.Tensor
      objects are specifically handled by concatenation.

    Returns:
    - dict: A single merged dictionary representing the combined batch. For each key, if the value
      is of type torch.Tensor, the returned dictionary contains a concatenated tensor of all batches'
      tensors for that key. For keys associated with other types of values, it contains a list of
      values combined from all batches.

    Example:
    >>> batch1 = {'a': torch.tensor([1, 2]), 'b': [1, 2]}
    >>> batch2 = {'a': torch.tensor([3, 4]), 'b': [3, 4]}
    >>> merged_batch = merge_multiple_batches([batch1, batch2])
    >>> merged_batch['a']
    tensor([1, 2, 3, 4])
    >>> merged_batch['b']
    [1, 2, 3, 4]

    Note:
    - This function is designed to work with batches containing torch.Tensor objects and/or
      collections of other types. It's particularly useful in data processing pipelines where
      batches of data need to be combined before being fed into a model for training or inference.
    """
    merged_dict = {}
    for key, value in batches[0].items():
        # Concatenate tensors from all dictionaries for each key
        if isinstance(value, torch.Tensor):
            merged_dict[key] = torch.cat([batch[key] for batch in batches], dim=0)
        else:
            merged_dict[key]=list(itertools.chain(*[batch[key] for batch in batches]))
    return merged_dict

def split_into_batches(merged_dict, batch_size):
    """
    Splits a dictionary of merged data into batches of a specified size.

    This function takes a dictionary where each key maps to a collection of data items
    (e.g., a list or a torch.Tensor) and splits this data into smaller batches. Each batch
    is a dictionary with the same keys as the input dictionary but with values containing
    only a subset of the items, corresponding to the batch size.

    Parameters:
    - merged_dict (dict): A dictionary with keys mapping to collections of data items.
      Each collection must have the same length.
    - batch_size (int): The desired number of items in each batch.

    Returns:
    - list of dict: A list of dictionaries, where each dictionary represents a batch of
      data split according to the specified batch_size.

    Example:
    >>> merged_dict = {'a': torch.arange(10), 'b': list(range(10))}
    >>> batches = split_into_batches(merged_dict, 4)
    >>> len(batches)
    3
    >>> batches[0]['a']
    tensor([0, 1, 2, 3])
    >>> batches[0]['b']
    [0, 1, 2, 3]

    Note:
    - The last batch may contain fewer items than batch_size if the total number of items
      in merged_dict collections is not perfectly divisible by batch_size.
    """
    # Check if all elements have the same length
    lengths = [len(value) if not isinstance(value, torch.Tensor) else value.size(0) for value in merged_dict.values()]
    if len(set(lengths)) > 1:
        raise ValueError("All values in merged_dict must have the same number of elements.")

    total_length = lengths[0]
    batches = []

    for start_idx in range(0, total_length, batch_size):
        batch_dict = {key: value[start_idx:start_idx+batch_size]
                      for key, value in merged_dict.items()}
        batches.append(batch_dict)

    return batches

def get_batch_size_from_items_dict(items_dict):
    """
    Retrieves the batch size from a dictionary of items based on the first torch.Tensor found.

    This function iterates through the values of the input dictionary looking for the first
    instance of a torch.Tensor. It returns the size of the 0th dimension of the tensor, which
    is typically used to represent the batch size in machine learning data structures.

    Parameters:
    - items_dict (dict): A dictionary where the values are expected to be collections of items,
      including torch.Tensor objects among potentially other types.

    Returns:
    - int: The size of the 0th dimension of the first torch.Tensor found, representing the batch size.

    Raises:
    - ValueError: If no torch.Tensor object is found in the values of the dictionary, or if tensors
      with inconsistent sizes are found, indicating an error in batch formation.

    Example:
    >>> items_dict = {'features': torch.rand(32, 3, 64, 64), 'labels': torch.randint(0, 10, (32,))}
    >>> batch_size = get_batch_size_from_items_dict(items_dict)
    >>> batch_size
    32

    Note:
    - This function assumes that all tensors in the dictionary should have the same size along their
      0th dimension, which is a common assumption in batched data processing. If tensors with different
      0th dimension sizes are found, an exception is raised to highlight this inconsistency.
    """
    batch_sizes = [v.size(0) for k, v in items_dict.items() if isinstance(v, torch.Tensor)]
    
    if not batch_sizes:
        raise ValueError("Unable to get batch size: No torch.Tensor objects found in items_dict.")
    
    if len(set(batch_sizes)) > 1:
        raise ValueError("Inconsistent batch sizes found among tensors in items_dict.")

    return batch_sizes[0]