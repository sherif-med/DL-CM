
import torch
import yaml

# Define a custom constructor for the `!tensor` tag
def tensor_constructor(loader, node):
    """Construct a tensor from a scalar or a sequence in YAML."""
    # Determine if the node is a single value or a sequence
    if isinstance(node, yaml.ScalarNode):
        # Handle scalar values
        value = loader.construct_scalar(node)
        # Convert the value to a float or a sequence of floats and create a tensor
        tensor_value = [float(value)]
    elif isinstance(node, yaml.SequenceNode):
        # Handle sequences
        value = loader.construct_sequence(node)
        # Convert the list of values to the appropriate dtype
        tensor_value = [float(i) for i in value]
    else:
        raise TypeError(f"Unsupported YAML node type: {type(node)}. Only scalar and sequence nodes are supported.")
    
    return torch.tensor(tensor_value, dtype=torch.float32)

# Add the constructor to the PyYAML loader
yaml.SafeLoader.add_constructor('!tensor', tensor_constructor)

def open_config_file(conf_file_path: str) -> dict :
    with open(conf_file_path, "r") as f:
        config = yaml.safe_load(f)
        return config

def load_named_entity(registry, entity_config):
    entity_cls = registry.get(entity_config.get("name"))
    entity_params = entity_config.get("params")
    return entity_cls(**entity_params)