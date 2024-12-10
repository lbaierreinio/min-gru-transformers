import itertools
from transformers import AutoTokenizer

from models.MinGRUSquadQA import MinGRUSquadQA, MinGRUSquadQAConfig

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def test_model_config(model_class, config_class, vocab_size, n_layer, hidden_dim, classification_head_dim, bidirectional):
    """
    Function to instantiate the model, calculate parameter count, and return it.
    Args:
    - model_class: The class for the model.
    - config_class: The class for the model configuration.
    - vocab_size: Vocabulary size for the tokenizer.
    - n_layer: Number of layers in the model.
    - hidden_dim: Dimension of hidden states.
    - classification_head_dim: Dimension of the classification head.
    - bidirectional: Whether the model is bidirectional.

    Returns:
    - Parameter count of the model.
    """
    config = config_class(
        vocab_size=vocab_size,
        n_layer=n_layer,
        hidden_dim=hidden_dim,
        classification_head_dim=classification_head_dim,
        bidirectional=bidirectional,
    )
    model = model_class(config)
    return sum(p.numel() for p in model.parameters())

def generate_configs_with_constraints(
    target_size,
    model_class,
    config_class,
    vocab_size,
    n_layer_range,
    hidden_dim_range,
    bidirectional_values,
    tolerance,
):
    """
    Generate configurations with constraints: n_layers even, hidden_dim = classification_head_dim,
    and use early stopping to skip invalid configurations, resetting for each new `n_layer`.
    """
    valid_configs = []

    for n_layer in range(n_layer_range[0], n_layer_range[1] + 1, 2):  # ensure n_layer is even
        for hidden_dim in range(hidden_dim_range[0], hidden_dim_range[1] + 1, 2):  # ensure valid hidden_dim steps
            exceeded = False  # reset for every (n_layer, hidden_dim) combination
            for bidirectional in bidirectional_values:
                try:
                    # Calculate parameter count for this configuration
                    param_count = test_model_config(
                        model_class=model_class,
                        config_class=config_class,
                        vocab_size=vocab_size,
                        n_layer=n_layer,
                        hidden_dim=hidden_dim,
                        classification_head_dim=hidden_dim,  # Enforce hidden_dim = classification_head_dim
                        bidirectional=bidirectional,
                    )

                    # check if parameter count is within the tolerance range
                    if target_size * (1 - tolerance) <= param_count <= target_size * (1 + tolerance):
                        valid_configs.append(
                            {
                                "n_layer": n_layer,
                                "hidden_dim": hidden_dim,
                                "classification_head_dim": hidden_dim,
                                "bidirectional": bidirectional,
                                "param_count": param_count,
                            }
                        )

                    # if the parameter count exceeds the upper limit, skip further exploration for this hidden_dim
                    elif param_count > target_size * (1 + tolerance):
                        exceeded = True  # trigger early stopping
                        break
                except Exception as e:
                    print(f"Error with config (n_layer={n_layer}, hidden_dim={hidden_dim}): {e}")
                    continue

            if exceeded:
                break  # stop exploring this hidden_dim for current n_layer

    # return sorted configurations
    return sorted(valid_configs, key=lambda x: x["param_count"])

# define target model sizes and model
target_sizes = [16_000_000, 24_000_000, 32_000_000, 40_000_000]
model_class = MinGRUSquadQA  
config_class = MinGRUSquadQAConfig
vocab_size = tokenizer.vocab_size  

# generate configs for each target size with constraints
all_constrained_configs = {
    target_size: generate_configs_with_constraints(
        target_size,
        model_class=model_class,
        config_class=config_class,
        vocab_size=vocab_size,
        n_layer_range=(10, 30),
        hidden_dim_range=(300, 1000),
        bidirectional_values=[True],
        tolerance=0.05,
    )
    for target_size in target_sizes
}

for target_size, configs in all_constrained_configs.items():
    print(f"Target size: {target_size} parameters")
    for config in configs: 
        print(config)
    print("\n")