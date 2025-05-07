"""Audio Spectrogram Transformer (AST) model definition."""

from transformers import ASTFeatureExtractor, ASTForAudioClassification
import torch 
import torch.nn as nn 

# Define the pre-trained model checkpoint
import config

def build_transformer_model(num_classes: int, model_checkpoint: str):
    """
    Loads a pre-trained Audio Spectrogram Transformer (AST) model
    from Hugging Face for PyTorch.

    Args:
        num_classes (int): The number of output classes for the classification layer.
        model_checkpoint (str): The Hugging Face AST model identifier.

    Returns:
        torch.nn.Module: The PyTorch AST model with a potentially resized classification head.
    """
    # Load the pre-trained AST model
    # Set ignore_mismatched_sizes=True to allow replacing the classification head
    # if the number of classes differs from the pre-trained model.
    model = ASTForAudioClassification.from_pretrained(
        model_checkpoint,
        cache_dir=config.CACHE_DIR,
        num_labels=num_classes,
        ignore_mismatched_sizes=True, # Allows replacing the classification head
    )
    return model

def get_feature_extractor(model_checkpoint: str, target_sr: int):
    """
    Loads the AST feature extractor.

    Args:
        model_checkpoint (str): The Hugging Face AST model identifier.

    Returns:
        ASTFeatureExtractor: The AST feature extractor instance.
    """
    # Load the feature extractor
    try:
        feature_extractor = ASTFeatureExtractor.from_pretrained(
            model_checkpoint,
            sampling_rate=target_sr
        )
        return feature_extractor
    except Exception as e:
        print(f"Error loading feature extractor: {e}")
        raise

# Example usage (optional, for testing)
if __name__ == '__main__':
    try:
        print("--- AST Model Example ---")
        # Example: Get the feature extractor
        extractor = get_feature_extractor(config.MODEL_CHECKPOINT, config.TARGET_SAMPLE_RATE)
        print(f"Loaded AST feature extractor: {extractor}")
        print(f"Feature Extractor Sampling Rate: {extractor.sampling_rate}")
        print(f"Feature Extractor Max Length: {extractor.max_length}")
        print(f"Feature Extractor Num Mel Bins: {extractor.nb_mel_bins}")

        # Example: Build the model for a specific number of classes
        num_example_classes = 2
        ast_model = build_transformer_model(num_example_classes, config.MODEL_CHECKPOINT)

        # Example of creating dummy input matching extractor specs
        dummy_batch_size = 2
        dummy_input_features = torch.randn(
            dummy_batch_size,
            extractor.num_mel_bins,
            extractor.max_length
        )
        # For AST, input is passed as 'input_values'
        dummy_input_dict = {"input_values": dummy_input_features}

        # Example forward pass
        ast_model.eval()
        with torch.no_grad():
            outputs = ast_model(**dummy_input_dict)
            logits = outputs.logits

        print(f"Dummy input shape (batch, bins, time_frames): {dummy_input_features.shape}")
        print(f"Output logits shape (batch, num_classes): {logits.shape}")
        print(f"Output logits sample: {logits[0]}")

    except Exception as e:
        print(f"Could not run AST example: {e}")
        print("Ensure the MODEL_CHECKPOINT is correct and dependencies (torch, transformers) are installed.")
        import traceback
        traceback.print_exc() 