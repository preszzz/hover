# Hyperparameter Tuning for AST Model

This directory contains tools for hyperparameter optimization of the Audio Spectrogram Transformer (AST) model using [Optuna](https://optuna.org/).

## Requirements

Before running the hyperparameter tuning script, ensure you have the required dependencies:

```bash
uv add optuna
```

## Usage

### Running Hyperparameter Tuning

To start the hyperparameter tuning process:

```bash
uv run hyperparameter/tune.py
```

## Hyperparameters Optimized

The tuning process optimizes the following hyperparameters:

1. **Learning Rate**: Range from 1e-5 to 1e-3 (log scale)
2. **Batch Size**: Options are 16, 32, or 64
3. **Freeze Backbone**: Whether to freeze the backbone layers (True/False)
4. **Dropout Rate**: Range from 0.1 to 0.5
5. **Weight Decay**: Range from 1e-6 to 1e-3 (log scale)
6. **Optimizer**: AdamW or Adam

## Results

The tuning process generates several visualizations to help understand parameter importance:

1. **Optimization History**: Shows how the objective value improved over trials
2. **Parallel Coordinate Plot**: Visualizes relationships between hyperparameters and performance
3. **Parameter Importance**: Shows which parameters had the biggest impact
4. **Slice Plot**: Shows the effect of individual parameters on performance


## Additional Information

- Logs are saved in the `hyperparameter` directory with timestamps
- The tuning process uses a pruner to stop unpromising trials early
- Each trial runs for a maximum of 5 epochs to save computation time