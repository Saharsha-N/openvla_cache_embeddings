# OpenVLA Embedding Extraction Script

This script extracts and caches embeddings from the residual stream at different token positions in OpenVLA models. It supports hook-based extraction, multiple output formats, and visualization capabilities.

## Features

- **Hook-based extraction**: Registers forward hooks on transformer layers to capture hidden states
- **Flexible layer selection**: Extract from specific layers or all layers
- **Token position control**: Extract embeddings from specific token positions
- **Multiple output formats**: Save embeddings as PyTorch (.pt) or NumPy (.npy) files
- **Visualization**: Generate PCA plots of extracted embeddings
- **Comprehensive metadata**: Save extraction parameters and embedding information
- **Test mode**: Mock extraction for testing script functionality
- **Error handling**: Timeout protection for CPU inference

## Installation

Ensure you have the required dependencies:

```bash
pip install torch transformers scikit-learn matplotlib pillow numpy
```

## Usage

### Basic Usage

Extract embeddings from the last token of specific layers:

```bash
python vla-scripts/cache_embeddings.py \
    --model_path openvla/openvla-7b \
    --image_path path/to/image.jpg \
    --prompt "Pick up the red block" \
    --output_dir embeddings/ \
    --layers 0,6,11,23 \
    --positions -1 \
    --auto_format_prompt
```

### Advanced Usage

Extract from all layers and multiple positions with visualization:

```bash
python vla-scripts/cache_embeddings.py \
    --model_path openvla/openvla-7b \
    --image_path path/to/image.jpg \
    --prompt "Pick up the red block" \
    --output_dir embeddings/ \
    --layers all \
    --positions 0,5,10,-1 \
    --format npy \
    --visualize \
    --auto_format_prompt
```

### Test Mode

Test the script functionality without loading large models:

```bash
python vla-scripts/cache_embeddings.py \
    --model_path openvla/openvla-7b \
    --image_path test_image.jpg \
    --prompt "Pick up the red block" \
    --output_dir embeddings_test/ \
    --layers 0,1,2 \
    --positions -1 \
    --test_mode \
    --visualize
```

## Command Line Arguments

- `--model_path`: Path to OpenVLA model (local path or HuggingFace Hub ID)
- `--image_path`: Path to input image
- `--prompt`: Text prompt/instruction for the model
- `--output_dir`: Output directory for cached embeddings
- `--layers`: Layer indices to extract from (comma-separated, ranges with '-', or 'all')
- `--positions`: Token positions to extract (comma-separated, ranges with '-', or 'all')
- `--format`: Output format for embeddings ('pt' or 'npy')
- `--device`: Device to use ('cuda' or 'cpu', auto-detected if not specified)
- `--dtype`: Model dtype ('float32', 'float16', 'bfloat16')
- `--no_flash_attention`: Disable flash attention
- `--visualize`: Create PCA visualization of embeddings
- `--auto_format_prompt`: Automatically format prompt for OpenVLA
- `--test_mode`: Run in test mode with mock model

## Output Structure

The script creates the following files in the output directory:

```
output_dir/
├── layer_0_pos_-1.pt          # Embedding from layer 0, last token
├── layer_6_pos_-1.pt          # Embedding from layer 6, last token
├── layer_11_pos_-1.pt         # Embedding from layer 11, last token
├── layer_23_pos_-1.pt         # Embedding from layer 23, last token
├── metadata.json              # Extraction metadata
└── pca_visualization.png      # PCA plot (if --visualize is used)
```

## Metadata Format

The `metadata.json` file contains:

```json
{
  "model_path": "openvla/openvla-7b",
  "image_path": "path/to/image.jpg",
  "prompt": "In: What action should the robot take to pick up the red block?\nOut:",
  "layer_indices": [0, 6, 11, 23],
  "token_positions": [-1],
  "format": "pt",
  "dtype": "bfloat16",
  "device": "cuda",
  "num_layers": 32,
  "embedding_shapes": {
    "layer_0_pos_-1": [4096],
    "layer_6_pos_-1": [4096],
    "layer_11_pos_-1": [4096],
    "layer_23_pos_-1": [4096]
  }
}
```

## Performance Considerations

- **GPU Recommended**: Large models like OpenVLA-7B require significant computational resources
- **CPU Limitations**: CPU inference is very slow and may timeout for large models
- **Memory Requirements**: Ensure sufficient RAM/VRAM for model loading
- **Timeout Protection**: The script includes timeout protection for CPU inference

## Examples

### Extract from Multiple Positions

```bash
python vla-scripts/cache_embeddings.py \
    --model_path openvla/openvla-7b \
    --image_path robot_scene.jpg \
    --prompt "What action should the robot take to pick up the red block?" \
    --output_dir analysis/embeddings/ \
    --layers 0,8,16,24,31 \
    --positions 0,5,10,-1 \
    --format pt \
    --visualize
```

### Extract from Layer Range

```bash
python vla-scripts/cache_embeddings.py \
    --model_path openvla/openvla-7b \
    --image_path robot_scene.jpg \
    --prompt "Navigate to the kitchen" \
    --output_dir navigation_embeddings/ \
    --layers 10-20 \
    --positions -1 \
    --auto_format_prompt
```

## Troubleshooting

1. **Model Loading Issues**: Ensure you have sufficient memory and the model path is correct
2. **CUDA Errors**: Check GPU availability and CUDA installation
3. **Timeout Errors**: Use test mode or a smaller model for CPU inference
4. **Import Errors**: Install missing dependencies with pip

## Integration

The extracted embeddings can be used for:

- Analyzing model representations across layers
- Studying how visual and textual information is processed
- Comparing embeddings across different prompts or images
- Research into model interpretability and behavior
