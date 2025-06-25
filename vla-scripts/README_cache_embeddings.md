# OpenVLA Embedding Extraction Script (for RLDS/OXE Datasets)

This script extracts and caches embeddings from the residual stream at different token positions in OpenVLA models. It is designed to process episodes from **Open X-Embodiment (OXE) datasets** (or other RLDS-formatted datasets), iterating through episodes and steps. It supports hook-based extraction, multiple output formats, and visualization capabilities.

## Features

- **RLDS/OXE Dataset Processing**: Iterates through episodes and steps of TFDS-compatible robotics datasets.
- **Hook-based extraction**: Registers forward hooks on transformer layers to capture hidden states.
- **Flexible Layer & Stream Selection**: Extract from specific layers, residual stream types ('input', 'output'), and include initial token embeddings (Layer 0).
- **Token Position Control**: Extract embeddings from specific token positions within each processed step.
- **Multiple Output Formats**: Save embeddings as PyTorch (`.pt`) or NumPy (`.npy`) files.
- **Visualization**: Generate PCA plots of extracted embeddings for each processed step.
- **Comprehensive Metadata**: Save extraction parameters, dataset information, and embedding details per step.
- **Mock Mode**: Simulate extraction for quick testing of script logic and dataset iteration flow.
- **Legitimate Mode**: Load real VLA models and process actual datasets for embedding extraction.

## Installation

1.  **Clone the necessary repository if you haven't already (e.g., OpenVLA).**
2.  **Install Python dependencies:**
    This script requires PyTorch, Transformers, TensorFlow (for TFDS), and other common libraries.
    ```bash
    pip install torch torchvision torchaudio
    pip install transformers accelerate sentencepiece protobuf # Common for Transformers
    pip install tensorflow tensorflow_datasets # For RLDS/OXE dataset handling
    pip install scikit-learn matplotlib pillow numpy tqdm
    # For some Hugging Face datasets via TFDS, tfds-nightly might be needed:
    # pip install tfds-nightly
    ```
    Ensure your environment also meets the requirements for the specific OpenVLA model you intend to use (e.g., CUDA version if using GPU).

## Usage

The script now processes datasets episode by episode, and step by step within each episode.

### 1. Legitimate Mode (Extracting Real Embeddings)

This mode loads a real VLA model and processes an actual OXE/RLDS dataset.

**a. Using a registered TFDS dataset name (e.g., from Hugging Face Hub):**
This is common for datasets like `openvla/modified_libero_rlds`.

```bash
python path/to/cache_embeddings.py \
    --model_path openvla/openvla-7b \
    --dataset_name openvla/modified_libero_rlds \
    --dataset_split train \
    --image_key observation.image \
    --instruction_key episode_metadata.natural_language_instruction \
    --output_dir ./embeddings_real_hf/ \
    --layers 0,11,23 \
    --positions -1 \
    --num_episodes 2 \
    --max_steps_per_episode 10 \
    --device cuda
```

**b. Using a GCS path to a TFDS dataset (common in Open X-Embodiment examples):**
Provide the full `gs://...` path.

```bash
python path/to/cache_embeddings.py \
    --model_path openvla/openvla-7b \
    --dataset_name "gs://gresearch/robotics/fractal20220817_data/0.1.0" \
    --dataset_split train \
    --image_key observation.image \
    --instruction_key episode_metadata.language_instruction \
    --output_dir ./embeddings_real_gcs/ \
    --num_episodes 1 \
    --max_steps_per_episode 5
```

**c. Using a local path to a TFDS-formatted dataset:**
If you have a dataset downloaded and structured for TFDS locally.

```bash
python path/to/cache_embeddings.py \
    --model_path openvla/openvla-7b \
    --dataset_name "/path/to/your/local_tfds_dataset_folder" \
    --dataset_split train \
    --image_key observation.image \
    --instruction_key episode_metadata.language_instruction \
    --output_dir ./embeddings_real_local/ \
    --num_episodes 1
```

### 2. Mock Mode (Testing Script Logic)

Use the `--test_mode` flag. This simulates model loading and dataset iteration, useful for quick checks of the script's flow and output structure without heavy computation.

```bash
python path/to/cache_embeddings.py \
    --model_path "mock/model_id" \
    --dataset_name "mock_dataset_for_testing" \
    --dataset_split train \
    --image_key observation.image \
    --instruction_key episode_metadata.natural_language_instruction \
    --output_dir ./embeddings_mock_output/ \
    --layers 0,1,2 \
    --positions -1 \
    --num_episodes 1 \
    --max_steps_per_episode 3 \
    --test_mode \
    --visualize
```

## Command Line Arguments

**Model & Extraction Parameters:**
*   `--model_path` (str, required): Path or HuggingFace Hub ID to the OpenVLA model (or a placeholder in mock mode).
*   `--output_dir` (str, required): Base output directory for cached embeddings.
*   `--layers` (str, default: "all"): Layer indices to extract from (e.g., "0,6,11,23", "0-5", "-1" for last, "all").
*   `--positions` (str, default: "-1"): Token positions to extract from each sequence (e.g., "-1" for last, "0,5,10", "all").
*   `--format` (str, choices: ["pt", "npy"], default: "pt"): Output format for saved embeddings.
*   `--device` (str, default: auto-detect): PyTorch device to use (e.g., "cuda", "cpu").
*   `--dtype` (str, choices: ["float32", "float16", "bfloat16"], default: "bfloat16"): Model data type.
*   `--no_flash_attention` (flag): Disable Flash Attention 2 if the model supports it.
*   `--residual_stream_types` (list of str, choices: ["input", "output"], default: ["output"]): Residual streams to cache (input or output of transformer blocks).
*   `--include_layer0_embedding` (flag): Cache the initial token embeddings (output of the embedding layer).
*   `--visualize` (flag): Generate PCA plots for the embeddings of each processed step.

**Dataset Parameters:**
*   `--dataset_name` (str, required):
    *   Registered TFDS name (e.g., "openvla/modified_libero_rlds").
    *   GCS path to a TFDS dataset (e.g., "gs://gresearch/robotics/dataset_name/version").
    *   Local filesystem path to a TFDS-formatted dataset directory.
*   `--dataset_split` (str, default: "train"): Dataset split to use (e.g., "train", "validation", "train[:10%]").
*   `--tfds_data_dir` (str, default: None): Optional directory for TFDS to download and cache data. Uses TFDS default if None.
*   `--num_episodes` (int, default: 1): Number of episodes to process from the dataset.
*   `--max_steps_per_episode` (int, default: None): Maximum number of steps to process from each episode. Processes all if None.
*   `--image_key` (str, default: "observation.image"): Key to access the image tensor within a dataset step. Use dot notation for nested keys (e.g., "observation.wrist_image").
*   `--instruction_key` (str, default: "episode_metadata.natural_language_instruction"): Key to access the language instruction. Use dot notation. Can be from episode metadata (e.g., "episode_metadata.lang_instruction") or step data (e.g., "language.instruction").

**Mode:**
*   `--test_mode` (flag): Run in MOCK mode for quick testing. Bypasses real model loading and uses a mock dataset iterator.

## Output Structure

The script creates a structured output in the specified `--output_dir`:

```
<output_dir>/
├── <episode_id_1>/
│   ├── <step_0000>/
│   │   ├── layer_L0_embedding_embedding_output.pt  # If --include_layer0_embedding
│   │   ├── layer_0_residual_input.pt             # If "input" in --residual_stream_types
│   │   ├── layer_0_residual_output.pt            # If "output" in --residual_stream_types
│   │   ├── ... (other layers and streams) ...
│   │   ├── step_metadata.json                    # Metadata for this specific step
│   │   └── pca_step_visualization.png            # PCA plot (if --visualize)
│   ├── <step_0001>/
│   │   └── ...
│   └── ...
├── <episode_id_2>/
│   └── ...
└── ...
```

## Step Metadata Format (`step_metadata.json`)

Each step's metadata file contains information like:

```json
{
  "episode_id": "ep_000_some_id_from_dataset",
  "step_index": 0,
  "original_instruction": "pick up the red block from the table",
  "image_key_used": "observation.image",
  "instruction_key_used": "episode_metadata.natural_language_instruction",
  "layers_extracted": [0, 11, 23], // or "all"
  "positions_extracted": [-1],    // or "all" or list of positions
  "embedding_shapes": {
    "L_L0_embedding_embedding_output": [1, 768], // Example: [num_tokens_extracted, hidden_dim]
    "L_0_residual_output": [1, 768],
    "L_11_residual_output": [1, 768],
    "L_23_residual_output": [1, 768]
  },
  "action_from_dataset": { // Example, structure depends on dataset
    "world_vector": [0.1, -0.05, 0.02],
    "rotation_delta": [0.0, 0.0, 0.1],
    "gripper_closedness_action": [1.0]
  }
}
```
*(Note: `prompt` or `formatted_prompt` might be long and is not included by default in the example above to keep it concise, but could be added).*

## Performance Considerations

- **GPU Recommended**: Processing large VLA models and datasets is computationally intensive.
- **CPU Inference**: Extremely slow for large models; primarily for very small tests or debugging.
- **Memory**: Ensure sufficient RAM (for data processing) and VRAM (for model and inference).
- **Dataset Size**: OXE datasets can be very large. Ensure enough disk space for `tfds_data_dir` if downloading.

## Troubleshooting

1.  **Model Loading Issues**:
    *   Verify `--model_path` is correct.
    *   Ensure sufficient VRAM/RAM.
    *   Check for `trust_remote_code=True` requirements if loading custom models from Hub.
2.  **Dataset Loading Errors (`tfds`):**
    *   Verify `--dataset_name`. For registered names, check spelling. For GCS/local paths, ensure the path is exact and accessible.
    *   For GCS paths, ensure `gsutil` is configured and you have access permissions.
    *   Some Hugging Face datasets loaded via `tfds` might require `tfds-nightly` or Hugging Face CLI login (`huggingface-cli login`).
    *   Check `--dataset_split` exists for the dataset.
    *   Ensure `--tfds_data_dir` is writable and has enough space if datasets are being downloaded.
3.  **Key Errors (Image/Instruction):**
    *   Double-check `--image_key` and `--instruction_key` by inspecting a sample from your target dataset (see helper script example in `cache_embeddings.py` comments or use TFDS CLI).
4.  **CUDA Errors**: Confirm PyTorch can see your GPU and CUDA/cuDNN setup is correct.
5.  **Import Errors**: Install all dependencies listed in the "Installation" section.

## Integration & Use Cases

The extracted embeddings can be valuable for:

- Analyzing how representations evolve across model layers and residual streams.
- Studying the fusion of visual and textual information at different processing stages.
- Comparing model behavior (via embeddings) across different instructions, visual inputs, or dataset episodes.
- Research into model interpretability, robustness, and internal decision-making processes.
- Probing for specific concepts or features encoded in the hidden states.