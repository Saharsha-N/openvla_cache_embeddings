#!/usr/bin/env python3
"""
cache_embeddings.py

Script to extract and cache embeddings from the residual stream at different token positions 
in an OpenVLA model. Supports hook-based extraction, multiple output formats, and visualization.

Usage:
    python vla-scripts/cache_embeddings.py \
        --model_path openvla/openvla-7b \
        --image_path path/to/image.jpg \
        --prompt "In: Pick up the red block\nOut:" \
        --output_dir embeddings/ \
        --layers 0 6 11 23 \
        --positions -1 \
        --format pt
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.decomposition import PCA
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
from tqdm import tqdm

# Add the project root to the path to import OpenVLA classes
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
except ImportError:
    print("Warning: Could not import OpenVLA classes. This is expected if using models from HF Hub.")
    OpenVLAConfig = None
    OpenVLAForActionPrediction = None
    PrismaticImageProcessor = None
    PrismaticProcessor = None


class EmbeddingExtractor:
    """Class to extract embeddings from OpenVLA models using forward hooks."""
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        use_flash_attention: bool = True,
    ):
        """
        Initialize the embedding extractor.
        
        Args:
            model_path: Path to the OpenVLA model (local or HF Hub)
            device: Device to load the model on (auto-detected if None)
            torch_dtype: Torch dtype for model weights
            use_flash_attention: Whether to use flash attention
        """
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.use_flash_attention = use_flash_attention
        
        # Storage for extracted embeddings
        self.extracted_embeddings: Dict[int, torch.Tensor] = {}
        self.hooks = []

        # Progress tracking
        self.progress_bar = None
        self.layers_processed = 0
        self.total_layers_to_process = 0
        
        # Load model and processor
        self._load_model()
        
    def _load_model(self):
        """Load the OpenVLA model and processor."""
        print(f"[*] Loading OpenVLA model from: {self.model_path}")
        print(f"[*] Using device: {self.device}")
        print(f"[*] Using dtype: {self.torch_dtype}")
        
        # Register OpenVLA classes if loading local checkpoint
        if os.path.isdir(self.model_path) and OpenVLAConfig is not None:
            print("[*] Registering OpenVLA classes for local checkpoint")
            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        
        # Load model
        model_kwargs = {
            "torch_dtype": self.torch_dtype,
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        
        if self.use_flash_attention and self.device == "cuda":
            model_kwargs["attn_implementation"] = "flash_attention_2"
            
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_path, **model_kwargs
        ).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        print(f"[*] Model loaded successfully")
        print(f"[*] Model type: {type(self.model)}")
        
        # Get information about the model structure
        self._analyze_model_structure()
        
    def _analyze_model_structure(self):
        """Analyze the model structure to understand layer organization."""
        print(f"[*] Analyzing model structure...")
        
        # Access the language model
        if hasattr(self.model, 'language_model'):
            self.language_model = self.model.language_model
            print(f"[*] Found language_model: {type(self.language_model)}")
        else:
            raise ValueError("Could not find language_model in the loaded model")
            
        # Find transformer layers
        if hasattr(self.language_model, 'model') and hasattr(self.language_model.model, 'layers'):
            self.transformer_layers = self.language_model.model.layers
            print(f"[*] Found {len(self.transformer_layers)} transformer layers")
        elif hasattr(self.language_model, 'layers'):
            self.transformer_layers = self.language_model.layers
            print(f"[*] Found {len(self.transformer_layers)} transformer layers")
        else:
            raise ValueError("Could not find transformer layers in the language model")
            
        self.num_layers = len(self.transformer_layers)
        print(f"[*] Model has {self.num_layers} transformer layers")

    def _create_hook(self, layer_idx: int):
        """Create a forward hook for a specific layer."""
        def hook_fn(_module, _input, output):
            # Store the output (hidden states) for this layer
            # Output is typically a tuple, we want the hidden states (first element)
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Store on CPU to save GPU memory
            self.extracted_embeddings[layer_idx] = hidden_states.detach().cpu()

            # Update progress
            self.layers_processed += 1
            if self.progress_bar is not None:
                self.progress_bar.update(1)
                self.progress_bar.set_description(f"Processing layer {layer_idx}")

        return hook_fn

    def register_hooks(self, layer_indices: List[int]):
        """Register forward hooks on specified layers."""
        print(f"[*] Registering hooks on layers: {layer_indices}")

        # Clear any existing hooks
        self.clear_hooks()

        # Validate layer indices
        for layer_idx in layer_indices:
            if layer_idx < 0:
                layer_idx = self.num_layers + layer_idx
            if layer_idx < 0 or layer_idx >= self.num_layers:
                raise ValueError(f"Layer index {layer_idx} is out of range [0, {self.num_layers-1}]")

        # Initialize progress tracking
        self.total_layers_to_process = len(layer_indices)
        self.layers_processed = 0

        # Register hooks
        for layer_idx in layer_indices:
            if layer_idx < 0:
                layer_idx = self.num_layers + layer_idx

            hook = self.transformer_layers[layer_idx].register_forward_hook(
                self._create_hook(layer_idx)
            )
            self.hooks.append(hook)

        print(f"[*] Registered {len(self.hooks)} hooks")

    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.extracted_embeddings.clear()

        # Clean up progress tracking
        if self.progress_bar is not None:
            self.progress_bar.close()
            self.progress_bar = None
        self.layers_processed = 0
        self.total_layers_to_process = 0

    def extract_embeddings(
        self,
        image_path: str,
        prompt: str,
        layer_indices: List[int],
        token_positions: List[int],
    ) -> Dict[str, torch.Tensor]:
        """
        Extract embeddings from specified layers and token positions.

        Args:
            image_path: Path to the input image
            prompt: Text prompt for the model
            layer_indices: List of layer indices to extract from
            token_positions: List of token positions to extract

        Returns:
            Dictionary mapping "layer_{layer_idx}_pos_{pos_idx}" to embeddings
        """
        print(f"[*] Extracting embeddings from image: {image_path}")
        print(f"[*] Using prompt: {prompt}")

        # Load and process image
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        print(f"[*] Loaded image with size: {image.size}")

        # Register hooks
        self.register_hooks(layer_indices)

        # Process inputs
        inputs = self.processor(prompt, image).to(self.device, dtype=self.torch_dtype)
        print(f"[*] Processed inputs - input_ids shape: {inputs['input_ids'].shape}")

        # Initialize progress bar
        self.progress_bar = tqdm(
            total=self.total_layers_to_process,
            desc="Extracting embeddings",
            unit="layer",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} layers [{elapsed}<{remaining}, {rate_fmt}]"
        )

        # Run forward pass with hooks active
        print(f"[*] Running forward pass through {self.total_layers_to_process} layers...")
        if self.device == "cpu":
            print(f"[*] Warning: Running on CPU - this may take a very long time for large models")
            print(f"[*] Consider using a GPU or a smaller model.")
            print(f"[*] Progress will be shown as layers are processed...")
        try:
            with torch.no_grad():
                if self.device == "cpu":
                    print(f"[*] Warning: Running on CPU - this may take a very long time for large models")
                    print(f"[*] Consider using a smaller model or GPU if available")

                # Cross-platform timeout implementation
                import threading
                import time

                # Flag to track if forward pass completed
                forward_pass_completed = threading.Event()
                forward_pass_exception = None
                forward_pass_result = None

                def run_forward_pass():
                    nonlocal forward_pass_exception, forward_pass_result
                    try:
                        forward_pass_result = self.model(**inputs, output_hidden_states=False)
                        forward_pass_completed.set()
                    except Exception as e:
                        forward_pass_exception = e
                        forward_pass_completed.set()

                if self.device == "cpu":
                    # Run forward pass in a separate thread with timeout
                    forward_thread = threading.Thread(target=run_forward_pass)
                    forward_thread.daemon = True
                    forward_thread.start()

                    # Monitor progress with longer timeout
                    timeout_seconds = 300  # 5 minutes
                    start_time = time.time()

                    while not forward_pass_completed.is_set():
                        elapsed = time.time() - start_time
                        if elapsed > timeout_seconds:
                            print(f"\n[*] Forward pass timed out after {timeout_seconds} seconds.")
                            print(f"[*] Processed {self.layers_processed}/{self.total_layers_to_process} layers before timeout.")
                            print(f"[*] The model may be too large for CPU inference.")
                            print(f"[*] Consider using a GPU or a smaller model.")
                            raise TimeoutError("Forward pass timed out")

                        # Check every second
                        if forward_pass_completed.wait(timeout=1.0):
                            break

                    if forward_pass_exception:
                        raise forward_pass_exception
                else:
                    # GPU inference - no timeout needed, but still show progress
                    _ = self.model(**inputs, output_hidden_states=False)

        except TimeoutError:
            print(f"[*] Forward pass timed out. The model may be too large for CPU inference.")
            print(f"[*] Consider using a GPU or a smaller model.")
            raise
        except Exception as e:
            print(f"[*] Error during forward pass: {e}")
            raise

        # Complete the progress bar
        if self.progress_bar is not None:
            self.progress_bar.close()

        print(f"[*] Forward pass completed")
        print(f"[*] Extracted embeddings from {len(self.extracted_embeddings)} layers")

        # Extract embeddings at specified token positions
        results = {}
        seq_length = inputs['input_ids'].shape[1]

        for layer_idx, hidden_states in self.extracted_embeddings.items():
            print(f"[*] Processing layer {layer_idx}, hidden_states shape: {hidden_states.shape}")

            for pos_idx in token_positions:
                # Handle negative indices
                if pos_idx < 0:
                    actual_pos = seq_length + pos_idx
                else:
                    actual_pos = pos_idx

                if actual_pos < 0 or actual_pos >= seq_length:
                    print(f"Warning: Position {pos_idx} (actual: {actual_pos}) is out of range for sequence length {seq_length}")
                    continue

                # Extract embedding at this position
                embedding = hidden_states[0, actual_pos, :]  # [batch_size=1, seq_len, hidden_dim]
                key = f"layer_{layer_idx}_pos_{pos_idx}"
                results[key] = embedding

                print(f"[*] Extracted embedding for {key}, shape: {embedding.shape}")

        # Clear hooks to free memory
        self.clear_hooks()

        return results


class MockEmbeddingExtractor:
    """Mock extractor for testing script functionality without loading large models."""

    def __init__(self, model_path: str, **kwargs):
        self.model_path = model_path
        self.device = "cpu"
        self.num_layers = 12  # Mock 12 layers
        # Ignore kwargs for mock mode
        _ = kwargs
        print(f"[*] Mock mode: Simulating model with {self.num_layers} layers")

    def extract_embeddings(
        self,
        image_path: str,
        prompt: str,
        layer_indices: List[int],
        token_positions: List[int],
    ) -> Dict[str, torch.Tensor]:
        """Mock embedding extraction."""
        print(f"[*] Mock extraction from image: {image_path}")
        print(f"[*] Mock prompt: {prompt}")

        # Simulate some processing time
        import time
        time.sleep(2)

        results = {}
        hidden_dim = 768  # Mock hidden dimension

        for layer_idx in layer_indices:
            for pos_idx in token_positions:
                # Create random embeddings
                embedding = torch.randn(hidden_dim)
                key = f"layer_{layer_idx}_pos_{pos_idx}"
                results[key] = embedding
                print(f"[*] Mock extracted embedding for {key}, shape: {embedding.shape}")

        return results


def save_embeddings(
    embeddings: Dict[str, torch.Tensor],
    output_dir: str,
    format: str = "pt",
    metadata: Optional[Dict] = None,
):
    """
    Save extracted embeddings to disk.

    Args:
        embeddings: Dictionary of embeddings to save
        output_dir: Output directory
        format: Output format ('pt' or 'npy')
        metadata: Optional metadata to save alongside embeddings
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[*] Saving {len(embeddings)} embeddings to: {output_path}")

    for key, embedding in embeddings.items():
        if format == "pt":
            file_path = output_path / f"{key}.pt"
            torch.save(embedding, file_path)
        elif format == "npy":
            file_path = output_path / f"{key}.npy"
            np.save(file_path, embedding.numpy())
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"[*] Saved {key} to {file_path}")

    # Save metadata if provided
    if metadata:
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[*] Saved metadata to {metadata_path}")


def visualize_embeddings(
    embeddings: Dict[str, torch.Tensor],
    output_dir: str,
    visualization_type: str = "pca",
):
    """
    Create visualizations of the extracted embeddings.

    Args:
        embeddings: Dictionary of embeddings to visualize
        output_dir: Output directory for visualizations
        visualization_type: Type of visualization ('pca' or 'tsne')
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"[*] Creating {visualization_type} visualizations")

    if visualization_type == "pca":
        # Collect all embeddings for PCA
        all_embeddings = []
        labels = []

        for key, embedding in embeddings.items():
            all_embeddings.append(embedding.numpy())
            labels.append(key)

        if len(all_embeddings) > 1:
            # Stack embeddings and apply PCA
            stacked_embeddings = np.stack(all_embeddings)
            pca = PCA(n_components=2)
            reduced_embeddings = pca.fit_transform(stacked_embeddings)

            # Create plot
            plt.figure(figsize=(10, 8))
            plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1])

            # Add labels
            for i, label in enumerate(labels):
                plt.annotate(label, (reduced_embeddings[i, 0], reduced_embeddings[i, 1]))

            plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
            plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
            plt.title("PCA Visualization of Extracted Embeddings")
            plt.grid(True, alpha=0.3)

            # Save plot
            plot_path = output_path / "pca_visualization.png"
            plt.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(f"[*] Saved PCA visualization to {plot_path}")
        else:
            print("[*] Need at least 2 embeddings for PCA visualization")


def format_prompt(instruction: str, model_path: str) -> str:
    """
    Format the instruction into the correct prompt format for OpenVLA.

    Args:
        instruction: The instruction text
        model_path: Path to the model (to determine version)

    Returns:
        Formatted prompt string
    """
    if "v01" in model_path:
        system_prompt = (
            "A chat between a curious user and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the user's questions."
        )
        return f"{system_prompt} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def parse_layer_indices(layer_str: str, num_layers: int) -> List[int]:
    """Parse layer indices from command line argument."""
    if layer_str.lower() == "all":
        return list(range(num_layers))

    indices = []
    for part in layer_str.split(","):
        part = part.strip()
        if "-" in part and not part.startswith("-"):
            # Range specification like "0-5"
            start, end = map(int, part.split("-"))
            indices.extend(range(start, end + 1))
        else:
            # Single index
            indices.append(int(part))

    return indices


def parse_token_positions(pos_str: str) -> List[int]:
    """Parse token positions from command line argument."""
    if pos_str.lower() == "all":
        return "all"  # Special marker for all positions

    positions = []
    for part in pos_str.split(","):
        part = part.strip()
        if "-" in part and not part.startswith("-"):
            # Range specification like "0-10"
            start, end = map(int, part.split("-"))
            positions.extend(range(start, end + 1))
        else:
            # Single position
            positions.append(int(part))

    return positions


def main():
    """Main function to run the embedding extraction script."""
    parser = argparse.ArgumentParser(
        description="Extract and cache embeddings from OpenVLA models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Extract from last token of specific layers
    python vla-scripts/cache_embeddings.py \\
        --model_path openvla/openvla-7b \\
        --image_path path/to/image.jpg \\
        --prompt "Pick up the red block" \\
        --output_dir embeddings/ \\
        --layers 0,6,11,23 \\
        --positions -1

    # Extract from all layers and positions
    python vla-scripts/cache_embeddings.py \\
        --model_path openvla/openvla-7b \\
        --image_path path/to/image.jpg \\
        --prompt "Pick up the red block" \\
        --output_dir embeddings/ \\
        --layers all \\
        --positions all \\
        --format npy \\
        --visualize
        """
    )

    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to OpenVLA model (local path or HuggingFace Hub ID)"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Path to input image"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Text prompt/instruction for the model"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for cached embeddings"
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Layer indices to extract from (comma-separated, ranges with '-', or 'all'). Default: all"
    )
    parser.add_argument(
        "--positions",
        type=str,
        default="-1",
        help="Token positions to extract (comma-separated, ranges with '-', or 'all'). Default: -1 (last token)"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["pt", "npy"],
        default="pt",
        help="Output format for embeddings. Default: pt"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Default: auto-detect"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        default="bfloat16",
        help="Model dtype. Default: bfloat16"
    )
    parser.add_argument(
        "--no_flash_attention",
        action="store_true",
        help="Disable flash attention"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create PCA visualization of embeddings"
    )
    parser.add_argument(
        "--auto_format_prompt",
        action="store_true",
        help="Automatically format prompt for OpenVLA (adds 'In: What action should the robot take to {prompt}?\\nOut:')"
    )
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Run in test mode with a mock model (for testing script functionality)"
    )

    args = parser.parse_args()

    # Convert dtype string to torch dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map[args.dtype]

    # Format prompt if requested
    if args.auto_format_prompt:
        formatted_prompt = format_prompt(args.prompt, args.model_path)
        print(f"[*] Auto-formatted prompt: {formatted_prompt}")
    else:
        formatted_prompt = args.prompt

    try:
        # Initialize extractor
        print("=" * 60)
        if args.test_mode:
            print("OpenVLA Embedding Extraction (TEST MODE)")
        else:
            print("OpenVLA Embedding Extraction")
        print("=" * 60)

        if args.test_mode:
            extractor = MockEmbeddingExtractor(model_path=args.model_path)
        else:
            extractor = EmbeddingExtractor(
                model_path=args.model_path,
                device=args.device,
                torch_dtype=torch_dtype,
                use_flash_attention=not args.no_flash_attention,
            )

        # Parse layer indices and token positions
        if args.layers.lower() == "all":
            layer_indices = list(range(extractor.num_layers))
        else:
            layer_indices = parse_layer_indices(args.layers, extractor.num_layers)

        if args.positions.lower() == "all":
            # We'll handle this in extract_embeddings by getting all positions
            token_positions = "all"
        else:
            token_positions = parse_token_positions(args.positions)

        print(f"[*] Extracting from layers: {layer_indices}")
        print(f"[*] Extracting from positions: {token_positions}")

        # Handle "all" positions case
        if token_positions == "all":
            # We need to do a preliminary pass to get sequence length
            image = Image.open(args.image_path).convert("RGB")
            inputs = extractor.processor(formatted_prompt, image).to(extractor.device, dtype=torch_dtype)
            seq_length = inputs['input_ids'].shape[1]
            token_positions = list(range(seq_length))
            print(f"[*] Using all {seq_length} token positions")

        # Extract embeddings
        embeddings = extractor.extract_embeddings(
            image_path=args.image_path,
            prompt=formatted_prompt,
            layer_indices=layer_indices,
            token_positions=token_positions,
        )

        # Create metadata
        metadata = {
            "model_path": args.model_path,
            "image_path": args.image_path,
            "prompt": formatted_prompt,
            "layer_indices": layer_indices,
            "token_positions": token_positions if token_positions != "all" else list(range(seq_length)),
            "format": args.format,
            "dtype": args.dtype,
            "device": args.device or extractor.device,
            "num_layers": extractor.num_layers,
            "embedding_shapes": {key: list(emb.shape) for key, emb in embeddings.items()},
        }

        # Save embeddings
        save_embeddings(embeddings, args.output_dir, args.format, metadata)

        # Create visualizations if requested
        if args.visualize:
            visualize_embeddings(embeddings, args.output_dir)

        print("=" * 60)
        print(f"Successfully extracted and saved {len(embeddings)} embeddings!")
        print(f"Output directory: {args.output_dir}")
        print("=" * 60)

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
