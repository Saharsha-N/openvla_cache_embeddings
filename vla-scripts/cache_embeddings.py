#!/usr/bin/env python3
"""
cache_embeddings.py

Script to extract and cache embeddings from the residual stream at different token positions
in an OpenVLA model. Supports hook-based extraction, multiple output formats, and visualization.
Processes episodes from Open X-Embodiment (OXE) datasets.

-----------------------------------------------------------------------------------------------
HOW TO RUN:

1. LEGITIMATE MODE (Extracting real embeddings from a model):
   Ensure you have a VLA model (local path or HF Hub ID) and an OXE dataset.
   Example:
    python vla-scripts/cache_embeddings.py \\
        --model_path openvla/openvla-7b \\
        --dataset_name openvla/modified_libero_rlds \\
        --image_key observation.image \\
        --instruction_key episode_metadata.natural_language_instruction \\
        --output_dir embeddings_real/ \\
        --layers 0,11,23 \\
        --positions -1 \\
        --num_episodes 2

2. MOCK MODE (Testing script logic without loading models or heavy processing):
   Use the --test_mode flag. This will use a mock extractor that simulates the process.
   Example:
    python vla-scripts/cache_embeddings.py \\
        --model_path "mock/model" \\
        --dataset_name "mock/dataset" \\
        --image_key observation.image \\
        --instruction_key episode_metadata.natural_language_instruction \\
        --output_dir embeddings_mock/ \\
        --layers 0,1,2 \\
        --positions -1 \\
        --num_episodes 1 \\
        --test_mode
   (In mock mode, --dataset_name, --image_key, --instruction_key are still used to simulate
    dataset iteration logic, but actual data loading from these sources is bypassed by TFDS's
    mocking capabilities if available or the script's internal mock data generation for steps.)
-----------------------------------------------------------------------------------------------
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

# Import tensorflow_datasets
import tensorflow_datasets as tfds
import tensorflow as tf # Required by tfds for some operations

# Disable GPU for TF if PyTorch is using it, to avoid conflicts
tf.config.set_visible_devices([], 'GPU')


# Add the project root to the path to import OpenVLA classes
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
    from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
    from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
except ImportError:
    print("Warning: Could not import OpenVLA classes. This is expected if using models from HF Hub or in mock mode.")
    OpenVLAConfig = None
    OpenVLAForActionPrediction = None
    PrismaticImageProcessor = None
    PrismaticProcessor = None


class EmbeddingExtractor:
    """
    LEGITIMATE Embedding Extractor:
    Class to extract real embeddings from OpenVLA models using forward hooks.
    """
    def __init__(
        self,
        model_path: str,
        residual_stream_types: Optional[List[str]] = None,
        include_layer0_embedding: bool = False,
        device: Optional[str] = None,
        torch_dtype: torch.dtype = torch.bfloat16,
        use_flash_attention: bool = True,
    ):
        print("[LEGITIMATE MODE] Initializing Real EmbeddingExtractor...")
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype
        self.use_flash_attention = use_flash_attention
        self.residual_stream_types = residual_stream_types or ["output"]
        self.include_layer0_embedding = include_layer0_embedding

        self.extracted_embeddings: Dict[Tuple[Union[int, str], str], torch.Tensor] = {}
        self.hooks = []
        self.progress_bar = None
        self.layers_processed = 0
        self.total_layers_to_process = 0
        self._load_model()

    def _load_model(self):
        print(f"[LEGITIMATE MODE] Loading OpenVLA model from: {self.model_path}")
        print(f"[LEGITIMATE MODE] Using device: {self.device}")
        print(f"[LEGITIMATE MODE] Using dtype: {self.torch_dtype}")
        if os.path.isdir(self.model_path) and OpenVLAConfig is not None:
            print("[LEGITIMATE MODE] Registering OpenVLA classes for local checkpoint")
            AutoConfig.register("openvla", OpenVLAConfig)
            AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
            AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
            AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
        
        self.processor = AutoProcessor.from_pretrained(self.model_path, trust_remote_code=True)
        
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
        self.model.eval()
        print(f"[LEGITIMATE MODE] Model loaded successfully. Type: {type(self.model)}")
        self._analyze_model_structure()

    def _analyze_model_structure(self):
        print(f"[LEGITIMATE MODE] Analyzing model structure...")
        if hasattr(self.model, 'language_model'):
            self.language_model = self.model.language_model
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
             self.language_model = self.model
        else:
            raise ValueError("[LEGITIMATE MODE] Could not find language_model attribute or compatible structure.")

        if hasattr(self.language_model, 'model') and hasattr(self.language_model.model, 'layers'):
            self.transformer_layers = self.language_model.model.layers
        elif hasattr(self.language_model, 'layers'):
            self.transformer_layers = self.language_model.layers
        else:
            raise ValueError("[LEGITIMATE MODE] Could not find transformer layers.")
        self.num_layers = len(self.transformer_layers)
        print(f"[LEGITIMATE MODE] Model has {self.num_layers} transformer layers.")

        if hasattr(self.language_model, 'model') and hasattr(self.language_model.model, 'embed_tokens'):
            self.embedding_layer = self.language_model.model.embed_tokens
        elif hasattr(self.language_model, 'embed_tokens'):
            self.embedding_layer = self.language_model.embed_tokens
        else:
            self.embedding_layer = None
            if self.include_layer0_embedding:
                print("[LEGITIMATE MODE] Warning: Could not find embedding layer for Layer 0.")

    def _create_input_hook(self, layer_identifier: Union[int, str], stream_name: str):
        def hook_fn(_module, _input):
            tensor_to_cache = _input[0]
            self.extracted_embeddings[(layer_identifier, stream_name)] = tensor_to_cache.detach().cpu()
        return hook_fn

    def _create_output_hook(self, layer_identifier: Union[int, str], stream_name: str):
        def hook_fn(_module, _input, output):
            tensor_to_cache = output[0] if isinstance(output, tuple) else output
            self.extracted_embeddings[(layer_identifier, stream_name)] = tensor_to_cache.detach().cpu()
            self.layers_processed += 1
            if self.progress_bar: self.progress_bar.update(1)
        return hook_fn

    def _get_layers_to_hook(self, layer_indices: Optional[List[int]] = None) -> List[int]:
        if layer_indices is None: return list(range(self.num_layers))
        valid_hooks = []
        for idx in layer_indices:
            actual_idx = self.num_layers + idx if idx < 0 else idx
            if 0 <= actual_idx < self.num_layers: valid_hooks.append(actual_idx)
            else: print(f"[LEGITIMATE MODE] Warning: Layer index {idx} out of range. Skipping.")
        return valid_hooks

    def register_hooks(self, layer_indices: Optional[List[int]] = None):
        self.clear_hooks()
        layers_to_hook = self._get_layers_to_hook(layer_indices)
        print(f"[LEGITIMATE MODE] Registering hooks on layers: {layers_to_hook or 'all'}")
        print(f"[LEGITIMATE MODE] Caching streams: {self.residual_stream_types}, Layer0: {self.include_layer0_embedding}")

        self.total_layers_to_process = 0
        if "output" in self.residual_stream_types: self.total_layers_to_process += len(layers_to_hook)
        if self.include_layer0_embedding and self.embedding_layer and ("output" in self.residual_stream_types or not self.residual_stream_types):
            self.total_layers_to_process += 1
        self.layers_processed = 0

        if self.include_layer0_embedding and self.embedding_layer:
            self.hooks.append(self.embedding_layer.register_forward_hook(self._create_output_hook("L0_embedding", "embedding_output")))

        for layer_idx in layers_to_hook:
            module = self.transformer_layers[layer_idx]
            if "input" in self.residual_stream_types: self.hooks.append(module.register_forward_pre_hook(self._create_input_hook(layer_idx, "residual_input")))
            if "output" in self.residual_stream_types: self.hooks.append(module.register_forward_hook(self._create_output_hook(layer_idx, "residual_output")))
        print(f"[LEGITIMATE MODE] Registered {len(self.hooks)} hooks. Expecting {self.total_layers_to_process} output hook calls.")

    def clear_hooks(self):
        for hook in self.hooks: hook.remove()
        self.hooks.clear()
        self.extracted_embeddings.clear()
        if self.progress_bar: self.progress_bar.close(); self.progress_bar = None
        self.layers_processed = 0

    def extract_embeddings(self, image_pil: Image.Image, prompt: str, token_positions: Union[List[int], str] = "all") -> Dict[Tuple[Union[int, str], str], torch.Tensor]:
        self.extracted_embeddings.clear()
        self.layers_processed = 0
        inputs = self.processor(prompt, image_pil.convert("RGB"), return_tensors="pt").to(self.device, dtype=self.torch_dtype)

        self.progress_bar = tqdm(total=self.total_layers_to_process, desc="Extracting step (real)", unit="layer", leave=False, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        try:
            with torch.no_grad(): _ = self.model(**inputs, output_hidden_states=False)
        finally:
            if self.progress_bar:
                if self.layers_processed < self.total_layers_to_process: self.progress_bar.update(self.total_layers_to_process - self.layers_processed)
                self.progress_bar.close(); self.progress_bar = None

        final_embeddings = {}
        for key, full_emb in self.extracted_embeddings.items():
            seq_emb = full_emb.squeeze(0) if full_emb.ndim == 3 and full_emb.shape[0] == 1 else full_emb
            if token_positions == "all": final_embeddings[key] = seq_emb
            else:
                indices = [seq_emb.shape[0] + p if p < 0 else p for p in token_positions if 0 <= (seq_emb.shape[0] + p if p < 0 else p) < seq_emb.shape[0]]
                if indices: final_embeddings[key] = seq_emb[indices]
        return final_embeddings.copy()


class MockEmbeddingExtractor:
    """
    MOCK Embedding Extractor:
    Simulates the embedding extraction process for testing script logic quickly.
    Does not load any real models or perform actual computations.
    """
    def __init__(self, model_path: str, residual_stream_types: Optional[List[str]] = None, include_layer0_embedding: bool = False, **kwargs):
        print(f"[MOCK MODE] Initializing MockEmbeddingExtractor for model_path: '{model_path}'")
        self.model_path = model_path
        self.device = "cpu"  # Mock device
        self.num_layers = 12  # Mock number of layers, adjust if needed for parse_layer_indices
        self.processor = "mock_processor_object" # Placeholder for processor attribute
        self.residual_stream_types = residual_stream_types or ["output"]
        self.include_layer0_embedding = include_layer0_embedding
        
        self.total_layers_to_process = 0 # Calculated in register_hooks
        self.progress_bar = None # Mock progress bar handling
        self.layers_processed = 0 # Mock progress bar handling
        _ = kwargs # Absorb other arguments

    def register_hooks(self, layer_indices: Optional[List[int]] = None):
        """Simulates registering hooks."""
        # Determine actual layers to "hook" based on input
        if layer_indices is None:
            layers_to_hook_indices = list(range(self.num_layers))
        else:
            layers_to_hook_indices = []
            for idx in layer_indices:
                actual_idx = self.num_layers + idx if idx < 0 else idx
                if 0 <= actual_idx < self.num_layers:
                    layers_to_hook_indices.append(actual_idx)
        
        print(f"[MOCK MODE] Simulating registering hooks for layers: {layers_to_hook_indices or 'all'}")
        
        # Calculate total_layers_to_process for mock progress bar
        self.total_layers_to_process = 0
        if "output" in self.residual_stream_types:
            self.total_layers_to_process += len(layers_to_hook_indices)
        if self.include_layer0_embedding and ("output" in self.residual_stream_types or not self.residual_stream_types):
            self.total_layers_to_process += 1
        
        print(f"[MOCK MODE] Expecting {self.total_layers_to_process} mock output hook calls.")


    def clear_hooks(self):
        """Simulates clearing hooks."""
        print("[MOCK MODE] Simulating clearing hooks.")
        if self.progress_bar: self.progress_bar.close(); self.progress_bar = None


    def extract_embeddings(
        self,
        image_pil: Image.Image,
        prompt: str,
        token_positions: Union[List[int], str] = "all",
    ) -> Dict[Tuple[Union[int, str], str], torch.Tensor]:
        """Simulates extracting embeddings."""
        print(f"[MOCK MODE] Simulating extraction for image size: {image_pil.size}, prompt: '{prompt[:30]}...'")
        
        # Simulate progress bar
        self.layers_processed = 0
        self.progress_bar = tqdm(total=self.total_layers_to_process, desc="Extracting step (mock)", unit="layer", leave=False, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        
        # Simulate some work and progress updates
        for _ in range(self.total_layers_to_process):
            time.sleep(0.001) # Tiny sleep
            self.layers_processed +=1
            self.progress_bar.update(1)
        if self.progress_bar: self.progress_bar.close(); self.progress_bar = None


        mock_results = {}
        hidden_dim = 768  # Standard mock hidden dimension
        mock_seq_len = 25  # Mock sequence length

        # Generate mock data for the layers/streams it expects based on total_layers_to_process
        # This is a simplification; real hooks are more specific.
        # Let's assume 'output' stream for all processed layers.
        
        num_mock_transformer_layers = self.total_layers_to_process
        if self.include_layer0_embedding and ("output" in self.residual_stream_types or not self.residual_stream_types):
            mock_results[("L0_embedding", "embedding_output")] = torch.randn(mock_seq_len, hidden_dim)
            num_mock_transformer_layers -=1


        for i in range(max(0, num_mock_transformer_layers)): # Iterate for remaining expected outputs
            # Use a dummy layer index; doesn't have to match precisely for mock,
            # just needs to generate the right number of entries.
            # A more robust mock would use the parsed layer_indices.
            layer_id = i 
            if "output" in self.residual_stream_types:
                mock_results[(layer_id, "residual_output")] = torch.randn(mock_seq_len, hidden_dim)
            if "input" in self.residual_stream_types: # Though input hooks don't update progress in real one
                 mock_results[(layer_id, "residual_input")] = torch.randn(mock_seq_len, hidden_dim)


        # Simulate token selection
        final_embeddings = {}
        for key, full_seq_embedding in mock_results.items():
            if token_positions == "all":
                final_embeddings[key] = full_seq_embedding
            else:
                # Mock selection logic (e.g., specific indices or last token)
                selected_indices = []
                for pos in token_positions:
                    actual_pos = mock_seq_len + pos if pos < 0 else pos
                    if 0 <= actual_pos < mock_seq_len:
                        selected_indices.append(actual_pos)
                if selected_indices:
                    final_embeddings[key] = full_seq_embedding[selected_indices]
                # else:
                #     print(f"[MOCK MODE] No valid tokens for {key} with positions {token_positions}")
        
        return final_embeddings.copy()


def save_embeddings(
    embeddings: Dict[Tuple[Union[int, str], str], torch.Tensor],
    output_dir: Path,
    format: str = "pt",
    metadata: Optional[Dict] = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    # print(f"[*] Saving {len(embeddings)} embedding tensors to: {output_dir}") # Can be verbose in loops

    for key_tuple, embedding in embeddings.items():
        layer_id, stream_name = key_tuple
        safe_layer_id = str(layer_id).replace("/", "_") 
        filename_base = f"layer_{safe_layer_id}_{stream_name}"

        if format == "pt":
            file_path = output_dir / f"{filename_base}.pt"
            torch.save(embedding, file_path)
        elif format == "npy":
            file_path = output_dir / f"{filename_base}.npy"
            np.save(file_path, embedding.cpu().numpy())
        else:
            raise ValueError(f"Unsupported format: {format}")

    if metadata:
        metadata_path = output_dir / "step_metadata.json"
        with open(metadata_path, "w") as f:
            if "embedding_shapes" in metadata:
                metadata["embedding_shapes"] = {
                    f"layer_{str(k_tuple[0]).replace('/', '_')}_{k_tuple[1]}": list(v_shape)
                    for k_tuple, v_shape in metadata["embedding_shapes"].items()
                }
            json.dump(metadata, f, indent=2)


def visualize_embeddings(
    embeddings: Dict[Tuple[Union[int, str], str], torch.Tensor],
    output_dir: Path,
    visualization_type: str = "pca",
):
    output_dir.mkdir(parents=True, exist_ok=True)

    if visualization_type == "pca":
        all_embeddings_list, labels = [], []
        for key_tuple, emb_tensor in embeddings.items():
            layer_id_str = str(key_tuple[0]).replace('/', '_')
            label = f"L{layer_id_str}_{key_tuple[1]}"
            # Average if multiple tokens selected for a layer/stream for PCA point
            emb_to_plot = emb_tensor.mean(dim=0).cpu().numpy() if emb_tensor.ndim == 2 and emb_tensor.shape[0] > 0 else emb_tensor.cpu().numpy()
            if emb_to_plot.ndim == 1: # Ensure it's a 1D vector for PCA
                all_embeddings_list.append(emb_to_plot)
                labels.append(label)
        
        if len(all_embeddings_list) >= 2:
            stacked_embeddings = np.array(all_embeddings_list)
            if stacked_embeddings.ndim == 1: stacked_embeddings = np.vstack(all_embeddings_list) # if all were 1D
            if stacked_embeddings.shape[0] < 2: return

            try:
                n_comp = min(2, stacked_embeddings.shape[1], stacked_embeddings.shape[0])
                if n_comp < 2: return
                
                pca = PCA(n_components=n_comp)
                reduced = pca.fit_transform(stacked_embeddings)
                plt.figure(figsize=(10, 8))
                plt.scatter(reduced[:, 0], reduced[:, 1])
                for i, lbl in enumerate(labels): plt.annotate(lbl, (reduced[i, 0], reduced[i, 1]))
                
                pc1_var = pca.explained_variance_ratio_[0] if len(pca.explained_variance_ratio_) > 0 else 0
                pc2_var = pca.explained_variance_ratio_[1] if len(pca.explained_variance_ratio_) > 1 else 0
                plt.xlabel(f"PC1 ({pc1_var:.2%})"); plt.ylabel(f"PC2 ({pc2_var:.2%})")
                plt.title(f"PCA (Step in {output_dir.name})"); plt.grid(True, alpha=0.3)
                plt.savefig(output_dir / "pca_step_visualization.png", dpi=300, bbox_inches="tight")
                plt.close()
            except ValueError as e:
                print(f"[*] PCA Error for {output_dir.name}: {e}. Data shape: {stacked_embeddings.shape}. Skipping.")


def format_prompt_from_instruction(instruction: str, model_path_or_name: str) -> str:
    if instruction.strip().startswith("In:") and "\nOut:" in instruction: return instruction
    return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


def parse_layer_indices(layer_str: str, num_layers: int) -> List[int]:
    if layer_str.lower() == "all": return list(range(num_layers))
    indices = set()
    for part in layer_str.split(","):
        part = part.strip()
        try:
            if "-" in part and not part.startswith("-"):
                start, end = map(int, part.split("-"))
                indices.update(range(start, end + 1))
            else: indices.add(int(part))
        except ValueError: print(f"Warning: Invalid layer spec '{part}'. Skipping.")
    return sorted(list(indices))

def parse_token_positions(pos_str: str) -> Union[List[int], str]:
    if pos_str.lower() == "all": return "all"
    positions = set()
    for part in pos_str.split(","):
        part = part.strip()
        try:
            if "-" in part and not part.startswith("-"):
                start, end = map(int, part.split("-"))
                positions.update(range(start, end + 1))
            else: positions.add(int(part))
        except ValueError: print(f"Warning: Invalid token position spec '{part}'. Skipping.")
    return sorted(list(positions))


def main():
    parser = argparse.ArgumentParser(
        description="Extract and cache embeddings from OpenVLA models using OXE datasets.\n" + \
                    "Run with --help for detailed options.\n" + \
                    "Use --test_mode for quick script logic testing without real model loading.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog= # Copied from top for easy access with --help
"""
-----------------------------------------------------------------------------------------------
HOW TO RUN EXAMPLES:

1. LEGITIMATE MODE (Extracting real embeddings):
    python cache_embeddings.py \\
        --model_path openvla/openvla-7b \\
        --dataset_name openvla/modified_libero_rlds \\
        --image_key observation.image \\
        --instruction_key episode_metadata.natural_language_instruction \\
        --output_dir embeddings_real/ \\
        --layers 0,11,23 \\
        --positions -1 \\
        --num_episodes 2

2. MOCK MODE (Testing script logic):
    python cache_embeddings.py \\
        --model_path "mock/model" \\
        --dataset_name "mock/rlds_dataset_name" \\
        --image_key observation.image \\
        --instruction_key episode_metadata.natural_language_instruction \\
        --output_dir embeddings_mock/ \\
        --layers 0,1,2 \\
        --num_episodes 1 \\
        --test_mode
-----------------------------------------------------------------------------------------------
"""
    )
    # Model and Extraction parameters
    parser.add_argument("--model_path", type=str, required=True, help="Path or HF ID to OpenVLA model (or mock path in test_mode)")
    parser.add_argument("--output_dir", type=str, required=True, help="Base output directory for cached embeddings")
    parser.add_argument("--layers", type=str, default="all", help="Layer indices (e.g., '0,6,11,23', '0-5', 'all')")
    parser.add_argument("--positions", type=str, default="-1", help="Token positions (e.g., '-1', '0,1,-1', 'all')")
    parser.add_argument("--format", type=str, choices=["pt", "npy"], default="pt", help="Output format for embeddings")
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda', 'cpu'). Auto-detects if None.")
    parser.add_argument("--dtype", type=str, choices=["float32", "float16", "bfloat16"], default="bfloat16", help="Model dtype")
    parser.add_argument("--no_flash_attention", action="store_true", help="Disable flash attention")
    parser.add_argument("--residual_stream_types", type=str, nargs="+", choices=["input", "output"], default=["output"], help="Residual stream types")
    parser.add_argument("--include_layer0_embedding", action="store_true", help="Include initial token embeddings (Layer 0)")
    parser.add_argument("--visualize", action="store_true", help="Create PCA visualization for each step")

    # Dataset parameters
    parser.add_argument("--dataset_name", type=str, required=True, help="TFDS dataset name (e.g., 'openvla/modified_libero_rlds') or GCS path (gs://...)")
    parser.add_argument("--dataset_split", type=str, default="train", help="Dataset split (e.g., 'train', 'validation', 'train[:1%%]')")
    parser.add_argument("--tfds_data_dir", type=str, default=None, help="TFDS download/storage directory (optional)")
    parser.add_argument("--num_episodes", type=int, default=1, help="Number of episodes to process")
    parser.add_argument("--max_steps_per_episode", type=int, default=None, help="Max steps per episode (optional)")
    parser.add_argument("--image_key", type=str, default="observation.image", help="Key for image in step data (dot notation for nested)")
    parser.add_argument("--instruction_key", type=str, default="episode_metadata.natural_language_instruction", help="Key for language instruction (dot notation)")
    
    # Test Mode Flag
    parser.add_argument("--test_mode", action="store_true", help="Run in MOCK mode for quick testing without loading real models.")

    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    torch_dtype = dtype_map[args.dtype]

    print("=" * 80)
    if args.test_mode:
        print("OpenVLA Embedding Extraction (MOCK MODE)")
    else:
        print("OpenVLA Embedding Extraction (LEGITIMATE MODE)")
    print("=" * 80)

    try:
        # Conditional instantiation of the extractor based on --test_mode
        if args.test_mode:
            # In test_mode, some real model args are passed but mostly ignored by MockExtractor
            extractor = MockEmbeddingExtractor(
                model_path=args.model_path,
                residual_stream_types=args.residual_stream_types,
                include_layer0_embedding=args.include_layer0_embedding,
                # Other args like device, dtype, flash_attention are not used by mock
            )
        else:
            # Real mode
            extractor = EmbeddingExtractor(
                model_path=args.model_path,
                residual_stream_types=args.residual_stream_types,
                include_layer0_embedding=args.include_layer0_embedding,
                device=args.device,
                torch_dtype=torch_dtype,
                use_flash_attention=not args.no_flash_attention,
            )

        # Common setup for both modes
        if args.layers.lower() == "all":
            layer_indices_to_hook = None 
        else:
            layer_indices_to_hook = parse_layer_indices(args.layers, extractor.num_layers)
        
        token_positions_to_extract = parse_token_positions(args.positions)
        extractor.register_hooks(layer_indices=layer_indices_to_hook) # Mock or Real register_hooks called
        # print(f"[*] Hooks registered for layers: {layer_indices_to_hook if layer_indices_to_hook is not None else 'all available'}")
        # print(f"[*] Will extract from token positions: {token_positions_to_extract}")

        print(f"[*] Loading dataset: {args.dataset_name}, split: {args.dataset_split}")
        # Dataset loading logic (mock or real)
        # In test_mode, TFDS can use mock data if the dataset_name is recognized by tfds.testing.mock_data
        # or if you implement a specific mock data generator. For simplicity, we'll let it try.
        # If dataset loading fails in test_mode, it's often okay as the core logic is what's tested.
        # However, for robust mock testing of dataset iteration, a tfds.testing.mock_data setup is best.
        # For this script, the mock extractor bypasses needing real data from the dataset iterator.
        ds = None
        if args.test_mode:
            print("[MOCK MODE] Skipping real dataset loading. Will use mock iteration.")
            # Create a mock iterable for episodes and steps
            def mock_dataset_generator(num_episodes, max_steps):
                for i in range(num_episodes):
                    mock_episode_id = f"mock_ep_{i:03d}"
                    mock_steps_data = []
                    num_mock_steps = max_steps if max_steps is not None else np.random.randint(5,15)
                    for j in range(num_mock_steps):
                        mock_step = {
                            # Populate with minimal data mock extractor might look for, if any.
                            # For our current mock, it doesn't really use the step_data content.
                            'observation': {'image': tf.zeros((64,64,3), dtype=tf.uint8)}, # Mock image tensor
                            'action': {'mock_action': tf.zeros((7,), dtype=tf.float32)}
                        }
                        # If instruction_key is not episode_metadata, mock it here
                        if not args.instruction_key.startswith("episode_metadata"):
                             nested_keys = args.instruction_key.split('.')
                             current_dict = mock_step
                             for k_idx, k_part in enumerate(nested_keys[:-1]):
                                 current_dict.setdefault(k_part, {})
                                 current_dict = current_dict[k_part]
                             current_dict[nested_keys[-1]] = tf.constant(f"Mock step instruction {j}")


                        mock_steps_data.append(mock_step)
                    
                    mock_episode = {'episode_id': tf.constant(mock_episode_id), 'steps': tf.data.Dataset.from_generator(lambda: (s for s in mock_steps_data), output_signature={k: tf.TensorSpec.from_tensor(v) if isinstance(v,tf.Tensor) else {sk: tf.TensorSpec.from_tensor(sv) for sk,sv in v.items()} for k,v in mock_steps_data[0].items()})}
                    
                    # Mock episode_metadata for instruction key
                    if args.instruction_key.startswith("episode_metadata"):
                        metadata_keys = args.instruction_key.split('.') # e.g. episode_metadata.natural_language_instruction
                        current_meta_dict = mock_episode
                        # current_meta_dict.setdefault(metadata_keys[0], {}) # episode_metadata
                        # current_meta_dict = current_meta_dict[metadata_keys[0]]
                        # for k_idx, k_part in enumerate(metadata_keys[1:-1]):
                        #      current_meta_dict.setdefault(k_part, {})
                        #      current_meta_dict = current_meta_dict[k_part]
                        # current_meta_dict[metadata_keys[-1]] = tf.constant(f"Mock episode instruction for {mock_episode_id}")
                        # Simplified mocking for episode_metadata:
                        mock_episode['episode_metadata'] = {'natural_language_instruction': tf.constant(f"Mock instruction for {mock_episode_id}")}


                    yield mock_episode
            ds = tf.data.Dataset.from_generator(lambda: mock_dataset_generator(args.num_episodes, args.max_steps_per_episode), 
                                                output_signature={'episode_id': tf.TensorSpec(shape=(), dtype=tf.string), 
                                                                  'steps': tf.data.DatasetSpec(tfds.features.FeaturesDict({'observation': {'image': tfds.features.Tensor(shape=(64,64,3), dtype=tf.uint8)}, 'action':{'mock_action': tfds.features.Tensor(shape=(7,), dtype=tf.float32)}}).get_tensor_spec()),
                                                                  'episode_metadata': tfds.features.FeaturesDict({'natural_language_instruction': tfds.features.Tensor(shape=(), dtype=tf.string)}).get_tensor_spec()
                                                                  }
                                                )
        else: # Legitimate mode - load real dataset
            try:
                if "/" in args.dataset_name and not args.dataset_name.startswith("gs://"):
                     dataset_builder = tfds.builder(args.dataset_name, data_dir=args.tfds_data_dir)
                     dataset_builder.download_and_prepare()
                     ds_full = dataset_builder.as_dataset(split=f"{args.dataset_split}")
                else:
                     dataset_builder = tfds.builder_from_directory(builder_dir=args.dataset_name)
                     ds_full = dataset_builder.as_dataset(split=f"{args.dataset_split}")
                ds = ds_full.take(args.num_episodes)
            except Exception as e:
                print(f"Error loading dataset '{args.dataset_name}': {e}")
                sys.exit(1)

        total_steps_processed = 0
        # Determine total for episode pbar (can be tricky if ds.take is used on infinite dataset)
        # For mock, num_episodes is known. For real, if ds.take was used, it's also num_episodes.
        episode_pbar_total = args.num_episodes

        episode_pbar = tqdm(ds, total=episode_pbar_total, desc="Processing Episodes")

        for i, episode_data in enumerate(episode_pbar):
            try:
                episode_id_tensor = episode_data.get('episode_id', tf.constant(f"ep_{i:04d}", dtype=tf.string))
                episode_id = episode_id_tensor.numpy().decode('utf-8') if isinstance(episode_id_tensor, tf.Tensor) and episode_id_tensor.dtype == tf.string else str(episode_id_tensor)
                
                episode_pbar.set_description(f"Episode {episode_id}")
                episode_output_dir = Path(args.output_dir) / episode_id
                episode_output_dir.mkdir(parents=True, exist_ok=True)

                current_instruction = "No instruction"
                instr_keys = args.instruction_key.split('.')
                source = episode_data
                valid_path = True
                for key_part in instr_keys:
                    if isinstance(source, dict) and key_part in source: source = source[key_part]
                    elif hasattr(source, key_part): source = getattr(source, key_part)
                    else: valid_path = False; break
                if valid_path and source is not None:
                    if isinstance(source, tf.Tensor) and source.dtype == tf.string:
                        current_instruction = source.numpy().decode('utf-8') if source.shape.rank == 0 else (source.numpy()[0].decode('utf-8') if source.shape.rank > 0 and source.shape[0]>0 else "Empty instruction array")
                    elif isinstance(source, (str, bytes)): current_instruction = source.decode('utf-8') if isinstance(source, bytes) else source
                
                steps_iterable = episode_data['steps']
                num_steps_tf = tf.data.experimental.cardinality(steps_iterable)
                num_steps = num_steps_tf.numpy() if num_steps_tf >= 0 else None
                step_pbar_total = args.max_steps_per_episode if args.max_steps_per_episode else num_steps
                step_pbar = tqdm(steps_iterable, desc=f"Steps in Ep {episode_id}", total=step_pbar_total, leave=False)

                for step_idx, step_data in enumerate(step_pbar):
                    if args.max_steps_per_episode and step_idx >= args.max_steps_per_episode: break
                    
                    step_id_str = f"step_{step_idx:04d}"
                    step_output_dir = episode_output_dir / step_id_str
                    # step_output_dir.mkdir(parents=True, exist_ok=True) # Parent is already created

                    img_val = step_data
                    valid_img = True
                    for key_part in args.image_key.split('.'):
                        if isinstance(img_val, dict) and key_part in img_val: img_val = img_val[key_part]
                        elif hasattr(img_val, key_part): img_val = getattr(img_val, key_part)
                        else: valid_img = False; break
                    if not valid_img or img_val is None:
                        # print(f"  [!] Image key '{args.image_key}' not found in step {step_id_str}, ep {episode_id}. Skipping.")
                        continue
                    
                    try:
                        image_pil = Image.fromarray(img_val.numpy()) # TF Tensor to PIL
                    except Exception as img_err:
                        # print(f"  [!] Error converting image to PIL for step {step_id_str}: {img_err}. Skipping.")
                        continue

                    step_instruction = current_instruction # Default to episode
                    if not args.instruction_key.startswith("episode_metadata"): # Try step-level
                        step_instr_src = step_data; valid_step_instr = True
                        for key_part in instr_keys: # Re-use instr_keys for step_data
                            if isinstance(step_instr_src, dict) and key_part in step_instr_src: step_instr_src = step_instr_src[key_part]
                            elif hasattr(step_instr_src, key_part): step_instr_src = getattr(step_instr_src, key_part)
                            else: valid_step_instr = False; break
                        if valid_step_instr and step_instr_src is not None:
                            if isinstance(step_instr_src, tf.Tensor) and step_instr_src.dtype == tf.string:
                                step_instruction = step_instr_src.numpy().decode('utf-8') if step_instr_src.shape.rank == 0 else (step_instr_src.numpy()[0].decode('utf-8') if step_instr_src.shape.rank>0 and step_instr_src.shape[0]>0 else step_instruction)
                            elif isinstance(step_instr_src, (str, bytes)): step_instruction = step_instr_src.decode('utf-8') if isinstance(step_instr_src, bytes) else step_instr_src

                    formatted_prompt = format_prompt_from_instruction(step_instruction, args.model_path)
                    
                    step_embeddings = extractor.extract_embeddings(
                        image_pil=image_pil, prompt=formatted_prompt, token_positions=token_positions_to_extract
                    )

                    action_np = {}
                    if 'action' in step_data:
                        for k, v_tensor in step_data['action'].items():
                            if isinstance(v_tensor, tf.Tensor): action_np[k] = v_tensor.numpy().tolist()
                            elif isinstance(v_tensor, (int, float, str, bool, list, dict)): action_np[k] = v_tensor
                    
                    metadata = {"episode_id": episode_id, "step_index": step_idx, "original_instruction": step_instruction,
                                "image_key_used": args.image_key, "instruction_key_used": args.instruction_key,
                                "layers_extracted": layer_indices_to_hook if layer_indices_to_hook is not None else list(range(extractor.num_layers)),
                                "positions_extracted": token_positions_to_extract,
                                "embedding_shapes": {k_tuple: list(v.shape) for k_tuple, v in step_embeddings.items()},
                                "action_from_dataset": action_np}
                    
                    if step_embeddings: # Only save if embeddings were extracted
                         save_embeddings(step_embeddings, step_output_dir, args.format, metadata)
                         if args.visualize: visualize_embeddings(step_embeddings, step_output_dir)
                    
                    total_steps_processed += 1
                step_pbar.close()
            except Exception as episode_err:
                print(f"\nError in episode {i} (ID: {episode_id if 'episode_id' in locals() else 'unknown'}): {episode_err}")
                traceback.print_exc()
            
        episode_pbar.close()
        extractor.clear_hooks()

        print("=" * 80)
        print(f"Processing Complete. Total Episodes: {args.num_episodes}, Total Steps: {total_steps_processed}.")
        print(f"Output directory: {Path(args.output_dir).resolve()}")
        print("=" * 80)

    except Exception as e:
        print(f"CRITICAL SCRIPT ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()