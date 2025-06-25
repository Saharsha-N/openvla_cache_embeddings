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
import traceback # For more detailed error printing

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
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'): # e.g. some LLMs store layers under model.model
             self.language_model = self.model # language_model component might be the model itself if layers are nested deeper
             print(f"[LEGITIMATE MODE] Using model itself as language_model base for layers: {type(self.language_model)}")
        else:
            # This case might indicate an unexpected model structure or that self.model itself is the language model
            # For robustness, assume self.model might directly have 'layers' or 'embed_tokens' if 'language_model' is not found
            self.language_model = self.model 
            print(f"[LEGITIMATE MODE] 'language_model' attribute not found. Assuming model attributes are on: {type(self.language_model)}")


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
            else: print(f"[LEGITIMATE MODE] Warning: Layer index {idx} (resolved: {actual_idx}) out of range [0, {self.num_layers-1}]. Skipping.")
        return sorted(list(set(valid_hooks))) # Ensure unique and sorted

    def register_hooks(self, layer_indices: Optional[List[int]] = None):
        self.clear_hooks()
        layers_to_hook = self._get_layers_to_hook(layer_indices)
        print(f"[LEGITIMATE MODE] Registering hooks on layers: {layers_to_hook or 'all applicable'}") # Changed 'all' to 'all applicable'
        print(f"[LEGITIMATE MODE] Caching streams: {self.residual_stream_types}, Layer0: {self.include_layer0_embedding}")

        self.total_layers_to_process = 0
        if "output" in self.residual_stream_types: self.total_layers_to_process += len(layers_to_hook)
        if self.include_layer0_embedding and self.embedding_layer and ("output" in self.residual_stream_types or not self.residual_stream_types):
            self.total_layers_to_process += 1
        self.layers_processed = 0

        if self.include_layer0_embedding and self.embedding_layer:
            self.hooks.append(self.embedding_layer.register_forward_hook(self._create_output_hook("L0_embedding", "embedding_output")))

        for layer_idx in layers_to_hook:
            # Additional check for safety, though _get_layers_to_hook should prevent out-of-bounds
            if 0 <= layer_idx < len(self.transformer_layers):
                module = self.transformer_layers[layer_idx]
                if "input" in self.residual_stream_types: self.hooks.append(module.register_forward_pre_hook(self._create_input_hook(layer_idx, "residual_input")))
                if "output" in self.residual_stream_types: self.hooks.append(module.register_forward_hook(self._create_output_hook(layer_idx, "residual_output")))
            else:
                print(f"[LEGITIMATE MODE] Warning: Skipped registering hook for invalid layer index {layer_idx} during loop.")

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
            with torch.no_grad(): _ = self.model(**inputs, output_hidden_states=False) # output_hidden_states=False is often default. Hooks grab intermediates.
        finally:
            if self.progress_bar:
                if self.layers_processed < self.total_layers_to_process: self.progress_bar.update(self.total_layers_to_process - self.layers_processed)
                self.progress_bar.close(); self.progress_bar = None

        final_embeddings = {}
        for key, full_emb in self.extracted_embeddings.items():
            seq_emb = full_emb.squeeze(0) if full_emb.ndim == 3 and full_emb.shape[0] == 1 else full_emb
            if seq_emb.ndim != 2: # Expect (seq_len, hidden_dim)
                print(f"[LEGITIMATE MODE] Warning: Embedding for {key} has unexpected shape {seq_emb.shape}. Skipping.")
                continue
            if token_positions == "all": final_embeddings[key] = seq_emb
            else:
                current_seq_len = seq_emb.shape[0]
                indices = [current_seq_len + p if p < 0 else p for p in token_positions if 0 <= (current_seq_len + p if p < 0 else p) < current_seq_len]
                if indices: final_embeddings[key] = seq_emb[indices]
        return final_embeddings.copy()


class MockEmbeddingExtractor:
    """
    MOCK Embedding Extractor:
    Simulates the embedding extraction process for testing script logic quickly.
    Does not load any real models or perform actual computations.
    """
    def __init__(self, model_path: str, residual_stream_types: Optional[List[str]] = None, include_layer0_embedding: bool = False, device: Optional[str]=None, torch_dtype: Optional[torch.dtype]=None, use_flash_attention: Optional[bool]=None, **kwargs): # Matched args with Real Extractor
        print(f"[MOCK MODE] Initializing MockEmbeddingExtractor for model_path: '{model_path}'")
        self.model_path = model_path
        self.device = "cpu"
        self.num_layers = 12 # Default mock layers, can be adjusted if needed for testing parse_layer_indices
        self.processor = "mock_processor_object"
        self.residual_stream_types = residual_stream_types or ["output"]
        self.include_layer0_embedding = include_layer0_embedding
        
        self.total_layers_to_process = 0
        self.progress_bar = None
        self.layers_processed = 0
        _ = kwargs

    def register_hooks(self, layer_indices: Optional[List[int]] = None):
        if layer_indices is None:
            layers_to_hook_indices = list(range(self.num_layers))
        else:
            layers_to_hook_indices = []
            for idx in layer_indices: # Simplified _get_layers_to_hook logic for mock
                actual_idx = self.num_layers + idx if idx < 0 else idx
                if 0 <= actual_idx < self.num_layers:
                    layers_to_hook_indices.append(actual_idx)
        layers_to_hook_indices = sorted(list(set(layers_to_hook_indices))) # Ensure unique & sorted for mock consistency
        
        print(f"[MOCK MODE] Simulating registering hooks for layers: {layers_to_hook_indices or 'all (based on num_layers)'}")
        
        self.total_layers_to_process = 0
        if "output" in self.residual_stream_types:
            self.total_layers_to_process += len(layers_to_hook_indices)
        if self.include_layer0_embedding and ("output" in self.residual_stream_types or not self.residual_stream_types):
            self.total_layers_to_process += 1
        
        print(f"[MOCK MODE] Expecting {self.total_layers_to_process} mock output hook calls.")


    def clear_hooks(self):
        print("[MOCK MODE] Simulating clearing hooks.")
        if self.progress_bar: self.progress_bar.close(); self.progress_bar = None


    def extract_embeddings(
        self,
        image_pil: Image.Image,
        prompt: str,
        token_positions: Union[List[int], str] = "all",
    ) -> Dict[Tuple[Union[int, str], str], torch.Tensor]:
        print(f"[MOCK MODE] Simulating extraction for image size: {image_pil.size}, prompt: '{prompt[:30]}...'")
        
        self.layers_processed = 0
        self.progress_bar = tqdm(total=self.total_layers_to_process, desc="Extracting step (mock)", unit="layer", leave=False, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        
        for _ in range(self.total_layers_to_process):
            time.sleep(0.001)
            self.layers_processed +=1
            self.progress_bar.update(1)
        if self.progress_bar: self.progress_bar.close(); self.progress_bar = None

        mock_results = {}
        hidden_dim = 768
        mock_seq_len = 25

        # This mock generation is simplified. A more advanced one would use the
        # actual layer_indices passed to register_hooks if they were stored.
        num_outputs_to_generate = self.total_layers_to_process
        
        if self.include_layer0_embedding and ("output" in self.residual_stream_types or not self.residual_stream_types):
            if num_outputs_to_generate > 0:
                mock_results[("L0_embedding", "embedding_output")] = torch.randn(mock_seq_len, hidden_dim)
                num_outputs_to_generate -=1

        for i in range(num_outputs_to_generate):
            # This 'i' is just a counter, not necessarily the true layer index
            if "output" in self.residual_stream_types:
                mock_results[(i, "residual_output")] = torch.randn(mock_seq_len, hidden_dim)
            # Input streams are not counted in total_layers_to_process for mock progress
            # if "input" in self.residual_stream_types:
            #      mock_results[(i, "residual_input")] = torch.randn(mock_seq_len, hidden_dim)
        
        final_embeddings = {}
        for key, full_seq_embedding in mock_results.items():
            if token_positions == "all":
                final_embeddings[key] = full_seq_embedding
            else:
                selected_indices = []
                for pos in token_positions:
                    actual_pos = mock_seq_len + pos if pos < 0 else pos
                    if 0 <= actual_pos < mock_seq_len:
                        selected_indices.append(actual_pos)
                if selected_indices:
                    final_embeddings[key] = full_seq_embedding[selected_indices]
        
        return final_embeddings.copy()


def save_embeddings(
    embeddings: Dict[Tuple[Union[int, str], str], torch.Tensor],
    output_dir: Path,
    format_str: str, # Renamed from 'format' to avoid conflict with built-in
    metadata: Optional[Dict] = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    for key_tuple, embedding in embeddings.items():
        layer_id, stream_name = key_tuple
        safe_layer_id = str(layer_id).replace("/", "_") 
        filename_base = f"layer_{safe_layer_id}_{stream_name}"

        if format_str == "pt":
            file_path = output_dir / f"{filename_base}.pt"
            torch.save(embedding, file_path)
        elif format_str == "npy":
            file_path = output_dir / f"{filename_base}.npy"
            np.save(file_path, embedding.cpu().numpy())
        else:
            raise ValueError(f"Unsupported format: {format_str}")

    if metadata:
        metadata_path = output_dir / "step_metadata.json"
        with open(metadata_path, "w") as f:
            # Sanitize metadata for JSON (embedding_shapes keys)
            if "embedding_shapes" in metadata:
                sanitized_shapes = {}
                # metadata["embedding_shapes"] is Dict[Tuple, List]
                # JSON keys must be strings.
                for k_tuple_meta, v_shape_list_meta in metadata["embedding_shapes"].items():
                    # k_tuple_meta is (layer_id, stream_name)
                    str_key_meta = f"L_{str(k_tuple_meta[0]).replace('/', '_')}_{k_tuple_meta[1]}"
                    sanitized_shapes[str_key_meta] = list(v_shape_list_meta) 
                metadata["embedding_shapes"] = sanitized_shapes
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
            if stacked_embeddings.ndim == 1: # Should not happen if len(data) >= 2
                if len(all_embeddings_list) >=2 : stacked_embeddings = np.vstack(all_embeddings_list)
                else: return # Not enough data points
            
            if stacked_embeddings.shape[0] < 2: return # Need at least 2 samples for PCA

            try:
                n_comp = min(2, stacked_embeddings.shape[1], stacked_embeddings.shape[0]) # n_components <= min(n_samples, n_features)
                if n_comp < 2: return # Cannot make a 2D plot
                
                pca = PCA(n_components=n_comp)
                reduced = pca.fit_transform(stacked_embeddings)
                plt.figure(figsize=(10, 8))
                plt.scatter(reduced[:, 0], reduced[:, 1])
                for i, lbl in enumerate(labels): plt.annotate(lbl, (reduced[i, 0], reduced[i, 1]))
                
                pc1_var = pca.explained_variance_ratio_[0] if n_comp > 0 and len(pca.explained_variance_ratio_)>0 else 0
                pc2_var = pca.explained_variance_ratio_[1] if n_comp > 1 and len(pca.explained_variance_ratio_)>1 else 0
                plt.xlabel(f"PC1 ({pc1_var:.2%})"); plt.ylabel(f"PC2 ({pc2_var:.2%})")
                plt.title(f"PCA (Step in {output_dir.name})"); plt.grid(True, alpha=0.3)
                plt.savefig(output_dir / "pca_step_visualization.png", dpi=300, bbox_inches="tight")
                plt.close()
            except ValueError as e: # Catch errors like n_components > n_features or n_samples
                print(f"[*] PCA Error for {output_dir.name}: {e}. Data shape: {stacked_embeddings.shape}. Skipping visualization.")
            except Exception as e_pca: # Catch other unexpected PCA errors
                 print(f"[*] Unexpected PCA Error for {output_dir.name}: {e_pca}. Skipping visualization.")


def format_prompt_from_instruction(instruction: str, model_path_or_name: str) -> str:
    instr_strip = instruction.strip()
    if instr_strip.startswith("In:") and "\nOut:" in instr_strip: return instruction # Avoid double-prompting
    return f"In: What action should the robot take to {instr_strip.lower()}?\nOut:" # Standard OpenVLA format


def parse_layer_indices(layer_str: str, num_layers: int) -> List[int]:
    if num_layers == 0: return [] # No layers to parse if model has no layers
    if layer_str.lower() == "all": return list(range(num_layers))
    global indices
    indices = set()
    for part in layer_str.split(","):
        part = part.strip()
        try:
            if "-" in part and not part.startswith("-"):
                start, end = map(int, part.split("-"))
                # Handle negative indices in ranges correctly relative to num_layers
                actual_start = num_layers + start if start < 0 else start
                actual_end = num_layers + end if end < 0 else end
                if actual_start > actual_end: # Support ranges like -1 - -5 (last 5 layers)
                    actual_start, actual_end = actual_end, actual_start
                indices.update(range(actual_start, actual_end + 1))
            else: 
                indices.add(int(part))
        except ValueError: print(f"Warning: Invalid layer specification '{part}'. Skipping.")
    # Filter for valid indices (including negative) and sort
    return sorted([idx for idx in list(indices) if -num_layers <= idx < num_layers])


def parse_token_positions(pos_str: str) -> Union[List[int], str]:
    if pos_str.lower() == "all": return "all"
    positions = set()
    for part in pos_str.split(","):
        part = part.strip()
        try:
            if "-" in part and not part.startswith("-"):
                start, end = map(int, part.split("-"))
                # Token positions don't usually use num_tokens for negative indexing range start/end yet
                # as seq_len is dynamic. Negative indices are resolved per-sequence.
                # Here, we just store the range as is.
                indices.update(range(start, end + 1))

            else: 
                positions.add(int(part))
        except ValueError: print(f"Warning: Invalid token position specification '{part}'. Skipping.")
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
    parser.add_argument("--format", type=str, choices=["pt", "npy"], default="pt", help="Output format for embeddings") # Changed arg name
    parser.add_argument("--device", type=str, default=None, help="Device ('cuda', 'cpu'). Auto-detects if None.")
    parser.add_argument("--dtype", type=str, choices=["float32", "float16", "bfloat16"], default="bfloat16", help="Model dtype")
    parser.add_argument("--no_flash_attention", action="store_true", help="Disable flash attention")
    parser.add_argument("--residual_stream_types", type=str, nargs="+", choices=["input", "output"], default=["output"], help="Residual stream types")
    parser.add_argument("--include_layer0_embedding", action="store_true", help="Include initial token embeddings (Layer 0)")
    parser.add_argument("--visualize", action="store_true", help="Create PCA visualization for each step")

    # Dataset parameters
    parser.add_argument("--dataset_name", type=str, required=True, help="TFDS dataset name (e.g., 'openvla/modified_libero_rlds') or GCS path (gs://...) or local dir path")
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
            extractor = MockEmbeddingExtractor(
                model_path=args.model_path,
                residual_stream_types=args.residual_stream_types,
                include_layer0_embedding=args.include_layer0_embedding,
                # device, torch_dtype, use_flash_attention are implicitly defaults or unused by mock
            )
        else:
            extractor = EmbeddingExtractor(
                model_path=args.model_path,
                residual_stream_types=args.residual_stream_types,
                include_layer0_embedding=args.include_layer0_embedding,
                device=args.device,
                torch_dtype=torch_dtype,
                use_flash_attention=not args.no_flash_attention,
            )

        # Common setup for both modes
        # Ensure extractor.num_layers is available. For mock, it's set in __init__.
        layer_indices_to_hook = parse_layer_indices(args.layers, extractor.num_layers) if extractor.num_layers > 0 else []
        
        token_positions_to_extract = parse_token_positions(args.positions)
        
        # Pass None to register_hooks if "all" layers are requested, otherwise the parsed list.
        # The _get_layers_to_hook method in extractors will handle None as "all layers".
        hooks_arg = None if args.layers.lower() == "all" else layer_indices_to_hook
        extractor.register_hooks(layer_indices=hooks_arg)

        # --- Dataset Loading ---
        print(f"[*] Preparing dataset: {args.dataset_name}, split: {args.dataset_split}")
        ds = None
        if args.test_mode:
            print("[MOCK MODE] Using mock dataset generator.")
            # Mock dataset generator from your provided script
            def mock_dataset_generator(num_episodes, max_steps):
                for i in range(num_episodes):
                    mock_episode_id = f"mock_ep_{i:03d}"
                    mock_steps_data = []
                    num_mock_steps = max_steps if max_steps is not None else np.random.randint(5,15)
                    for j in range(num_mock_steps):
                        mock_step = {
                            'observation': {'image': tf.zeros((64,64,3), dtype=tf.uint8)},
                            'action': {'mock_action': tf.zeros((7,), dtype=tf.float32)}
                        }
                        if not args.instruction_key.startswith("episode_metadata"):
                             nested_keys = args.instruction_key.split('.')
                             current_dict = mock_step
                             for k_idx, k_part in enumerate(nested_keys[:-1]):
                                 current_dict = current_dict.setdefault(k_part, {}) # Use setdefault
                             current_dict[nested_keys[-1]] = tf.constant(f"Mock step instruction {j}", dtype=tf.string) # Ensure dtype
                        mock_steps_data.append(mock_step)
                    
                    # Create step dataset signature from first mock step
                    first_step_sig = {}
                    if mock_steps_data: # Ensure there's at least one step to get signature
                        for k,v_top in mock_steps_data[0].items():
                            if isinstance(v_top, dict):
                                first_step_sig[k] = {ik: tf.TensorSpec.from_tensor(iv) for ik, iv in v_top.items()}
                            elif isinstance(v_top, tf.Tensor):
                                 first_step_sig[k] = tf.TensorSpec.from_tensor(v_top)
                            # Add other type handling if necessary
                    else: # Fallback signature if no steps
                         first_step_sig = {'observation': {'image': tf.TensorSpec(shape=(64,64,3),dtype=tf.uint8)}, 'action':{'mock_action':tf.TensorSpec(shape=(7,),dtype=tf.float32)}}


                    episode_entry = {'episode_id': tf.constant(mock_episode_id, dtype=tf.string), # Ensure dtype
                                     'steps': tf.data.Dataset.from_generator(lambda: (s for s in mock_steps_data), output_signature=first_step_sig)}
                    
                    # Mock episode_metadata for instruction key
                    # This part needs to be careful to match the expected signature
                    ep_meta_data_mock = {}
                    if args.instruction_key.startswith("episode_metadata"):
                        meta_keys_mock = args.instruction_key.split('.') # e.g. episode_metadata.natural_language_instruction
                        current_meta_dict_mock = ep_meta_data_mock
                        # Skip 'episode_metadata' part if it's the first key, as we are building inside 'episode_metadata' key
                        keys_to_nest_mock = meta_keys_mock[1:] if meta_keys_mock[0] == 'episode_metadata' else meta_keys_mock
                        
                        for k_idx_mock, k_part_mock in enumerate(keys_to_nest_mock[:-1]):
                            current_meta_dict_mock = current_meta_dict_mock.setdefault(k_part_mock, {})
                        current_meta_dict_mock[keys_to_nest_mock[-1]] = tf.constant(f"Mock episode instruction for {mock_episode_id}", dtype=tf.string) # Ensure dtype
                    else: # Ensure a placeholder if instruction is not in episode_metadata
                        ep_meta_data_mock['placeholder_meta'] = tf.constant("mock_placeholder", dtype=tf.string) # Ensure dtype

                    episode_entry['episode_metadata'] = ep_meta_data_mock
                    yield episode_entry

            # Define output signature for the generator based on args.image_key and args.instruction_key
            # This is complex due to nested keys.
            _mock_img_sig = tf.TensorSpec(shape=(None,None,3), dtype=tf.uint8) # Flexible image size for mock
            _mock_instr_sig = tf.TensorSpec(shape=(), dtype=tf.string)

            step_sig_dict = {}
            # Image key signature (handles nesting)
            current_level_img = step_sig_dict
            img_keys_list = args.image_key.split('.')
            for k_part_img in img_keys_list[:-1]: current_level_img = current_level_img.setdefault(k_part_img, {})
            current_level_img[img_keys_list[-1]] = _mock_img_sig
            step_sig_dict.setdefault('action', {})['mock_action'] = tf.TensorSpec(shape=(None,),dtype=tf.float32)

            # Step instruction key signature (if not in episode_metadata)
            if not args.instruction_key.startswith("episode_metadata"):
                current_level_instr_step = step_sig_dict
                instr_keys_list_step = args.instruction_key.split('.')
                for k_part_instr_step in instr_keys_list_step[:-1]: current_level_instr_step = current_level_instr_step.setdefault(k_part_instr_step, {})
                current_level_instr_step[instr_keys_list_step[-1]] = _mock_instr_sig
            
            # Episode metadata signature
            ep_meta_sig_dict_final = {}
            if args.instruction_key.startswith("episode_metadata"):
                meta_instr_key_actual = args.instruction_key.split('.', 1)[1] if '.' in args.instruction_key else args.instruction_key # key within episode_metadata
                current_level_meta = ep_meta_sig_dict_final
                meta_instr_keys_parts_actual = meta_instr_key_actual.split('.')
                for k_part_meta in meta_instr_keys_parts_actual[:-1]: current_level_meta = current_level_meta.setdefault(k_part_meta, {})
                current_level_meta[meta_instr_keys_parts_actual[-1]] = _mock_instr_sig
            else: # Ensure a placeholder for episode_metadata if instruction is step-level
                 ep_meta_sig_dict_final['placeholder_meta'] = tf.TensorSpec(shape=(), dtype=tf.string)


            episode_final_sig = {'episode_id': tf.TensorSpec(shape=(),dtype=tf.string), 
                                 'steps': tf.data.DatasetSpec(step_sig_dict),
                                 'episode_metadata': ep_meta_sig_dict_final}
            
            ds = tf.data.Dataset.from_generator(lambda: mock_dataset_generator(args.num_episodes, args.max_steps_per_episode), output_signature=episode_final_sig)
        
        # ========== START OF MODIFIED LEGITIMATE MODE DATASET LOADING ==========
        else: # LEGITIMATE MODE - Actual dataset loading
            dataset_specifier = args.dataset_name
            builder = None
            print(f"[LEGITIMATE MODE] Attempting to load: '{dataset_specifier}'")
            
            load_from_hf_or_registered = False # Flag to call download_and_prepare

            if dataset_specifier.startswith("gs://"):
                print(f"[*] Detected GCS path. Using tfds.builder_from_directory.")
                try:
                    builder = tfds.builder_from_directory(builder_dir=dataset_specifier)
                except Exception as e_gcs:
                    print(f"Error with tfds.builder_from_directory for GCS path '{dataset_specifier}': {e_gcs}")
                    print("Ensure the GCS path is correct and accessible (e.g., gsutil auth login might be needed).")
                    sys.exit(1)
            elif os.path.isdir(dataset_specifier): # Check if it's an existing local directory
                print(f"[*] Detected local directory path. Using tfds.builder_from_directory.")
                try:
                    builder = tfds.builder_from_directory(builder_dir=dataset_specifier)
                except Exception as e_local:
                    print(f"Error with tfds.builder_from_directory for local path '{dataset_specifier}': {e_local}")
                    print("Ensure the path points to a valid TFDS dataset directory.")
                    sys.exit(1)
            else: # Assume it's a registered TFDS name (e.g., from HF Hub)
                print(f"[*] Assuming registered TFDS name. Using tfds.builder with data_dir='{args.tfds_data_dir}'.")
                try:
                    builder = tfds.builder(dataset_specifier, data_dir=args.tfds_data_dir)
                    load_from_hf_or_registered = True # Set flag to download and prepare
                except tfds.core.DatasetNotFoundError:
                    print(f"Error: Dataset '{dataset_specifier}' not found by tfds.builder().")
                    print("If it's a GCS or local path, ensure it's correct. If it's a registered name, check spelling.")
                    sys.exit(1)
                except Exception as e_builder: # Catch other tfds.builder errors
                    print(f"An unexpected error occurred with tfds.builder for '{dataset_specifier}': {e_builder}")
                    sys.exit(1)
            
            if not builder: 
                print("Error: Dataset builder could not be initialized. Please check dataset_name and paths.")
                sys.exit(1)
            
            if load_from_hf_or_registered:
                print(f"[*] Calling download_and_prepare() for '{dataset_specifier}'. This may take time...")
                try:
                    builder.download_and_prepare(download_dir=args.tfds_data_dir) # download_dir for tfds.builder
                    print(f"[*] download_and_prepare() complete for '{dataset_specifier}'.")
                except Exception as e_prepare:
                    print(f"Error during download_and_prepare for '{dataset_specifier}': {e_prepare}")
                    sys.exit(1)
            
            # Validate split and load
            split_to_load = args.dataset_split
            main_split_name = split_to_load.split('[')[0] # e.g., 'train' from 'train[:10%]'
            if main_split_name not in builder.info.splits:
                print(f"Error: Split '{main_split_name}' not found in dataset '{args.dataset_name}'. Available splits: {list(builder.info.splits.keys())}")
                sys.exit(1)
            
            print(f"[*] Loading split '{split_to_load}' from dataset.")
            ds_full = builder.as_dataset(split=split_to_load) # TFDS handles slicing like "train[:10%]"
            ds = ds_full.take(args.num_episodes) # Then take N episodes from that potentially sliced split
        # ========== END OF MODIFIED LEGITIMATE MODE DATASET LOADING ==========


        # --- Episode and Step Processing Loop ---
        total_steps_processed = 0
        # For tqdm, if num_episodes is used with ds.take, total is args.num_episodes
        # If ds.take fails (e.g. on empty dataset), this might be problematic.
        # Fallback for total if ds is empty or cardinality is hard to get before iteration.
        pbar_total_episodes = args.num_episodes
        try:
            # Attempt to get cardinality if possible, but be careful as it can be costly for some datasets
            # For `ds.take(N)`, cardinality should be N if underlying dataset has at least N items.
            # However, if underlying dataset is smaller, it'll be that smaller number.
            # Using args.num_episodes as total for pbar is generally safe with ds.take().
            pass
        except Exception:
            pass # Keep args.num_episodes

        episode_pbar = tqdm(ds, total=pbar_total_episodes, desc="Episodes")
        
        processed_ep_count = 0
        for ep_idx, episode_data in enumerate(episode_pbar):
            processed_ep_count +=1
            try:
                # Safely get episode_id (robust for different TFDS versions/structures)
                ep_id_val = episode_data.get('episode_id') # Use .get for safety
                if isinstance(ep_id_val, tf.Tensor):
                    ep_id = ep_id_val.numpy().decode('utf-8') if ep_id_val.dtype == tf.string else str(ep_id_val.numpy())
                elif isinstance(ep_id_val, (str, bytes)):
                    ep_id = ep_id_val.decode('utf-8') if isinstance(ep_id_val, bytes) else str(ep_id_val)
                else: # Fallback if 'episode_id' is missing or not a tensor/str/bytes
                    ep_id = f"ep{ep_idx:03d}" # Use loop index as fallback

                episode_pbar.set_postfix_str(f"ID: {ep_id}")
                ep_output_dir = Path(args.output_dir) / ep_id
                ep_output_dir.mkdir(parents=True, exist_ok=True)

                # Helper to safely extract nested data from TFDS structures
                def get_nested_data(data_dict_or_obj, key_string):
                    keys = key_string.split('.')
                    current_data = data_dict_or_obj
                    for key_part in keys:
                        if isinstance(current_data, dict) and key_part in current_data:
                            current_data = current_data[key_part]
                        elif not isinstance(current_data, dict) and hasattr(current_data, key_part): # For object-like structures from TFDS
                            current_data = getattr(current_data, key_part)
                        else: return None # Key path not found
                    return current_data

                # Extract instruction
                instruction = "Default: No specific instruction provided for episode." # Default if key not found
                instr_val = get_nested_data(episode_data, args.instruction_key)

                if instr_val is not None:
                    if isinstance(instr_val, tf.Tensor) and instr_val.dtype == tf.string:
                        # Handle scalar tensor or first element of 1D array
                        instruction = instr_val.numpy().decode('utf-8') if instr_val.shape.rank == 0 else (instr_val.numpy()[0].decode('utf-8') if instr_val.shape.rank > 0 and instr_val.shape[0] > 0 else instruction)
                    elif isinstance(instr_val, (str,bytes)): 
                        instruction = instr_val.decode('utf-8') if isinstance(instr_val, bytes) else instr_val
                
                steps_dataset = episode_data['steps']
                
                # Determine number of steps for progress bar more robustly
                num_steps_avail_tf = tf.data.experimental.cardinality(steps_dataset)
                num_steps_avail = num_steps_avail_tf.numpy() if num_steps_avail_tf >= 0 else None # -2 for UNKNOWN, -1 for INFINITE
                
                steps_to_process_count = args.max_steps_per_episode if args.max_steps_per_episode is not None else num_steps_avail
                
                # Apply .take() to limit steps if max_steps_per_episode is set
                steps_iterable = steps_dataset
                if args.max_steps_per_episode is not None:
                    steps_iterable = steps_dataset.take(args.max_steps_per_episode)
                
                step_pbar = tqdm(steps_iterable, total=steps_to_process_count, desc=f"Steps in {ep_id}", leave=False)
                
                for step_idx, step_data in enumerate(step_pbar):
                    step_id_str = f"step_{step_idx:04d}"
                    
                    img_val = get_nested_data(step_data, args.image_key)
                    if img_val is None: 
                        print(f"Warning: Image not found for step {step_id_str}, ep {ep_id}. Key: {args.image_key}. Skipping step.")
                        continue
                    try: 
                        image_pil = Image.fromarray(img_val.numpy()) # TFDS usually provides numpy-compatible tensors
                    except Exception as e_img: 
                        print(f"Warning: Failed to convert image tensor to PIL for step {step_id_str}, ep {ep_id}: {e_img}. Type: {type(img_val)}. Skipping step.")
                        continue

                    current_step_instruction = instruction # Default to episode/global instruction
                    if not args.instruction_key.startswith("episode_metadata"): # If key targets step data, try to get it
                        step_instr_val = get_nested_data(step_data, args.instruction_key)
                        if step_instr_val is not None:
                             if isinstance(step_instr_val, tf.Tensor) and step_instr_val.dtype == tf.string:
                                current_step_instruction = step_instr_val.numpy().decode('utf-8') if step_instr_val.shape.rank == 0 else (step_instr_val.numpy()[0].decode('utf-8') if step_instr_val.shape.rank > 0 and step_instr_val.shape[0]>0 else current_step_instruction)
                             elif isinstance(step_instr_val, (str,bytes)): 
                                 current_step_instruction = step_instr_val.decode('utf-8') if isinstance(step_instr_val, bytes) else step_instr_val

                    prompt = format_prompt_from_instruction(current_step_instruction, args.model_path)
                    step_embs = extractor.extract_embeddings(image_pil, prompt, token_positions_to_extract)

                    if step_embs: # Only save if embeddings were successfully extracted
                        step_output_dir = ep_output_dir / step_id_str
                        # step_output_dir.mkdir(parents=True, exist_ok=True) # Parent ep_output_dir already exists

                        action_data = {}
                        action_val_from_step = get_nested_data(step_data, 'action') # Common key for actions
                        if isinstance(action_val_from_step, dict):
                            for k, v_tensor_action in action_val_from_step.items():
                                if isinstance(v_tensor_action, tf.Tensor): 
                                    action_data[k] = v_tensor_action.numpy().tolist() # Convert numpy arrays to lists for JSON
                                elif isinstance(v_tensor_action, (int,float,str,bool,list,dict)): 
                                    action_data[k] = v_tensor_action # Store as is if simple Python type
                                # else: print(f"Warning: Action value for key '{k}' is of unhandled type {type(v_tensor_action)}")

                        meta = {"episode_id": ep_id, "step_index": step_idx, "original_instruction": current_step_instruction,
                                "image_key_used": args.image_key, "instruction_key_used": args.instruction_key,
                                "layers_extracted": layer_indices_to_hook if args.layers.lower() != "all" else "all", # Store the actual list or "all"
                                "positions_extracted": token_positions_to_extract,
                                "embedding_shapes": {k_tuple:list(v.shape) for k_tuple,v in step_embs.items()}, # k_tuple is (layer_id, stream_name)
                                "action_from_dataset": action_data }
                        save_embeddings(step_embs, step_output_dir, args.format, meta) # Pass renamed arg
                        if args.visualize: visualize_embeddings(step_embs, step_output_dir)
                    total_steps_processed += 1
                step_pbar.close()
            except Exception as e_ep: 
                print(f"\nError processing episode data for ep_idx {ep_idx} (Resolved ID: {ep_id if 'ep_id' in locals() else 'unknown'}): {e_ep}")
                traceback.print_exc()
                print(f"Skipping rest of this episode.")
            
        episode_pbar.close()
        if processed_ep_count < args.num_episodes:
            print(f"Warning: Processed {processed_ep_count} episodes, but {args.num_episodes} were requested. Dataset might have fewer episodes than requested in the split.")

        extractor.clear_hooks()
        print("="*80 + f"\nProcessing Complete. Episodes processed: {processed_ep_count}, Total Steps: {total_steps_processed}." + f"\nOutput: {Path(args.output_dir).resolve()}" + "\n" + "="*80)
    except Exception as e_main: 
        print(f"CRITICAL SCRIPT ERROR: {e_main}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()