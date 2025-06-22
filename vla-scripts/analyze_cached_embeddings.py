#!/usr/bin/env python3
"""
analyze_cached_embeddings.py

Example script showing how to load and analyze cached embeddings from cache_embeddings.py.
Demonstrates various analysis techniques including similarity computation, clustering, and visualization.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity


def load_cached_embeddings(embeddings_dir: str) -> Dict[str, torch.Tensor]:
    """Load cached embeddings from a directory."""
    embeddings_path = Path(embeddings_dir)
    
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings directory not found: {embeddings_dir}")
    
    # Load metadata
    metadata_path = embeddings_path / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        print(f"[*] Loaded metadata for {len(metadata['embedding_shapes'])} embeddings")
        print(f"[*] Model: {metadata['model_path']}")
        print(f"[*] Prompt: {metadata['prompt']}")
    else:
        print("[*] No metadata found")
        metadata = {}
    
    # Load embeddings
    embeddings = {}
    for file_path in embeddings_path.glob("*.pt"):
        key = file_path.stem  # Remove .pt extension
        embedding = torch.load(file_path, map_location="cpu")
        embeddings[key] = embedding
        print(f"[*] Loaded {key}: {embedding.shape}")
    
    return embeddings, metadata


def compute_layer_similarities(embeddings: Dict[str, torch.Tensor]) -> np.ndarray:
    """Compute cosine similarities between embeddings from different layers."""
    # Extract layer embeddings (assuming format layer_X_pos_Y)
    layer_embeddings = {}
    for key, embedding in embeddings.items():
        if "layer_" in key and "pos_" in key:
            parts = key.split("_")
            layer_idx = int(parts[1])
            layer_embeddings[layer_idx] = embedding.numpy()
    
    if not layer_embeddings:
        print("[*] No layer embeddings found")
        return np.array([])
    
    # Sort by layer index
    sorted_layers = sorted(layer_embeddings.keys())
    embeddings_matrix = np.stack([layer_embeddings[layer] for layer in sorted_layers])
    
    # Compute cosine similarities
    similarities = cosine_similarity(embeddings_matrix)
    
    print(f"[*] Computed similarities for layers: {sorted_layers}")
    return similarities, sorted_layers


def visualize_layer_progression(embeddings: Dict[str, torch.Tensor], output_dir: str):
    """Visualize how embeddings change across layers using PCA."""
    # Extract layer embeddings
    layer_embeddings = {}
    for key, embedding in embeddings.items():
        if "layer_" in key and "pos_" in key:
            parts = key.split("_")
            layer_idx = int(parts[1])
            layer_embeddings[layer_idx] = embedding.numpy()
    
    if len(layer_embeddings) < 2:
        print("[*] Need at least 2 layer embeddings for progression visualization")
        return
    
    # Sort by layer index
    sorted_layers = sorted(layer_embeddings.keys())
    embeddings_matrix = np.stack([layer_embeddings[layer] for layer in sorted_layers])
    
    # Apply PCA
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings_matrix)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot trajectory
    plt.plot(embeddings_2d[:, 0], embeddings_2d[:, 1], 'b-', alpha=0.6, linewidth=2, label='Layer progression')
    
    # Plot points
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                         c=sorted_layers, cmap='viridis', s=100, alpha=0.8)
    
    # Add layer labels
    for i, layer in enumerate(sorted_layers):
        plt.annotate(f'L{layer}', (embeddings_2d[i, 0], embeddings_2d[i, 1]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    plt.colorbar(scatter, label='Layer Index')
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    plt.title('Embedding Evolution Across Transformer Layers')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    output_path = Path(output_dir) / "layer_progression.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[*] Saved layer progression plot to {output_path}")


def visualize_similarity_matrix(similarities: np.ndarray, layer_indices: List[int], output_dir: str):
    """Visualize the similarity matrix between layers."""
    plt.figure(figsize=(10, 8))
    
    im = plt.imshow(similarities, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, label='Cosine Similarity')
    
    # Set ticks and labels
    plt.xticks(range(len(layer_indices)), [f'L{i}' for i in layer_indices])
    plt.yticks(range(len(layer_indices)), [f'L{i}' for i in layer_indices])
    
    # Add text annotations
    for i in range(len(layer_indices)):
        for j in range(len(layer_indices)):
            plt.text(j, i, f'{similarities[i, j]:.2f}', 
                    ha='center', va='center', fontsize=8)
    
    plt.title('Layer-wise Embedding Similarities')
    plt.xlabel('Layer Index')
    plt.ylabel('Layer Index')
    
    # Save plot
    output_path = Path(output_dir) / "similarity_matrix.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[*] Saved similarity matrix to {output_path}")


def analyze_embedding_statistics(embeddings: Dict[str, torch.Tensor]):
    """Compute and display basic statistics about the embeddings."""
    print("\n" + "="*60)
    print("EMBEDDING STATISTICS")
    print("="*60)
    
    for key, embedding in embeddings.items():
        emb_np = embedding.numpy()
        
        print(f"\n{key}:")
        print(f"  Shape: {emb_np.shape}")
        print(f"  Mean: {emb_np.mean():.4f}")
        print(f"  Std: {emb_np.std():.4f}")
        print(f"  Min: {emb_np.min():.4f}")
        print(f"  Max: {emb_np.max():.4f}")
        print(f"  L2 Norm: {np.linalg.norm(emb_np):.4f}")
        print(f"  Sparsity: {(emb_np == 0).mean():.2%}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze cached embeddings from cache_embeddings.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic analysis
    python vla-scripts/analyze_cached_embeddings.py embeddings_test/
    
    # Full analysis with visualizations
    python vla-scripts/analyze_cached_embeddings.py embeddings_test/ --output_dir analysis/ --visualize
        """
    )
    
    parser.add_argument(
        "embeddings_dir",
        type=str,
        help="Directory containing cached embeddings"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis",
        help="Output directory for analysis results"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Create visualizations"
    )
    
    args = parser.parse_args()
    
    try:
        # Load embeddings
        print("="*60)
        print("LOADING CACHED EMBEDDINGS")
        print("="*60)
        
        embeddings, metadata = load_cached_embeddings(args.embeddings_dir)
        
        if not embeddings:
            print("[*] No embeddings found!")
            return
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Analyze statistics
        analyze_embedding_statistics(embeddings)
        
        # Compute similarities
        similarities, layer_indices = compute_layer_similarities(embeddings)
        
        if len(similarities) > 0:
            print(f"\n[*] Layer similarity matrix shape: {similarities.shape}")
            
            if args.visualize:
                print("\n" + "="*60)
                print("CREATING VISUALIZATIONS")
                print("="*60)
                
                visualize_similarity_matrix(similarities, layer_indices, args.output_dir)
                visualize_layer_progression(embeddings, args.output_dir)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
