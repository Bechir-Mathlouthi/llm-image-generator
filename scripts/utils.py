import os
import json
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

def load_metadata(metadata_dir="metadata"):
    """Load all metadata files from the metadata directory."""
    metadata_dir = Path(metadata_dir)
    all_metadata = []
    
    if metadata_dir.exists():
        for file in metadata_dir.glob("*.json"):
            with open(file, "r") as f:
                metadata = json.load(f)
                all_metadata.append(metadata)
    
    return sorted(all_metadata, key=lambda x: x["timestamp"], reverse=True)

def display_image_gallery(metadata_list, cols=3):
    """Display a gallery of generated images with their prompts."""
    rows = (len(metadata_list) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
    axes = axes.flatten() if rows > 1 else [axes]
    
    for idx, metadata in enumerate(metadata_list):
        if os.path.exists(metadata["image_path"]):
            img = plt.imread(metadata["image_path"])
            axes[idx].imshow(img)
            axes[idx].axis("off")
            axes[idx].set_title(f"Prompt: {metadata['prompt'][:50]}...")
    
    # Hide empty subplots
    for idx in range(len(metadata_list), len(axes)):
        axes[idx].axis("off")
    
    plt.tight_layout()
    return fig

def get_image_stats(metadata_dir="metadata"):
    """Get statistics about generated images."""
    metadata_list = load_metadata(metadata_dir)
    
    stats = {
        "total_images": len(metadata_list),
        "devices_used": set(m["device"] for m in metadata_list),
        "model_versions": set(m["model_id"] for m in metadata_list),
        "generation_dates": sorted(set(m["timestamp"][:8] for m in metadata_list))
    }
    
    return stats 