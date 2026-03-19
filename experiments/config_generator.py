from itertools import product
from typing import Dict, List


def generate_configs() -> List[Dict]:
    """
    Generate a small grid of RAG configuration dictionaries.
    """

    chunk_sizes = [300, 500]
    overlaps = [50, 100]
    top_ks = [3, 5]
    prompt_styles = ["basic", "strict"]

    configs: List[Dict] = []

    for chunk_size, overlap, top_k, prompt_style in product(
        chunk_sizes, overlaps, top_ks, prompt_styles
    ):
        configs.append(
            {
                "chunk_size": chunk_size,
                "overlap": overlap,
                "top_k": top_k,
                "prompt_style": prompt_style,
            }
        )

    return configs

