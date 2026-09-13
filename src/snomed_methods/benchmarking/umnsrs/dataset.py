"""UMNSRS Dataset loader for benchmarking semantic similarity/relatedness tasks."""

import os
from typing import Optional

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None


def _convert_labels_to_float(item: dict) -> dict:
    """Convert label string to float.

    Args:
        item: Dataset item dict with 'label' key

    Returns:
        Item with label converted to float
    """
    item = dict(item)
    if "label" in item and not isinstance(item["label"], (int, float)):
        item["label"] = float(item["label"])
    return item


def download_umnsrs(
    subset: str = "relatedness",
    split: str = "train",
    cache_dir: Optional[str] = None,
    convert_labels_to_float: bool = True,
):
    """Download and load the UMNSRS dataset from Hugging Face.

    Args:
        subset: Which subset to load. Options:
            - "relatedness": 588 concept pairs rated for semantic relatedness
            - "similarity": 566 concept pairs rated for semantic similarity
            - "relatedness_mod": 458 pairs (modified, excludes controls)
            - "similarity_mod": 449 pairs (modified, excludes controls)
        split: Dataset split (typically "train" for all subsets)
        cache_dir: Optional directory to cache downloaded data
        convert_labels_to_float: If True, convert label strings to float

    Returns:
        Hugging Face Dataset object with columns:
            - id: unique identifier
            - document_id: document identifier
            - text_1: first concept in pair
            - text_2: second concept in pair
            - label: similarity/relatedness score (float)

    Raises:
        ValueError: If invalid subset name is provided
        ConnectionError: If unable to download from Hugging Face Hub
        ImportError: If datasets library is not installed

    Example:
        >>> dataset = download_umnsrs("relatedness")
        >>> print(dataset[0])
        {'id': '0', 'document_id': '0', 'text_1': 'Carbatrol',
         'text_2': 'Dilantin', 'label': 797.5}
    """
    if load_dataset is None:
        raise ImportError(
            "The 'datasets' library is required. Install it with: "
            "pip install datasets"
        )

    valid_subsets = [
        "relatedness",
        "similarity",
        "relatedness_mod",
        "similarity_mod",
    ]

    if subset not in valid_subsets:
        raise ValueError(f"Invalid subset '{subset}'. Must be one of: {valid_subsets}")

    dataset_name = "bigbio/umnsrs"
    full_subset_name = f"umnsrs_{subset}_bigbio_pairs"

    try:
        dataset = load_dataset(
            dataset_name,
            name=full_subset_name,
            split=split,
            cache_dir=cache_dir,
        )
        if convert_labels_to_float:
            dataset = dataset.map(_convert_labels_to_float)
    except Exception as e:
        subset_info = {
            "relatedness": "(588 concept pairs rated for semantic relatedness)",
            "similarity": "(566 concept pairs rated for semantic similarity)",
            "relatedness_mod": "(458 modified pairs, excludes controls)",
            "similarity_mod": "(449 modified pairs, excludes controls)",
        }
        subset_desc = subset_info.get(subset, "")

        raise ConnectionError(
            f"Failed to download UMNSRS {subset} dataset from Hugging Face Hub.\n"
            f"{subset_desc}\n\n"
            f"Please check your internet connection and try again. If the "
            f"issue persists, you can manually download the dataset from:\n"
            f"https://huggingface.co/datasets/bigbio/umnsrs\n"
            f"Error details: {e}"
        ) from e

    return dataset


def get_umnsrs_pairs(
    subset: str = "relatedness",
    split: str = "train",
    cache_dir: Optional[str] = None,
    convert_labels_to_float: bool = True,
) -> list[dict]:
    """Get UMNSRS concept pairs as a list of dictionaries.

    Args:
        subset: Dataset subset (see download_umnsrs for options)
        split: Dataset split
        cache_dir: Optional cache directory
        convert_labels_to_float: If True, convert label strings to float

    Returns:
        List of dicts with keys: id, document_id, text_1, text_2, label
    """
    dataset = download_umnsrs(subset=subset, split=split, cache_dir=cache_dir)
    pairs = [dict(item) for item in dataset]
    if convert_labels_to_float:
        for pair in pairs:
            pair["label"] = float(pair["label"])
    return pairs


def save_umnsrs_to_csv(
    subset: str = "relatedness",
    output_path: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """Save UMNSRS dataset to CSV file.

    Args:
        subset: Dataset subset to save
        output_path: Path for output CSV
        cache_dir: Optional cache directory

    Returns:
        Path to saved CSV file
    """
    import pandas as pd  # noqa: F401

    if output_path is None:
        output_path = os.path.join(os.path.dirname(__file__), f"{subset}.csv")

    dataset = download_umnsrs(subset=subset, cache_dir=cache_dir)
    df = dataset.to_pandas()
    df["label"] = df["label"].astype(float)
    df.to_csv(output_path, index=False)

    return output_path


def load_from_csv(
    subset: str = "relatedness",
    csv_path: Optional[str] = None,
    convert_labels_to_float: bool = True,
) -> list[dict]:
    """Load UMNSRS data from a previously saved CSV file.

    Args:
        subset: Dataset subset
        csv_path: Path to CSV file
        convert_labels_to_float: If True, convert label strings to float

    Returns:
        List of dicts with keys: id, document_id, text_1, text_2, label
    """
    import pandas as pd  # noqa: F401

    if csv_path is None:
        csv_path = os.path.join(os.path.dirname(__file__), f"{subset}.csv")

    df = pd.read_csv(csv_path)
    if convert_labels_to_float and "label" in df.columns:
        df["label"] = df["label"].astype(float)
    return df.to_dict(orient="records")
