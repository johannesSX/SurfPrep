"""
datasets_adapter.py — Normalize the dataset readers into one record format
==========================================================================

Each dataset class in `datasets.py` (DatasetIXI, DatasetFCDBONN,
DatasetIdeas) returns its files in its own layout. This adapter converts
them into a single uniform per-subject record that the SurfPrep steps
consume:

  - study_uid:      unique, filesystem-safe subject ID
                    (cohort-prefixed, e.g. fcdbonn__sub-00055, ixi__IXI002-Guys)
  - t1/flair/t2/swi: list of {'par_name': str, 'nii_name': str, 'full_path': str}
  - seg_masks:      list of segmentation-mask paths (ground truth; eval only)
  - not_healthy:    bool (subject has a lesion)
  - keyword:        cohort name ('fcdbonn' | 'ixi' | 'ideas')
  - study_befund:   'mit Befund' / 'ohne Befund' (kept for compatibility)

Usage:
  from datasets_adapter import read_data_ext

  # all supported cohorts
  records = read_data_ext()

  # specific cohorts only
  records = read_data_ext(datasets=["FCDBONN", "IXI", "Ideas"])
"""

import os
import pathlib
from typing import Dict, List, Optional

# Import external dataset classes
from datasets import (
    DatasetIXI,
    DatasetFCDBONN,
    DatasetIdeas,
)


# ─────────────────────────────────────────────────────────
# Registry of available datasets
# ─────────────────────────────────────────────────────────

DATASET_REGISTRY = {
    "IXI":      DatasetIXI,
    "FCDBONN":  DatasetFCDBONN,
    "Ideas":    DatasetIdeas,
}


# ─────────────────────────────────────────────────────────
# Derive a unique subject ID from a NIfTI file path
# ─────────────────────────────────────────────────────────

def _derive_subject_id(file_path: str, keyword: str) -> str:
    """
    Derive a unique, filesystem-safe subject ID from a NIfTI path.

    Strategy per dataset:
      FCDBONN:  ../data/FCDBONN/sub-XXX/anat/sub-XXX_T1w.nii.gz  → fcdbonn__sub-XXX
      IXI:      ../data/IXI/IXI-T1/IXI002-Guys-0828-T1.nii.gz    → ixi__IXI002-Guys
      IDEAS:    ../data/IDEAS/ds005602/sub-1/anat/sub-1_T1w.nii.gz → ideas__sub-1
      Generic:  use parent dir or filename stem
    """
    p = pathlib.Path(file_path)
    kw = keyword.lower()

    if kw == "fcdbonn":
        # .../FCDBONN/sub-XXX/anat/sub-XXX_T1w.nii.gz
        # subject dir is 2 levels up from the .nii.gz
        subject_dir = p.parent.parent.name  # "sub-XXX"
        return f"fcdbonn__{subject_dir}"

    elif kw == "ixi":
        # IXI-XXX-...-T1.nii.gz → extract IXI ID
        stem = p.stem.replace(".nii", "")
        parts = stem.split("-")
        if len(parts) >= 2:
            return f"ixi__{parts[0]}-{parts[1]}"
        return f"ixi__{stem}"

    elif kw == "ideas":
        # ../data/IDEAS/ds005602/sub-1/anat/sub-1_T1w.nii.gz
        for part in p.parts:
            if part.startswith("sub-"):
                return f"ideas__{part}"
        return f"ideas__{p.parent.parent.name}"

    else:
        # Generic fallback
        stem = p.stem.replace(".nii", "")
        return f"{kw}__{stem}"


# ─────────────────────────────────────────────────────────
# Convert a single dataset entry → uniform record
# ─────────────────────────────────────────────────────────

def _path_to_entry(file_path: str) -> dict:
    """
    Convert an absolute file path into the {'par_name': ..., 'nii_name': ...}
    format that the SurfPrep steps expect.

    The pipeline uses these to build:
      src_path / f"UID_{study_uid}" / entry["par_name"] / entry["nii_name"]

    Since datasets has absolute paths, we set:
      par_name = ""  (empty — no subdirectory)
      nii_name = full absolute path

    Then in prepare_jobs we handle this by detecting absolute paths.
    Actually — better: store the full path and let the adapter's
    prepare_jobs equivalent handle it.
    """
    p = pathlib.Path(file_path)
    return {
        "par_name": str(p.parent),
        "nii_name": p.name,
        "full_path": str(p),  # Extra field for direct access
    }


def convert_ext_to_libdata(ext_dict: dict) -> Optional[dict]:
    """
    Convert one dataset entry → one uniform record dict.

    Returns None if no T1 is available (FastSurfer needs T1).
    """
    keyword = ext_dict.get("keyword", ["unknown"])[0]

    # Find the first available image path to derive subject ID
    first_path = None
    for seq_key in ["t1", "flair", "t2", "swi", "t1ks"]:
        paths = ext_dict.get(seq_key, [])
        if paths and isinstance(paths[0], str):
            first_path = paths[0]
            break

    if first_path is None:
        return None

    subject_id = _derive_subject_id(first_path, keyword)

    # Build uniform record dict
    result = {
        "study_uid": subject_id,
        "study_befund": "mit Befund" if ext_dict.get("not_healthy", [False])[0] else "ohne Befund",
        "city": None,
        "annotations": [],  # datasets uses mask-based seg, not point annotations

        # Sequence entries — convert paths to {'par_name', 'nii_name'} format
        "t1": [],
        "km_t1": [],
        "t2": [],
        "flair": [],
        "swi": [],
        "stir": [],

        # Extra fields from datasets (carried through for the steps)
        "seg_masks": ext_dict.get("seg", []),       # Segmentation mask paths
        "not_healthy": ext_dict.get("not_healthy", [False])[0],
        "keyword": keyword,
        "mask": ext_dict.get("mask", []),
    }

    # Convert sequence paths
    for seq_key in ["t1", "t2", "flair", "swi"]:
        paths = ext_dict.get(seq_key, [])
        for path in paths:
            if isinstance(path, str) and os.path.isabs(path):
                result[seq_key].append(_path_to_entry(path))
            elif isinstance(path, str):
                # Relative path — make absolute
                abs_path = str(pathlib.Path(path).resolve())
                result[seq_key].append(_path_to_entry(abs_path))

    # t1ks → km_t1
    for path in ext_dict.get("t1ks", []):
        if isinstance(path, str):
            abs_path = str(pathlib.Path(path).resolve()) if not os.path.isabs(path) else path
            result["km_t1"].append(_path_to_entry(abs_path))

    return result


# ─────────────────────────────────────────────────────────
# Main entry point — replaces the dataset adapter
# ─────────────────────────────────────────────────────────

def read_data_ext(
    datasets: Optional[List[str]] = None,
    require_t1: bool = True,
) -> List[dict]:
    """
    Load external datasets and return them in the uniform record format.

    This is a drop-in replacement for read_data_from_db() in the original pipeline
    in the original pipeline.

    Args:
        datasets:   List of dataset names to load (default: all).
                    Valid names: IXI, FCDBONN, Ideas
        require_t1: If True, skip subjects without a T1 sequence.

    Returns:
        List of uniform per-subject record dicts for the SurfPrep steps.
    """
    if datasets is None:
        datasets = list(DATASET_REGISTRY.keys())

    lst_dicts = []
    seen_ids = set()

    for ds_name in datasets:
        if ds_name not in DATASET_REGISTRY:
            print(f"WARNING: Unknown dataset '{ds_name}', skipping. "
                  f"Available: {list(DATASET_REGISTRY.keys())}")
            continue

        cls = DATASET_REGISTRY[ds_name]
        try:
            ds = cls()
            ext_data = ds.read()
        except Exception as e:
            print(f"WARNING: Failed to load {ds_name}: {e}")
            continue

        n_loaded = 0
        for ext_dict in ext_data:
            converted = convert_ext_to_libdata(ext_dict)
            if converted is None:
                continue
            if require_t1 and len(converted["t1"]) == 0:
                continue

            # Ensure unique subject IDs
            sid = converted["study_uid"]
            if sid in seen_ids:
                # Append counter for duplicates
                counter = 1
                while f"{sid}_{counter}" in seen_ids:
                    counter += 1
                sid = f"{sid}_{counter}"
                converted["study_uid"] = sid

            seen_ids.add(sid)
            lst_dicts.append(converted)
            n_loaded += 1

        print(f"  {ds_name}: loaded {n_loaded} subjects "
              f"(from {len(ext_data)} entries)")

    print(f"\nTotal: {len(lst_dicts)} subjects loaded from external datasets")
    return lst_dicts


# ─────────────────────────────────────────────────────────
# Helper: resolve full NIfTI path from adapted entry
# ─────────────────────────────────────────────────────────

def resolve_nii_path(entry: dict, study_uid: str = None, src_path: str = None) -> str:
    """
    Get the full NIfTI path from an entry dict.

    For datasets adapted entries, the path is stored in 'full_path'
    or reconstructed from par_name/nii_name (which are absolute).

    For path-based entries, uses:
      src_path / f"UID_{study_uid}" / entry["par_name"] / entry["nii_name"]
    """
    # Direct full path (datasets adapter)
    if "full_path" in entry:
        return entry["full_path"]

    # Reconstruct from par_name / nii_name
    par = entry.get("par_name", "")
    nii = entry.get("nii_name", "")

    # If par_name is an absolute path, just join with nii_name
    if os.path.isabs(par):
        return os.path.join(par, nii)

    # Original the uniform record format
    if src_path and study_uid:
        return os.path.join(src_path, f"UID_{study_uid}", par, nii)

    return os.path.join(par, nii)


# ─────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    ds_names = sys.argv[1:] if len(sys.argv) > 1 else None

    print("=" * 60)
    print("datasets_adapter — Test")
    print("=" * 60)

    lst_dicts = read_data_ext(datasets=ds_names)

    print(f"\n{'='*60}")
    print("Sample entries:")
    print(f"{'='*60}")
    for d in lst_dicts[:5]:
        print(f"\n  study_uid:    {d['study_uid']}")
        print(f"  study_befund: {d['study_befund']}")
        print(f"  keyword:      {d['keyword']}")
        n_t1 = len(d['t1'])
        n_flair = len(d['flair'])
        n_t2 = len(d['t2'])
        n_swi = len(d['swi'])
        print(f"  sequences:    T1={n_t1}, FLAIR={n_flair}, T2={n_t2}, SWI={n_swi}")
        if d['t1']:
            path = resolve_nii_path(d['t1'][0])
            exists = os.path.exists(path)
            print(f"  T1 path:      {path}  [exists={exists}]")
        if d['seg_masks']:
            print(f"  seg masks:    {d['seg_masks']}")
        print(f"  not_healthy:  {d['not_healthy']}")

    # Summary by keyword
    print(f"\n{'='*60}")
    print("Summary by dataset:")
    print(f"{'='*60}")
    from collections import Counter
    kw_counts = Counter(d['keyword'] for d in lst_dicts)
    befund_counts = Counter(d['study_befund'] for d in lst_dicts)
    seg_count = sum(1 for d in lst_dicts if d['seg_masks'])

    for kw, count in sorted(kw_counts.items()):
        n_seg = sum(1 for d in lst_dicts if d['keyword'] == kw and d['seg_masks'])
        n_flair = sum(1 for d in lst_dicts if d['keyword'] == kw and d['flair'])
        print(f"  {kw:12s}: {count:5d} subjects  "
              f"(seg: {n_seg}, flair: {n_flair})")

    print(f"\n  mit Befund:   {befund_counts.get('mit Befund', 0)}")
    print(f"  ohne Befund:  {befund_counts.get('ohne Befund', 0)}")
    print(f"  with seg mask: {seg_count}")