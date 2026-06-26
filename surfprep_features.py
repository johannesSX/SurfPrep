"""
run_2.py — Post-FastSurfer Feature Extraction Pipeline
=======================================================

Reusable post-FastSurfer feature-extraction engine used by
checks which reconstructions succeeded/failed, and extracts surface-based
features for downstream analysis.

Multi-contrast support:
  Each modality (FLAIR, T2, SWI) can have multiple sequences per subject.
  The best sequence is auto-selected via scoring (like T1 in SurfPrep),
  preferring 3D acquisitions, appropriate orientations, and penalizing
  scouts/post-contrast. Override with --contrast_selection first to skip scoring.

Pipeline Steps:
  1. QC — Check reconstruction quality (Euler number), mark failures in data_dict
  2. Register contrasts (FLAIR, T2, SWI) → T1 space (rigid, per subject)
  3. Sample each registered contrast onto cortical surface at 7 depths
  4. Compute gray–white contrast from T1 (pctsurfcon)
  5. Extract morphometric features (thickness, curvature, sulcal depth, area)
  6. Verify fsaverage registration (lh/rh.sphere.reg)
  7. Resample all features to fsaverage (163,842 vertices per hemisphere)
  8. Resample DKT atlas annotation to fsaverage (nearest-neighbor)
  9. Compute geodesic distance matrix on fsaverage mesh
 10. Compute vertex positions (x, y, z coordinates on fsaverage)
 11. Stack into vertex_features.npz per subject
 12. Split dataset — healthy ("ohne Befund") vs pathological ("mit Befund")

Usage:
  python run_2.py                                  # Full pipeline
  python run_2.py --step qc                        # QC only
  python run_2.py --step features                  # Feature extraction only
  python run_2.py --contrast_selection first        # Use first entry, no scoring
  python run_2.py --modalities flair t2             # Only FLAIR + T2, skip SWI
  python run_2.py --dry_run --limit 5               # Test on 5 subjects
  python run_2.py --workers 8                       # Parallel
"""

import argparse
import json
import logging
import os
import pathlib
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# ─────────────────────────────────────────────────────────
# FreeSurfer paths
# ─────────────────────────────────────────────────────────

FREESURFER_HOME = os.environ.get("FREESURFER_HOME", "/usr/local/freesurfer")
FREESURFER_BIN = os.path.join(FREESURFER_HOME, "bin")

# Set environment variables that FreeSurfer commands need internally
# We append FreeSurfer bin to PATH (at the end) so shell-script commands like
# pctsurfcon can find sub-commands (mri_vol2surf etc.), but the conda env's
# Python still takes priority since it comes first in PATH.
os.environ["FREESURFER_HOME"] = FREESURFER_HOME
os.environ["FREESURFER"] = FREESURFER_HOME
os.environ["SUBJECTS_DIR"] = os.environ.get(
    "SUBJECTS_DIR",
    os.path.join(FREESURFER_HOME, "subjects"),
)
os.environ["FUNCTIONALS_DIR"] = os.path.join(FREESURFER_HOME, "sessions")
os.environ["MINC_BIN_DIR"] = os.path.join(FREESURFER_HOME, "mni", "bin")
os.environ["MNI_DIR"] = os.path.join(FREESURFER_HOME, "mni")
os.environ["MINC_LIB_DIR"] = os.path.join(FREESURFER_HOME, "mni", "lib")
os.environ["MNI_DATAPATH"] = os.path.join(FREESURFER_HOME, "mni", "data")
os.environ["MNI_PERL5LIB"] = os.path.join(FREESURFER_HOME, "mni", "share", "perl5")
os.environ["PERL5LIB"] = os.environ.get(
    "PERL5LIB", os.path.join(FREESURFER_HOME, "mni", "share", "perl5")
)
# FreeSurfer needs its lib dir on LD_LIBRARY_PATH for some commands
_fs_lib = os.path.join(FREESURFER_HOME, "lib")
_fs_lib_gcc = os.path.join(FREESURFER_HOME, "lib", "gcc", "lib")
_existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
os.environ["LD_LIBRARY_PATH"] = f"{_fs_lib}:{_fs_lib_gcc}:{_existing_ld}".rstrip(":")
# Append FreeSurfer bin to PATH (at end, so conda Python is not overridden)
_existing_path = os.environ.get("PATH", "")
os.environ["PATH"] = f"{_existing_path}:{FREESURFER_BIN}".lstrip(":")


def fs_cmd(command):
    """Return full path to FreeSurfer command."""
    return os.path.join(FREESURFER_BIN, command)


# ─────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────

HEMIS = ("lh", "rh")
N_FSAVERAGE_VERTICES = 163842

# Cortical sampling depths: fraction of cortical thickness
# -0.5 = into WM, 0.0 = WM/GM boundary, 1.0 = pial surface
SAMPLING_DEPTHS = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]

# Morphometric features extracted from FastSurfer output
MORPHO_FEATURES = ["thickness", "curv", "sulc", "area"]

# Modalities that can be mapped onto the surface
ALL_MODALITIES = ["flair", "t2", "swi"]

# Euler number threshold — subjects worse than this are flagged
EULER_THRESHOLD = -200


# ─────────────────────────────────────────────────────────
# Contrast Scoring (like score_t1 in SurfPrep)
# ─────────────────────────────────────────────────────────

def _base_score(par_name: str) -> int:
    """
    Base scoring shared across all modalities.
    Rewards 3D, sagittal, isotropic; penalizes scouts, post-contrast, 2D thick-slice.
    """
    name = par_name.lower()
    score = 0

    # ── Strongly prefer 3D acquisitions ──
    if "3d" in name or "3 d" in name:
        score += 20

    # ── Prefer sagittal (usually 3D isotropic) ──
    if "sag" in name:
        score += 5

    # ── Isotropic indicators ──
    if "iso" in name:
        score += 5
    if "1mm" in name or "1.0mm" in name or "0.9mm" in name:
        score += 3
    if "0.5mm" in name or "0.6mm" in name or "0.7mm" in name:
        score += 2

    # ── Penalize 2D / thick slice sequences ──
    if "2d" in name:
        score -= 10
    if "tra" in name and "3d" not in name:
        score -= 5
    if "cor" in name and "3d" not in name:
        score -= 3

    # ── Hard penalties ──
    if "scout" in name or "localizer" in name:
        score -= 100
    if "survey" in name:
        score -= 100
    if "km" in name or "gad" in name or "contrast" in name or "post" in name:
        score -= 50

    return score


def score_flair(par_name: str) -> int:
    """
    Score a FLAIR sequence for surface projection.
    Prefers: 3D FLAIR (SPACE, CUBE) > 2D axial FLAIR.
    3D FLAIR has better through-plane resolution for cortical sampling.
    """
    name = par_name.lower()
    score = _base_score(par_name)

    # ── FLAIR-specific bonuses ──
    if "space" in name:           # Siemens 3D FLAIR
        score += 15
    if "cube" in name:            # GE 3D FLAIR
        score += 15
    if "vista" in name:           # Philips 3D FLAIR
        score += 15
    if "flair" in name:
        score += 5                # Confirm it's actually FLAIR

    # ── Penalize non-FLAIR that might be miscategorized ──
    if "tse" in name and "3d" not in name:
        score -= 3                # 2D TSE FLAIR is common but thick-sliced
    if "blade" in name:
        score -= 5                # Motion-corrected 2D, usually thick

    return score


def score_t2(par_name: str) -> int:
    """
    Score a T2 sequence for surface projection.
    Prefers: 3D T2 (SPACE, CUBE, VISTA) > 2D axial T2 TSE.
    """
    name = par_name.lower()
    score = _base_score(par_name)

    # ── T2-specific bonuses ──
    if "space" in name:           # Siemens 3D T2
        score += 15
    if "cube" in name:            # GE 3D T2
        score += 15
    if "vista" in name:           # Philips 3D T2
        score += 15
    if "drive" in name:           # Philips 3D T2 variant
        score += 10

    # ── T2 TSE is standard but 2D ──
    if "tse" in name and "3d" not in name:
        score -= 3
    if "blade" in name:
        score -= 5

    return score


def score_swi(par_name: str) -> int:
    """
    Score an SWI sequence for surface projection.
    SWI is almost always 3D. Prefer magnitude images, avoid phase-only.
    """
    name = par_name.lower()
    score = _base_score(par_name)

    # ── SWI-specific bonuses ──
    if "swi" in name:
        score += 10
    if "swan" in name:            # GE's SWI equivalent
        score += 10
    if "venobold" in name:        # Philips SWI
        score += 10

    # ── Prefer combined/magnitude, penalize phase-only ──
    if "mag" in name:
        score += 5
    if "mip" in name:             # Minimum intensity projection — processed
        score += 3
    if "pha" in name and "mag" not in name:
        score -= 10               # Phase-only is less useful for cortical mapping
    if "filtered" in name:
        score += 3                # Filtered SWI (combined) is better

    return score


SCORE_FUNCTIONS = {
    "flair": score_flair,
    "t2": score_t2,
    "swi": score_swi,
}


def select_best_contrast(entries: list, modality: str) -> Optional[dict]:
    """
    Select the best sequence from a list of entries for a given modality.
    Each entry: {'par_name': str, 'nii_name': str}
    """
    if not entries:
        return None
    if len(entries) == 1:
        return entries[0]

    score_fn = SCORE_FUNCTIONS.get(modality, _base_score)
    if modality not in SCORE_FUNCTIONS:
        scored = [(_base_score(e["par_name"]), e) for e in entries]
    else:
        scored = [(score_fn(e["par_name"]), e) for e in entries]

    scored.sort(key=lambda x: x[0], reverse=True)

    for rank, (s, e) in enumerate(scored):
        marker = " <-- SELECTED" if rank == 0 else ""
        logging.debug(f"  {modality.upper()} [{s:+3d}] {e['par_name']}/{e['nii_name']}{marker}")

    return scored[0][1]


# ─────────────────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────────────────

@dataclass
class SubjectRecord:
    """One subject's processing state."""
    subject_id: str
    study_uid: str
    subjects_dir_path: str          # e.g. /fastsurfer_subjects/<sid>
    t1_nii: Optional[str] = None    # Path to original T1 NIfTI
    befund: Optional[str] = None    # "mit Befund" / "ohne Befund"
    has_annotations: bool = False
    city: Optional[str] = None

    # Per-modality NIfTI paths (selected best or first)
    # Key: modality name ("flair", "t2", "swi"), Value: path string
    contrast_niis: Dict[str, str] = field(default_factory=dict)
    # Par names for logging
    contrast_par_names: Dict[str, str] = field(default_factory=dict)

    # QC
    euler_lh: Optional[int] = None
    euler_rh: Optional[int] = None
    has_seg: bool = False
    has_surf: bool = False
    has_sphere_reg: bool = False
    qc_pass: bool = False
    failure_reason: Optional[str] = None

    # Processing flags per modality
    registered: Dict[str, bool] = field(default_factory=dict)   # {"flair": True, ...}
    features_extracted: bool = False


# ─────────────────────────────────────────────────────────
# Quality Control
# ─────────────────────────────────────────────────────────

def get_euler_number(subject_dir: pathlib.Path, hemi: str) -> Optional[int]:
    """
    Compute Euler number for a hemisphere surface using mris_euler_number.
    Returns the Euler number (int) or None if it fails.
    """
    surf_file = subject_dir / "surf" / f"{hemi}.white"
    if not surf_file.exists():
        return None

    try:
        proc = subprocess.run(
            [fs_cmd("mris_euler_number"), str(surf_file)],
            capture_output=True, text=True, timeout=60,
        )
        for line in proc.stderr.splitlines() + proc.stdout.splitlines():
            if "euler" in line.lower() and "=" in line:
                val = line.split("=")[-1].strip().split()[0]
                return int(val)
    except Exception as e:
        logging.debug(f"Euler number failed for {surf_file}: {e}")
    return None


def run_qc(record: SubjectRecord, config: dict) -> SubjectRecord:
    """
    Quality-check a single subject's FastSurfer output.
    Sets qc_pass, failure_reason, euler numbers, and file existence flags.
    """
    sd = pathlib.Path(record.subjects_dir_path)

    # ── Check segmentation ──
    seg_file = sd / "mri" / "aparc.DKTatlas+aseg.deep.mgz"
    record.has_seg = seg_file.exists() and seg_file.stat().st_size > 1000

    if not record.has_seg:
        record.qc_pass = False
        record.failure_reason = "Failed:seg_missing"
        return record

    # ── Check surfaces ──
    surfs_ok = all(
        (sd / "surf" / f"{h}.{s}").exists()
        for h in HEMIS
        for s in ("white", "pial")
    )
    record.has_surf = surfs_ok

    if not surfs_ok:
        record.qc_pass = False
        record.failure_reason = "Failed:surf_missing"
        return record

    # ── Check fsaverage registration ──
    record.has_sphere_reg = all(
        (sd / "surf" / f"{h}.sphere.reg").exists()
        for h in HEMIS
    )
    if not record.has_sphere_reg:
        record.qc_pass = False
        record.failure_reason = "Failed:sphere_reg_missing"
        return record

    # ── Euler number check ──
    record.euler_lh = get_euler_number(sd, "lh")
    record.euler_rh = get_euler_number(sd, "rh")

    if record.euler_lh is not None and record.euler_lh < EULER_THRESHOLD:
        record.qc_pass = False
        record.failure_reason = f"Failed:euler_lh={record.euler_lh}"
        return record
    if record.euler_rh is not None and record.euler_rh < EULER_THRESHOLD:
        record.qc_pass = False
        record.failure_reason = f"Failed:euler_rh={record.euler_rh}"
        return record

    # ── Check key morphometric files ──
    for h in HEMIS:
        for feat in MORPHO_FEATURES:
            if not (sd / "surf" / f"{h}.{feat}").exists():
                record.qc_pass = False
                record.failure_reason = f"Failed:missing_{h}.{feat}"
                return record

    record.qc_pass = True
    record.failure_reason = None
    return record


# ─────────────────────────────────────────────────────────
# Generic Contrast Registration (FLAIR, T2, SWI → T1)
# ─────────────────────────────────────────────────────────

def register_contrast_to_t1(
    record: SubjectRecord,
    modality: str,
    config: dict,
) -> bool:
    """
    Rigid-body registration of any contrast volume to T1 space.
    Works for FLAIR, T2, SWI — same registration approach.

    Produces:
      mri/<modality>_orig.mgz       — converted input
      mri/<modality>_to_t1.lta      — transformation
      mri/<modality>_in_t1.mgz      — registered volume in T1 space
    """
    nii_path = record.contrast_niis.get(modality)
    if not nii_path or not record.qc_pass:
        return False

    sd = pathlib.Path(record.subjects_dir_path)
    src = pathlib.Path(nii_path)

    if not src.exists():
        logging.warning(f"  {modality.upper()} not found: {src}")
        return False

    # T1 reference in FreeSurfer space
    t1_mgz = sd / "mri" / "orig.mgz"
    if not t1_mgz.exists():
        t1_mgz = sd / "mri" / "T1.mgz"
    if not t1_mgz.exists():
        logging.warning(f"  T1 mgz not found for {record.subject_id}")
        return False

    reg_mgz = sd / "mri" / f"{modality}_in_t1.mgz"
    lta_file = sd / "mri" / f"{modality}_to_t1.lta"

    if config["skip_existing"] and reg_mgz.exists():
        record.registered[modality] = True
        return True

    if config["dry_run"]:
        logging.info(f"  [DRY] Would register {modality.upper()} -> T1 for {record.subject_id}")
        record.registered[modality] = True
        return True

    # ── Convert NIfTI → mgz ──
    orig_mgz = sd / "mri" / f"{modality}_orig.mgz"
    if not orig_mgz.exists():
        proc = subprocess.run(
            [fs_cmd("mri_convert"), str(src), str(orig_mgz)],
            capture_output=True, timeout=120,
        )
        if proc.returncode != 0:
            logging.warning(f"  mri_convert failed for {modality.upper()}: {record.subject_id}")
            return False

    # ── Rigid registration ──
    cmd = [
        fs_cmd("mri_robust_register"),
        "--mov", str(orig_mgz),
        "--dst", str(t1_mgz),
        "--lta", str(lta_file),
        "--mapmov", str(reg_mgz),
        "--satit",
        "--iscale",
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if proc.returncode == 0 and reg_mgz.exists():
            record.registered[modality] = True
            logging.info(f"  {modality.upper()} registered: {record.subject_id}")
            return True
        else:
            logging.warning(
                f"  {modality.upper()} registration failed: {record.subject_id} | "
                f"{proc.stderr[-200:] if proc.stderr else 'no stderr'}"
            )
    except subprocess.TimeoutExpired:
        logging.warning(f"  {modality.upper()} registration timeout: {record.subject_id}")
    except FileNotFoundError:
        logging.warning(
            f"  mri_robust_register not found — is FreeSurfer in PATH? "
            f"Skipping {modality.upper()} for {record.subject_id}"
        )

    return False


# ─────────────────────────────────────────────────────────
# Generic Surface Sampling (any registered volume)
# ─────────────────────────────────────────────────────────

def sample_contrast_to_surface(
    record: SubjectRecord,
    modality: str,
    config: dict,
) -> bool:
    """
    Sample a registered contrast volume onto the cortical surface at
    multiple depths using mri_vol2surf.

    Produces per hemisphere per depth:
      surf/{hemi}.{modality}_depth{depth_str}.mgh
    """
    if not record.registered.get(modality) or not record.qc_pass:
        return False

    sd = pathlib.Path(record.subjects_dir_path)
    reg_mgz = sd / "mri" / f"{modality}_in_t1.mgz"

    if not reg_mgz.exists():
        return False

    success = True
    for hemi in HEMIS:
        for depth in SAMPLING_DEPTHS:
            depth_str = _depth_to_str(depth)
            out_file = sd / "surf" / f"{hemi}.{modality}_depth{depth_str}.mgh"

            if config["skip_existing"] and out_file.exists():
                continue

            if config["dry_run"]:
                continue

            cmd = [
                fs_cmd("mri_vol2surf"),
                "--mov", str(reg_mgz),
                "--hemi", hemi,
                "--surf", "white",
                "--out", str(out_file),
                "--projfrac", str(depth),
                "--regheader", record.subject_id,
                "--sd", str(pathlib.Path(record.subjects_dir_path).parent),
                "--interp", "trilinear",
            ]

            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if proc.returncode != 0:
                    logging.warning(
                        f"  vol2surf failed: {record.subject_id} {modality} {hemi} d={depth}"
                    )
                    success = False
            except Exception as e:
                logging.warning(
                    f"  vol2surf error: {record.subject_id} {modality} {hemi} d={depth}: {e}"
                )
                success = False

    return success


def _depth_to_str(depth: float) -> str:
    """Convert depth float to filename-safe string: -0.50 -> 'm050', +0.25 -> 'p025'."""
    return f"{depth:+.2f}".replace("+", "p").replace("-", "m").replace(".", "")


# ─────────────────────────────────────────────────────────
# Gray-White Contrast (pctsurfcon)
# ─────────────────────────────────────────────────────────

def compute_pctsurfcon(record: SubjectRecord, config: dict) -> bool:
    """
    Compute percent contrast between gray and white matter surfaces.
    Produces lh.w-g.pct.mgh and rh.w-g.pct.mgh.

    Reimplemented in Python because the original pctsurfcon is a tcsh script
    that can't find FreeSurfer binaries when called from a conda environment.

    Equivalent to:
      pct = 100 * (W - G) / (0.5 * (W + G))
    where W = sampled 1mm into WM, G = sampled 30% into cortex.
    """
    if not record.qc_pass:
        return False

    sd = pathlib.Path(record.subjects_dir_path)
    subjects_parent = sd.parent

    # Check if both outputs already exist
    lh_out = sd / "surf" / "lh.w-g.pct.mgh"
    rh_out = sd / "surf" / "rh.w-g.pct.mgh"

    if config["skip_existing"] and lh_out.exists() and rh_out.exists():
        return True

    if config["dry_run"]:
        return True

    # Use rawavg.mgz (same as pctsurfcon default), fall back to orig.mgz
    vol = sd / "mri" / "rawavg.mgz"
    if not vol.exists():
        vol = sd / "mri" / "orig.mgz"
    if not vol.exists():
        logging.warning(f"  pctsurfcon: no rawavg/orig.mgz for {record.subject_id}")
        return False

    success = True
    for hemi in HEMIS:
        out_file = sd / "surf" / f"{hemi}.w-g.pct.mgh"
        tmp_dir = sd / "surf" / f"tmp.pctsurfcon.{os.getpid()}"
        tmp_dir.mkdir(exist_ok=True)

        wm_file = tmp_dir / f"{hemi}.wm.mgh"
        gm_file = tmp_dir / f"{hemi}.gm.mgh"

        try:
            # Sample WM: 1mm below white surface
            cmd_wm = [
                fs_cmd("mri_vol2surf"),
                "--mov", str(vol),
                "--hemi", hemi,
                "--noreshape",
                "--interp", "trilinear",
                "--projdist", "-1",
                "--o", str(wm_file),
                "--regheader", record.subject_id,
                "--sd", str(subjects_parent),
            ]
            proc = subprocess.run(cmd_wm, capture_output=True, text=True, timeout=120)
            if proc.returncode != 0:
                logging.warning(f"  pctsurfcon WM sampling failed: {record.subject_id} {hemi}")
                success = False
                continue

            # Sample GM: 30% of cortical thickness into gray matter
            cmd_gm = [
                fs_cmd("mri_vol2surf"),
                "--mov", str(vol),
                "--hemi", hemi,
                "--noreshape",
                "--interp", "trilinear",
                "--projfrac", "0.3",
                "--o", str(gm_file),
                "--regheader", record.subject_id,
                "--sd", str(subjects_parent),
            ]
            proc = subprocess.run(cmd_gm, capture_output=True, text=True, timeout=120)
            if proc.returncode != 0:
                logging.warning(f"  pctsurfcon GM sampling failed: {record.subject_id} {hemi}")
                success = False
                continue

            # Load and compute pct = 100*(W-G)/(0.5*(W+G))
            import nibabel as nib
            wm_data = np.squeeze(nib.load(str(wm_file)).get_fdata()).astype(np.float32)
            gm_img = nib.load(str(gm_file))
            gm_data = np.squeeze(gm_img.get_fdata()).astype(np.float32)

            denom = 0.5 * (wm_data + gm_data)
            pct = np.where(denom > 0, 100.0 * (wm_data - gm_data) / denom, 0.0).astype(np.float32)

            # Save as MGH using the GM image as template
            pct_img = nib.MGHImage(pct.reshape(gm_img.shape), gm_img.affine, gm_img.header)
            nib.save(pct_img, str(out_file))
            logging.debug(f"  pctsurfcon OK: {record.subject_id} {hemi}")

        except Exception as e:
            logging.warning(f"  pctsurfcon error: {record.subject_id} {hemi}: {e}")
            success = False
        finally:
            # Clean up temp files
            for f in tmp_dir.glob("*"):
                f.unlink(missing_ok=True)
            tmp_dir.rmdir()

    return success


# ─────────────────────────────────────────────────────────
# Resample to fsaverage
# ─────────────────────────────────────────────────────────

def resample_to_fsaverage(
    subject_id: str,
    subjects_dir: str,
    hemi: str,
    src_file: str,
    tgt_file: str,
    dry_run: bool = False,
) -> bool:
    """Resample a surface overlay from subject space to fsaverage."""
    if dry_run:
        return True

    cmd = [
        fs_cmd("mri_surf2surf"),
        "--srcsubject", subject_id,
        "--trgsubject", "fsaverage",
        "--hemi", hemi,
        "--sval", src_file,
        "--tval", tgt_file,
        "--sd", subjects_dir,
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        return proc.returncode == 0
    except Exception as e:
        logging.debug(f"  mri_surf2surf error: {e}")
        return False


def resample_all_features(record: SubjectRecord, config: dict) -> bool:
    """
    Resample all morphometric + contrast features to fsaverage for one subject.
    """
    if not record.qc_pass:
        return False

    sd = pathlib.Path(record.subjects_dir_path)
    subjects_dir = str(sd.parent)
    modalities = config.get("modalities", ALL_MODALITIES)

    fsavg_dir = sd / "fsaverage_features"
    fsavg_dir.mkdir(exist_ok=True)

    success = True

    for hemi in HEMIS:
        # ── Morphometric features ──
        for feat in MORPHO_FEATURES:
            src = sd / "surf" / f"{hemi}.{feat}"
            tgt = fsavg_dir / f"{hemi}.{feat}.fsaverage.mgh"

            if config["skip_existing"] and tgt.exists():
                continue
            if not src.exists():
                logging.warning(f"  Missing {hemi}.{feat} for {record.subject_id}")
                success = False
                continue

            ok = resample_to_fsaverage(
                record.subject_id, subjects_dir, hemi,
                str(src), str(tgt), config["dry_run"],
            )
            if not ok:
                logging.warning(f"  Resample failed: {hemi}.{feat} for {record.subject_id}")
                success = False

        # ── Gray-white contrast ──
        pct_src = sd / "surf" / f"{hemi}.w-g.pct.mgh"
        pct_tgt = fsavg_dir / f"{hemi}.w-g.pct.fsaverage.mgh"
        if pct_src.exists() and not (config["skip_existing"] and pct_tgt.exists()):
            resample_to_fsaverage(
                record.subject_id, subjects_dir, hemi,
                str(pct_src), str(pct_tgt), config["dry_run"],
            )

        # ── All contrast modalities (FLAIR, T2, SWI) at each depth ──
        for modality in modalities:
            if not record.registered.get(modality):
                continue

            for depth in SAMPLING_DEPTHS:
                depth_str = _depth_to_str(depth)
                src = sd / "surf" / f"{hemi}.{modality}_depth{depth_str}.mgh"
                tgt = fsavg_dir / f"{hemi}.{modality}_depth{depth_str}.fsaverage.mgh"

                if not src.exists():
                    continue
                if config["skip_existing"] and tgt.exists():
                    continue

                resample_to_fsaverage(
                    record.subject_id, subjects_dir, hemi,
                    str(src), str(tgt), config["dry_run"],
                )

    return success


def resample_atlas_labels(record: SubjectRecord, config: dict) -> bool:
    """
    Resample DKT atlas annotation from subject native space to fsaverage.

    Uses mri_surf2surf --sval-annot which does nearest-neighbor resampling
    appropriate for discrete parcellation labels (not interpolation).

    Produces per hemisphere:
      fsaverage_features/{hemi}.aparc.DKTatlas.fsaverage.annot
    """
    if not record.qc_pass:
        return False

    sd = pathlib.Path(record.subjects_dir_path)
    subjects_dir = str(sd.parent)

    fsavg_dir = sd / "fsaverage_features"
    fsavg_dir.mkdir(exist_ok=True)

    success = True

    for hemi in HEMIS:
        # Find source annotation in subject's native space
        src_annot = None
        for name in [f"{hemi}.aparc.DKTatlas.mapped.annot", f"{hemi}.aparc.DKTatlas.annot"]:
            candidate = sd / "label" / name
            if candidate.exists():
                src_annot = candidate
                break

        if src_annot is None:
            logging.warning(f"  No DKT atlas annot found for {record.subject_id} {hemi}")
            success = False
            continue

        tgt_annot = fsavg_dir / f"{hemi}.aparc.DKTatlas.fsaverage.annot"

        if config["skip_existing"] and tgt_annot.exists():
            continue

        if config["dry_run"]:
            logging.info(f"  [DRY] Would resample atlas {hemi} for {record.subject_id}")
            continue

        cmd = [
            fs_cmd("mri_surf2surf"),
            "--srcsubject", record.subject_id,
            "--trgsubject", "fsaverage",
            "--hemi", hemi,
            "--sval-annot", str(src_annot),
            "--tval", str(tgt_annot),
            "--sd", subjects_dir,
        ]

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if proc.returncode == 0 and tgt_annot.exists():
                logging.debug(f"  Atlas resampled: {record.subject_id} {hemi}")
            else:
                logging.warning(
                    f"  Atlas resample failed: {record.subject_id} {hemi} | "
                    f"{proc.stderr[-200:] if proc.stderr else 'no stderr'}"
                )
                success = False
        except Exception as e:
            logging.warning(f"  Atlas resample error: {record.subject_id} {hemi}: {e}")
            success = False

    return success

def load_freesurfer_surface(surf_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a FreeSurfer surface file. Returns (vertices, faces)."""
    try:
        import nibabel.freesurfer as fs
        verts, faces = fs.read_geometry(surf_path)
        return verts, faces
    except ImportError:
        pass

    # Manual reader for FreeSurfer triangle surface format
    with open(surf_path, "rb") as f:
        magic = f.read(3)
        if magic != b"\xff\xff\xfe":
            raise ValueError(f"Not a FreeSurfer triangle surface: {surf_path}")
        line = b""
        while True:
            c = f.read(1)
            if c == b"\n":
                c2 = f.read(1)
                if c2 == b"\n":
                    break
                line += c + c2
            else:
                line += c

        import struct
        n_verts, n_faces = struct.unpack(">ii", f.read(8))
        verts = np.frombuffer(f.read(n_verts * 3 * 4), dtype=">f4").reshape(n_verts, 3).astype(np.float64)
        faces = np.frombuffer(f.read(n_faces * 3 * 4), dtype=">i4").reshape(n_faces, 3).astype(np.int64)

    return verts, faces


def compute_geodesic_distance_from_surface(
    verts: np.ndarray,
    faces: np.ndarray,
    source_vertices: Optional[np.ndarray] = None,
    max_distance: float = 100.0,
) -> np.ndarray:
    """
    Compute geodesic distances on a triangulated mesh.
    Uses gdist if available, falls back to sparse-graph Dijkstra.
    """
    try:
        import gdist
        if source_vertices is not None:
            return gdist.compute_gdist(
                verts.astype(np.float64),
                faces.astype(np.int32),
                source_indices=source_vertices.astype(np.int32),
                max_distance=max_distance,
            )
        else:
            return gdist.local_gdist_matrix(
                verts.astype(np.float64),
                faces.astype(np.int32),
                max_distance=max_distance,
            )
    except ImportError:
        pass

    # ── Fallback: graph-based Dijkstra using edge lengths ──
    logging.info("  gdist not available — using graph-based approximation")
    from scipy.sparse import lil_matrix
    from scipy.sparse.csgraph import shortest_path

    n = len(verts)
    adj = lil_matrix((n, n), dtype=np.float64)

    for f in faces:
        for i in range(3):
            vi, vj = f[i], f[(i + 1) % 3]
            d = np.linalg.norm(verts[vi] - verts[vj])
            adj[vi, vj] = d
            adj[vj, vi] = d

    adj = adj.tocsr()

    if source_vertices is not None:
        return shortest_path(adj, indices=source_vertices, directed=False, limit=max_distance)
    else:
        return adj


def compute_fsaverage_geodesic(config: dict) -> Optional[str]:
    """
    Compute geodesic distance matrix on the fsaverage mesh.
    Done ONCE (shared across all subjects). Saves sparse distance matrix per hemi.
    """
    out_dir = pathlib.Path(config["subjects_dir"]) / "fsaverage_common"
    out_dir.mkdir(exist_ok=True)

    fsaverage_dir = _find_fsaverage(config)
    if fsaverage_dir is None:
        logging.error("Cannot find fsaverage directory — skipping geodesic computation")
        return None

    for hemi in HEMIS:
        out_file = out_dir / f"{hemi}.geodesic_dist.npz"

        if config["skip_existing"] and out_file.exists():
            logging.info(f"  Geodesic distance already computed: {out_file}")
            continue

        if config["dry_run"]:
            logging.info(f"  [DRY] Would compute geodesic distance for {hemi}")
            continue

        surf_path = pathlib.Path(fsaverage_dir) / "surf" / f"{hemi}.white"
        if not surf_path.exists():
            surf_path = pathlib.Path(fsaverage_dir) / "surf" / f"{hemi}.inflated"
        if not surf_path.exists():
            logging.error(f"  fsaverage surface not found: {surf_path}")
            continue

        logging.info(f"  Computing geodesic distance on fsaverage {hemi} "
                     f"(this may take several minutes)...")
        t0 = time.time()

        verts, faces = load_freesurfer_surface(str(surf_path))
        logging.info(f"    Surface: {len(verts)} vertices, {len(faces)} faces")

        dist_matrix = compute_geodesic_distance_from_surface(
            verts, faces, max_distance=config.get("geodesic_max_dist", 100.0),
        )

        from scipy import sparse
        if sparse.issparse(dist_matrix):
            sparse.save_npz(str(out_file), dist_matrix.tocsr())
        else:
            np.savez_compressed(str(out_file), distances=dist_matrix)

        elapsed = time.time() - t0
        logging.info(f"    Done in {elapsed:.1f}s -> {out_file}")

    return str(out_dir)


# ─────────────────────────────────────────────────────────
# Vertex Positions on fsaverage
# ─────────────────────────────────────────────────────────

def compute_fsaverage_positions(config: dict) -> Optional[str]:
    """Extract vertex (x, y, z) from the fsaverage mesh. Shared across all subjects."""
    out_dir = pathlib.Path(config["subjects_dir"]) / "fsaverage_common"
    out_dir.mkdir(exist_ok=True)

    fsaverage_dir = _find_fsaverage(config)
    if fsaverage_dir is None:
        return None

    for hemi in HEMIS:
        out_file = out_dir / f"{hemi}.positions.npy"

        if config["skip_existing"] and out_file.exists():
            logging.info(f"  Positions already computed: {out_file}")
            continue

        if config["dry_run"]:
            logging.info(f"  [DRY] Would extract positions for {hemi}")
            continue

        surf_path = pathlib.Path(fsaverage_dir) / "surf" / f"{hemi}.white"
        if not surf_path.exists():
            logging.error(f"  fsaverage surface not found: {surf_path}")
            continue

        verts, _ = load_freesurfer_surface(str(surf_path))
        np.save(str(out_file), verts.astype(np.float32))
        logging.info(f"  Positions saved: {out_file} — shape {verts.shape}")

    return str(out_dir)


def _find_fsaverage(config: dict) -> Optional[str]:
    """Locate the fsaverage subject directory."""
    sd = pathlib.Path(config["subjects_dir"])
    candidates = [
        sd / "fsaverage",
        pathlib.Path(FREESURFER_HOME) / "subjects" / "fsaverage",
        pathlib.Path(os.environ.get("FREESURFER_HOME", "")) / "subjects" / "fsaverage",
        pathlib.Path(os.environ.get("SUBJECTS_DIR", "")) / "fsaverage",
        pathlib.Path("/usr/local/freesurfer/subjects/fsaverage"),
    ]
    for c in candidates:
        if c.exists() and (c / "surf").exists():
            return str(c)

    logging.error(
        "fsaverage not found. Ensure FREESURFER_HOME is set or "
        "fsaverage exists in your subjects directory."
    )
    return None


# ─────────────────────────────────────────────────────────
# Feature Stacking -> vertex_features.npz
# ─────────────────────────────────────────────────────────

def load_surface_data(filepath: str) -> Optional[np.ndarray]:
    """Load a FreeSurfer surface overlay (.mgh, .mgz, or curv-format)."""
    fp = pathlib.Path(filepath)

    if fp.suffix in (".mgh", ".mgz"):
        try:
            import nibabel as nib
            img = nib.load(str(fp))
            data = np.squeeze(img.get_fdata())
            return data.astype(np.float32)
        except ImportError:
            tmp = fp.parent / f"_tmp_{fp.stem}.npy"
            proc = subprocess.run(
                [fs_cmd("mri_convert"), str(fp), str(tmp).replace(".npy", ".txt"),
                 "--ascii"],
                capture_output=True, timeout=60,
            )
            txt_file = str(tmp).replace(".npy", ".txt")
            if pathlib.Path(txt_file).exists():
                data = np.loadtxt(txt_file)
                pathlib.Path(txt_file).unlink()
                return data.astype(np.float32)
            return None
    else:
        try:
            import nibabel.freesurfer as fs
            data = fs.read_morph_data(str(fp))
            return data.astype(np.float32)
        except ImportError:
            tmp_asc = fp.parent / f"_tmp_{fp.name}.asc"
            proc = subprocess.run(
                [fs_cmd("mris_convert"), "-c", str(fp), str(_find_any_surf(fp.parent)), str(tmp_asc)],
                capture_output=True, timeout=60,
            )
            if tmp_asc.exists():
                data = np.loadtxt(str(tmp_asc), usecols=4)
                tmp_asc.unlink()
                return data.astype(np.float32)
            return None


def _find_any_surf(surf_dir: pathlib.Path) -> pathlib.Path:
    for name in ["lh.white", "rh.white", "lh.pial", "rh.pial"]:
        p = surf_dir / name
        if p.exists():
            return p
    return surf_dir / "lh.white"


def stack_features_for_subject(record: SubjectRecord, config: dict) -> Optional[dict]:
    """
    Stack all resampled features into vertex_features.npz for one subject.

    Output shape per hemisphere: (N_FSAVERAGE_VERTICES, n_features)

    Feature order (example with all 3 modalities available):
      0:  thickness
      1:  curvature
      2:  sulcal depth
      3:  area
      4:  gray-white contrast
      5-11:  FLAIR at 7 depths
      12-18: T2 at 7 depths
      19-25: SWI at 7 depths
    """
    if not record.qc_pass:
        return None

    sd = pathlib.Path(record.subjects_dir_path)
    fsavg_dir = sd / "fsaverage_features"
    modalities = config.get("modalities", ALL_MODALITIES)

    # ── Morphometrics + gray-white contrast → own file ──
    for hemi in HEMIS:
        morpho_arrays, morpho_names = [], []
        for feat in MORPHO_FEATURES:
            fpath = fsavg_dir / f"{hemi}.{feat}.fsaverage.mgh"
            data = load_surface_data(str(fpath)) if fpath.exists() else None
            arr = data if (data is not None and len(data) == N_FSAVERAGE_VERTICES) \
                else np.zeros(N_FSAVERAGE_VERTICES, dtype=np.float32)
            morpho_arrays.append(arr)
            morpho_names.append(feat)

        pct_path = fsavg_dir / f"{hemi}.w-g.pct.fsaverage.mgh"
        if pct_path.exists():
            data = load_surface_data(str(pct_path))
            if data is not None and len(data) == N_FSAVERAGE_VERTICES:
                morpho_arrays.append(data)
                morpho_names.append("wg_pct")

        out = sd / f"{hemi}.morpho.npz"
        np.savez_compressed(str(out),
                            data=np.column_stack(morpho_arrays),
                            feature_names=np.array(morpho_names))

    # ── Each contrast modality → own file ──
    for modality in modalities:
        if not record.registered.get(modality):
            continue
        for hemi in HEMIS:
            depth_arrays, depth_names = [], []
            for depth in SAMPLING_DEPTHS:
                depth_str = _depth_to_str(depth)
                fpath = fsavg_dir / f"{hemi}.{modality}_depth{depth_str}.fsaverage.mgh"
                if fpath.exists():
                    data = load_surface_data(str(fpath))
                    if data is not None and len(data) == N_FSAVERAGE_VERTICES:
                        depth_arrays.append(data)
                        depth_names.append(f"{modality}_d{depth:+.2f}")

            if depth_arrays:
                out = sd / f"{hemi}.{modality}.npz"
                np.savez_compressed(str(out),
                                    data=np.column_stack(depth_arrays),
                                    feature_names=np.array(depth_names))
                logging.info(f"  {record.subject_id}: saved {out.name} "
                             f"({N_FSAVERAGE_VERTICES} x {len(depth_arrays)})")

    record.features_extracted = True
    return {"subject_id": record.subject_id, "status": "success"}


# ─────────────────────────────────────────────────────────
# Dataset Split
# ─────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────
# Process One Subject (all steps)
# ─────────────────────────────────────────────────────────

def process_subject(record: SubjectRecord, config: dict) -> SubjectRecord:
    """
    Full per-subject pipeline:
      register contrasts -> sample to surface -> pctsurfcon -> resample -> atlas -> stack
    """
    steps = config.get("steps", ["contrasts", "pctsurfcon", "resample", "stack"])
    modalities = config.get("modalities", ALL_MODALITIES)

    if not record.qc_pass:
        return record

    # ── Register + sample all contrast modalities ──
    if "contrasts" in steps:
        for modality in modalities:
            if modality in record.contrast_niis:
                ok = register_contrast_to_t1(record, modality, config)
                if ok:
                    sample_contrast_to_surface(record, modality, config)

    # ── Gray-white contrast ──
    if "pctsurfcon" in steps:
        compute_pctsurfcon(record, config)

    # ── Resample to fsaverage ──
    if "resample" in steps:
        resample_all_features(record, config)
        resample_atlas_labels(record, config)

    # ── Stack features ──
    if "stack" in steps:
        stack_features_for_subject(record, config)

    record.features_extracted = True
    return record


# ─────────────────────────────────────────────────────────
# Build Records (helpers)
# ─────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────
# Update data_dict with failure info
# ─────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()