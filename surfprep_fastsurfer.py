"""
surfprep_fastsurfer.py — FastSurfer reconstruction engine
==================================================================

Reusable FastSurfer seg/surf job runner used by run_surfprep_step_1.py.
selects the best T1 per subject, and runs FastSurfer in two phases:
  Phase 1: GPU segmentation (sequential, ~1 min each)
  Phase 2: CPU surface reconstruction (parallel, ~55 min each)

T1 Selection (when multiple T1s per subject):
  Prefers 3D acquisitions (MPRAGE, SPACE, VIBE) over 2D.
  You can override with --t1_selection interactive to pick manually,
  or --t1_selection all to process every T1.

Usage:
  # Dry run — see which T1 would be selected per subject
  python SurfPrep --dry_run

  # Run with best T1 auto-selected
  python SurfPrep

  # Full pipeline with 4 parallel recon-surf workers
  python SurfPrep --mode full --cpu_workers 4

  # Seg only (fast, ~1 min per subject)
  python SurfPrep --mode seg_only

  # Surface reconstruction only (after seg done on GPU)
  python SurfPrep --mode surf_only --cpu_workers 8

  # Process only first 5 subjects (testing)
  python SurfPrep --limit 5

  # Process all T1 sequences (not just best)
  python SurfPrep --t1_selection all
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
from dataclasses import dataclass
from typing import List, Optional


# ─────────────────────────────────────────────────────────
#  T1 Selection
# ─────────────────────────────────────────────────────────

def score_t1(par_name: str) -> int:
    """
    Score a T1 sequence by how suitable it is for FastSurfer surface reconstruction.

    FastSurfer needs: 3D T1w, ideally isotropic ~1mm, MPRAGE or similar.
    Higher score = better candidate.
    """
    name = par_name.lower()
    score = 0

    # ── Strongly prefer 3D acquisitions ──
    if "3d" in name or "3 d" in name:
        score += 20
    if "mprage" in name:
        score += 15
    if "space" in name:  # 3D SPACE T1
        score += 12
    if "bravo" in name:  # GE's MPRAGE equivalent
        score += 12
    if "spgr" in name:   # Spoiled gradient echo 3D
        score += 10

    # ── Prefer sagittal (usually 3D isotropic) ──
    if "sag" in name:
        score += 5

    # ── Isotropic indicators ──
    if "iso" in name:
        score += 5
    if "1mm" in name or "1.0mm" in name or "0.9mm" in name:
        score += 3

    # ── Penalize 2D / thick slice sequences ──
    if "2d" in name:
        score -= 10
    if "blade" in name:  # Motion-corrected 2D, usually thick slices
        score -= 5
    if "tra" in name and "3d" not in name:  # Axial 2D
        score -= 5
    if "cor" in name and "3d" not in name:  # Coronal 2D
        score -= 3
    if "tse" in name or "se_" in name:  # Spin echo (usually 2D)
        score -= 5
    if "scout" in name or "localizer" in name:
        score -= 100
    if "survey" in name:
        score -= 100

    # ── Penalize post-contrast (km_t1 should already be filtered, but just in case) ──
    if "km" in name or "gad" in name or "contrast" in name:
        score -= 50

    return score


def select_best_t1(t1_entries: list) -> Optional[dict]:
    """
    Select the best T1 from a list of entries.
    Each entry: {'par_name': str, 'nii_name': str}
    """
    if not t1_entries:
        return None
    if len(t1_entries) == 1:
        return t1_entries[0]

    scored = [(score_t1(e["par_name"]), e) for e in t1_entries]
    scored.sort(key=lambda x: x[0], reverse=True)

    # Log the ranking
    for rank, (s, e) in enumerate(scored):
        marker = " ← SELECTED" if rank == 0 else ""
        logging.debug(f"    T1 [{s:+3d}] {e['par_name']}/{e['nii_name']}{marker}")

    return scored[0][1]


# ─────────────────────────────────────────────────────────
#  Job Preparation
# ─────────────────────────────────────────────────────────

@dataclass
class FastSurferJob:
    subject_id: str       # Used as --sid for FastSurfer
    study_uid: str        # Original study UID
    t1_path: str          # Full path to the T1 NIfTI
    par_name: str         # Sequence directory name (for logging)
    befund: Optional[str] # 'mit Befund' or 'ohne Befund'
    city: Optional[str]


def discover_subjects_from_output(subjects_dir: str) -> List[FastSurferJob]:
    """
    Discover subjects from existing FastSurfer segmentation output.
    Use this when running surf_only mode after seg was done elsewhere.
    """
    subjects_path = pathlib.Path(subjects_dir)
    jobs = []

    if not subjects_path.exists():
        logging.error(f"Subjects directory not found: {subjects_dir}")
        return jobs

    # Find all subject directories with segmentation output
    for subj_dir in sorted(subjects_path.iterdir()):
        if not subj_dir.is_dir():
            continue

        # Check if segmentation exists
        seg_file = subj_dir / "mri" / "aparc.DKTatlas+aseg.deep.mgz"
        if not seg_file.exists():
            logging.debug(f"Skipping {subj_dir.name} - no segmentation found")
            continue

        # Check if segmentation is valid (not corrupted/empty)
        if seg_file.stat().st_size < 1000:
            logging.warning(f"Skipping {subj_dir.name} - segmentation file too small (likely failed)")
            continue

        # Create job
        jobs.append(FastSurferJob(
            subject_id=subj_dir.name,
            study_uid=subj_dir.name,
            t1_path="",  # Not needed for surf_only
            par_name="",
            befund=None,
            city=None,
        ))

    logging.info(f"Discovered {len(jobs)} subjects with valid segmentation")
    return jobs


# ─────────────────────────────────────────────────────────
#  FastSurfer Execution
# ─────────────────────────────────────────────────────────

def run_fastsurfer_seg(job: FastSurferJob, config: dict) -> dict:
    """Run GPU segmentation for one subject."""
    result = {
        "subject_id": job.subject_id,
        "study_uid": job.study_uid,
        "par_name": job.par_name,
        "phase": "seg",
        "status": "unknown",
        "duration_sec": 0,
    }

    # Check existing
    seg_file = pathlib.Path(config["subjects_dir"]) / job.subject_id / "mri" / "aparc.DKTatlas+aseg.deep.mgz"
    if config["skip_existing"] and seg_file.exists():
        result["status"] = "skipped"
        return result

    # Check input exists
    if not pathlib.Path(job.t1_path).exists():
        result["status"] = "input_missing"
        logging.error(f"  T1 not found: {job.t1_path}")
        return result

    cmd = [
        config["fastsurfer_cmd"],
        "--sid", job.subject_id,
        "--sd", config["subjects_dir"],
        "--t1", job.t1_path,
        "--seg_only",
        "--device", config["device"],
    ]

    if config["dry_run"]:
        result["status"] = "dry_run"
        result["cmd"] = " ".join(cmd)
        logging.info(f"  [DRY] {job.subject_id}: {' '.join(cmd)}")
        return result

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        result["duration_sec"] = time.time() - t0
        if proc.returncode == 0:
            result["status"] = "success"
        else:
            result["status"] = "failed"
            result["error"] = proc.stderr[-500:] if proc.stderr else ""
            logging.error(f"  SEG FAIL: {job.subject_id} | {result['error'][:200]}")
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["duration_sec"] = time.time() - t0
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["duration_sec"] = time.time() - t0

    return result


def run_fastsurfer_surf(job: FastSurferJob, config: dict) -> dict:
    """Run CPU surface reconstruction for one subject via the FastSurfer wrapper."""
    result = {
        "subject_id": job.subject_id,
        "phase": "surf",
        "status": "unknown",
        "duration_sec": 0,
    }

    # Validate segmentation exists and is complete
    seg_file = pathlib.Path(config["subjects_dir"]) / job.subject_id / "mri" / "aparc.DKTatlas+aseg.deep.mgz"
    if not seg_file.exists():
        result["status"] = "seg_missing"
        logging.error(f"  SEG MISSING: {job.subject_id} - segmentation file not found")
        return result

    # Check if segmentation file is valid (not empty/corrupted)
    if seg_file.stat().st_size < 1000:  # Sanity check: should be several MB
        result["status"] = "seg_invalid"
        logging.error(f"  SEG INVALID: {job.subject_id} - segmentation file too small")
        return result

    thickness_file = pathlib.Path(config["subjects_dir"]) / job.subject_id / "surf" / "lh.thickness"
    if config["skip_existing"] and thickness_file.exists():
        result["status"] = "skipped"
        return result

    cmd = [
        config["fastsurfer_cmd"],
        "--sid", job.subject_id,
        "--sd", config["subjects_dir"],
        "--surf_only",
        "--fs_license", "/fs_license/license.txt",
        "--threads", str(config["threads_per_worker"]),
        "--parallel",
    ]

    if config["dry_run"]:
        result["status"] = "dry_run"
        return result

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
        result["duration_sec"] = time.time() - t0
        if proc.returncode == 0:
            result["status"] = "success"
        else:
            result["status"] = "failed"
            result["error"] = proc.stderr[-500:] if proc.stderr else ""
            logging.error(f"  SURF FAIL: {job.subject_id} | {result['error'][:200]}")
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["duration_sec"] = time.time() - t0
    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        result["duration_sec"] = time.time() - t0

    return result


# ─────────────────────────────────────────────────────────
#  Batch Orchestration
# ─────────────────────────────────────────────────────────

def run_batch(jobs: List[FastSurferJob], config: dict):
    mode = config["mode"]
    n = len(jobs)

    # ── Phase 1: GPU Segmentation (always sequential — single GPU) ──
    logging.info("=" * 60)
    logging.info(f"PHASE 1: GPU Segmentation — {n} subjects")
    logging.info("=" * 60)

    seg_results = []
    seg_ok_jobs = []

    for i, job in enumerate(jobs):
        logging.info(f"[{i+1}/{n}] {job.subject_id} | {job.par_name}")
        result = run_fastsurfer_seg(job, config)
        seg_results.append(result)

        if result["status"] in ("success", "skipped"):
            seg_ok_jobs.append(job)
            if result["status"] == "success":
                logging.info(f"  OK ({result['duration_sec']:.0f}s)")
        elif result["status"] == "dry_run":
            seg_ok_jobs.append(job)

    n_ok = len(seg_ok_jobs)
    logging.info(f"Segmentation: {n_ok}/{n} ready for surface reconstruction")

    if mode == "seg_only":
        return seg_results

    # ── Phase 2: CPU Surface Reconstruction ──
    cpu_workers = config["cpu_workers"]
    logging.info("=" * 60)
    logging.info(f"PHASE 2: Surface Reconstruction — {n_ok} subjects, {cpu_workers} workers")
    logging.info("=" * 60)

    surf_results = []

    if cpu_workers <= 1:
        for i, job in enumerate(seg_ok_jobs):
            logging.info(f"[{i+1}/{n_ok}] recon-surf: {job.subject_id}")
            result = run_fastsurfer_surf(job, config)
            surf_results.append(result)
            if result["status"] == "success":
                logging.info(f"  OK ({result['duration_sec']:.0f}s)")
    else:
        # Parallel surface reconstruction
        with ProcessPoolExecutor(max_workers=cpu_workers) as executor:
            future_map = {
                executor.submit(run_fastsurfer_surf, job, config): job
                for job in seg_ok_jobs
            }
            done_count = 0
            for future in as_completed(future_map):
                job = future_map[future]
                done_count += 1
                try:
                    result = future.result()
                    surf_results.append(result)
                    logging.info(
                        f"[{done_count}/{n_ok}] {result['status']}: "
                        f"{job.subject_id} ({result['duration_sec']:.0f}s)"
                    )
                except Exception as e:
                    logging.error(f"[{done_count}/{n_ok}] WORKER ERROR: {job.subject_id} | {e}")
                    surf_results.append({
                        "subject_id": job.subject_id,
                        "phase": "surf",
                        "status": "error",
                        "error": str(e),
                    })

    return seg_results + surf_results


# ─────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()

    # Register FLAIR → T1 space (rigid coregistration per subject)
    # Sample FLAIR onto surface at 7 cortical depths (-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0) using mri_vol2surf
    # Compute gray-white contrast from T1 using pctsurfcon
    # Extract morphometric features already produced by FastSurfer: thickness, curvature, sulcal depth, area
    # Register each subject's surface to fsaverage (FastSurfer does this — verify lh.sphere.reg exists)
    # Resample all features to fsaverage so every subject has the same 163,842 vertices per hemisphere
    # Stack into vertex_features.npz per subject: shape (163842, 12) per hemisphere
    # Compute Laplacian eigenvectors once on fsaverage mesh (shared across all subjects)
    # Quality control — exclude subjects with failed recon (check Euler number, visual spot-checks)
    # Split dataset — healthy controls for training, "mit Befund" held out for testing