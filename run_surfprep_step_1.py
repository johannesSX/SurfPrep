"""
run_surfprep_step_1.py — Run FastSurfer on external datasets (datasets)
=============================================================================

Thin wrapper around surfprep_fastsurfer.py that loads data from datasets
(FCDBONN, IXI, IDEAS) via datasets_adapter.

The only difference vs surfprep_fastsurfer.py:
  - Data loading: uses datasets_adapter.read_data_ext()
  - Path resolution: uses resolve_nii_path() for absolute paths
  - Subject IDs: prefixed with dataset keyword (e.g., fcdbonn__sub-0042)
  - No T1 scoring needed — external datasets typically have one T1

Everything else (FastSurfer seg, surf, batch orchestration) is reused
directly from surfprep_fastsurfer.py.

Usage:
  # Dry run — see what would be processed
  python run_surfprep_step_1.py --dry_run

  # Process only FCDBONN dataset
  python run_surfprep_step_1.py --datasets FCDBONN

  # Process FCDBONN + IXI, segmentation only
  python run_surfprep_step_1.py --datasets FCDBONN IXI --mode seg_only

  # Full pipeline
  python run_surfprep_step_1.py --datasets FCDBONN --mode full --cpu_workers 4

  # Surface only (after seg done)
  python run_surfprep_step_1.py --mode surf_only --cpu_workers 8

  # First 5 subjects
  python run_surfprep_step_1.py --datasets FCDBONN --limit 5 --dry_run
"""

import argparse
import json
import logging
import os
import pathlib
import sys
import time
from typing import List, Optional

# Reuse the FastSurfer engine
from surfprep_fastsurfer import (
    FastSurferJob,
    run_fastsurfer_seg,
    run_fastsurfer_surf,
    run_batch,
    discover_subjects_from_output,
    score_t1,
    select_best_t1,
)

from datasets_adapter import (
    read_data_ext,
    resolve_nii_path,
    DATASET_REGISTRY,
)


# ─────────────────────────────────────────────────────────
# Job Preparation (adapted for datasets format)
# ─────────────────────────────────────────────────────────

def prepare_jobs_ext(
    lst_dicts: list,
    t1_selection: str = "best",
) -> List[FastSurferJob]:
    """
    Create FastSurfer jobs from datasets_adapter output.

    Paths are resolved
    via resolve_nii_path() since external datasets use absolute paths.
    """
    jobs = []
    skipped_no_t1 = 0

    for data_dict in lst_dicts:
        study_uid = data_dict["study_uid"]
        t1_entries = data_dict.get("t1", [])

        if not t1_entries:
            skipped_no_t1 += 1
            continue

        # Select T1
        if t1_selection == "best" and len(t1_entries) > 1:
            # Score based on par_name (directory name)
            selected = [select_best_t1(t1_entries)]
        elif t1_selection == "all":
            selected = t1_entries
        else:
            selected = [t1_entries[0]]

        for idx, entry in enumerate(selected):
            if entry is None:
                continue

            # Resolve full path using adapter
            t1_path = resolve_nii_path(entry, study_uid)

            # Subject ID
            if t1_selection == "all" and len(selected) > 1:
                sid = f"{study_uid}__{entry.get('par_name', idx)}"
            else:
                sid = study_uid

            jobs.append(FastSurferJob(
                subject_id=sid,
                study_uid=study_uid,
                t1_path=t1_path,
                par_name=entry.get("par_name", ""),
                befund=data_dict.get("study_befund"),
                city=data_dict.get("city"),
            ))

    if skipped_no_t1 > 0:
        logging.warning(f"Skipped {skipped_no_t1} subjects with no T1 sequences")

    return jobs


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run FastSurfer on external datasets (datasets)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available datasets: {', '.join(DATASET_REGISTRY.keys())}

Examples:
  python run_surfprep_step_1.py --dry_run                              # See what would run
  python run_surfprep_step_1.py --datasets FCDBONN --mode seg_only     # Seg only, FCDBONN
  python run_surfprep_step_1.py --datasets FCDBONN IXI --mode full # Full, two datasets
  python run_surfprep_step_1.py --mode surf_only --cpu_workers 8       # Surf only (reuse seg)
  python run_surfprep_step_1.py --datasets FCDBONN --limit 5           # Test on 5 subjects
        """,
    )

    # Dataset selection
    parser.add_argument("--datasets", nargs="+", default=None,
                        help=f"Which datasets to process (default: all). "
                             f"Options: {', '.join(DATASET_REGISTRY.keys())}")

    # Output directory
    parser.add_argument("--subjects_dir", type=str,
                        default=os.environ.get("SURFPREP_SUBJECTS_DIR", "data/fastsurfer_subjects"),
                        help="FastSurfer output directory (SUBJECTS_DIR)")

    # FastSurfer
    parser.add_argument("--fastsurfer_dir", type=str,
                        default=os.environ.get("FASTSURFER_HOME", "fastsurfer"),
                        help="FastSurfer installation directory")
    parser.add_argument("--fs_license", type=str,
                        default=os.environ.get("FS_LICENSE", os.path.expanduser("~/freesurfer/license.txt")),
                        help="FreeSurfer license file")

    # Processing
    parser.add_argument("--mode", type=str, default="full",
                        choices=["full", "seg_only", "surf_only"],
                        help="full = seg + surfaces, seg_only, surf_only")
    parser.add_argument("--device", type=str, default="cuda",
                        choices=["cuda", "cpu"])
    parser.add_argument("--cpu_workers", type=int, default=4,
                        help="Parallel CPU workers for recon-surf")
    parser.add_argument("--threads_per_worker", type=int, default=4,
                        help="CPU threads per recon-surf worker")

    # T1 Selection
    parser.add_argument("--t1_selection", type=str, default="first",
                        choices=["best", "all", "first"],
                        help="How to handle multiple T1s (default: first, "
                             "since external datasets usually have one)")

    # Options
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip", action="store_false", dest="skip_existing")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show what would be processed without running")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only first N subjects")
    parser.add_argument("--filter_prefix", type=str, default=None,
                        help="Only process subjects whose ID starts with this prefix (e.g. ideas__)")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    # ── Logging ──
    log_level = logging.DEBUG if args.verbose else logging.INFO
    os.makedirs(args.subjects_dir, exist_ok=True)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                pathlib.Path(args.subjects_dir) / "fastsurfer_ext_batch.log",
                mode="a",
            ),
        ],
    )

    # ── Config dict ──
    config = {
        "subjects_dir": args.subjects_dir,
        "fastsurfer_cmd": str(pathlib.Path(args.fastsurfer_dir) / "run_fastsurfer.sh"),
        "fs_license": args.fs_license,
        "device": args.device,
        "mode": args.mode,
        "cpu_workers": args.cpu_workers,
        "threads_per_worker": args.threads_per_worker,
        "skip_existing": args.skip_existing,
        "dry_run": args.dry_run,
    }

    # ── Validate FastSurfer ──
    if args.mode != "surf_only":
        if not pathlib.Path(config["fastsurfer_cmd"]).exists():
            logging.error(f"FastSurfer not found at {config['fastsurfer_cmd']}")
            logging.error("Set --fastsurfer_dir to your FastSurfer installation.")
            sys.exit(1)

    if args.mode in ("full", "surf_only") and not pathlib.Path(args.fs_license).exists():
        logging.error(f"FreeSurfer license not found at {args.fs_license}")
        sys.exit(1)

    # ── Load Data ──
    if args.mode == "surf_only":
        logging.info("Mode: surf_only — discovering from existing segmentation...")
        jobs = discover_subjects_from_output(args.subjects_dir)

        if args.filter_prefix:
            jobs = [j for j in jobs if j.subject_id.startswith(args.filter_prefix)]
            logging.info(f"Filtered to {len(jobs)} subjects with prefix '{args.filter_prefix}'")

        if args.limit:
            jobs = jobs[:args.limit]

        if not jobs:
            logging.error("No subjects with valid segmentation found!")
            sys.exit(1)

        logging.info(f"Found {len(jobs)} subjects ready for surface reconstruction")

        # Time estimate
        surf_min = len(jobs) * 55.0 / max(config["cpu_workers"], 1)
        logging.info(f"Estimated time: ~{surf_min:.0f} min ({surf_min/60:.1f} hrs)")

        # Run surface reconstruction
        from concurrent.futures import ProcessPoolExecutor, as_completed

        results = []
        n = len(jobs)
        logging.info("=" * 60)
        logging.info(f"Surface Reconstruction — {n} subjects, {config['cpu_workers']} workers")
        logging.info("=" * 60)

        t_start = time.time()

        if config["cpu_workers"] <= 1:
            for i, job in enumerate(jobs):
                logging.info(f"[{i+1}/{n}] recon-surf: {job.subject_id}")
                result = run_fastsurfer_surf(job, config)
                results.append(result)
                if result["status"] == "success":
                    logging.info(f"  OK ({result['duration_sec']:.0f}s)")
        else:
            with ProcessPoolExecutor(max_workers=config["cpu_workers"]) as executor:
                future_map = {
                    executor.submit(run_fastsurfer_surf, job, config): job
                    for job in jobs
                }
                done_count = 0
                for future in as_completed(future_map):
                    job = future_map[future]
                    done_count += 1
                    try:
                        result = future.result()
                        results.append(result)
                        logging.info(
                            f"[{done_count}/{n}] {result['status']}: "
                            f"{job.subject_id} ({result.get('duration_sec', 0):.0f}s)"
                        )
                    except Exception as e:
                        logging.error(f"[{done_count}/{n}] ERROR: {job.subject_id} | {e}")
                        results.append({
                            "subject_id": job.subject_id,
                            "phase": "surf",
                            "status": "error",
                            "error": str(e),
                        })

        total_sec = time.time() - t_start

    else:
        # ── Load external datasets via adapter ──
        logging.info("Loading external datasets via datasets_adapter...")
        t0 = time.time()
        lst_dicts = read_data_ext(datasets=args.datasets)
        logging.info(f"Loaded {len(lst_dicts)} subjects in {time.time()-t0:.1f}s")

        # ── Summary ──
        from collections import Counter
        kw_counts = Counter(d['keyword'] for d in lst_dicts)
        for kw, count in sorted(kw_counts.items()):
            n_flair = sum(1 for d in lst_dicts if d['keyword'] == kw and d['flair'])
            n_seg = sum(1 for d in lst_dicts if d['keyword'] == kw and d['seg_masks'])
            logging.info(f"  {kw:12s}: {count:4d} subjects (FLAIR: {n_flair}, seg: {n_seg})")

        n_with_t1 = sum(1 for d in lst_dicts if d.get("t1"))
        n_with_flair = sum(1 for d in lst_dicts if d.get("flair"))
        logging.info(f"  Subjects with T1:    {n_with_t1}")
        logging.info(f"  Subjects with FLAIR: {n_with_flair}")

        # ── Prepare Jobs ──
        jobs = prepare_jobs_ext(lst_dicts, args.t1_selection)

        if args.filter_prefix:
            jobs = [j for j in jobs if j.subject_id.startswith(args.filter_prefix)]
            logging.info(f"Filtered to {len(jobs)} subjects with prefix '{args.filter_prefix}'")

        if args.limit:
            jobs = jobs[:args.limit]

        logging.info(f"Prepared {len(jobs)} FastSurfer jobs")

        if not jobs:
            logging.warning("No jobs to run. Check that T1 sequences exist.")
            return

        # ── Validate input paths exist ──
        n_missing = 0
        for job in jobs:
            if not os.path.exists(job.t1_path):
                if args.verbose or n_missing < 5:
                    logging.warning(f"  T1 NOT FOUND: {job.subject_id}: {job.t1_path}")
                n_missing += 1
        if n_missing > 0:
            logging.warning(f"{n_missing} subjects have missing T1 files!")

        # ── Show sample jobs ──
        if args.verbose or args.dry_run:
            for job in jobs[:10]:
                exists = "OK" if os.path.exists(job.t1_path) else "MISSING"
                logging.info(f"  {job.subject_id}: {job.t1_path} [{exists}]")
            if len(jobs) > 10:
                logging.info(f"  ... and {len(jobs) - 10} more")

        # ── Time Estimate ──
        if config["mode"] == "seg_only":
            est_min = len(jobs) * 1.5
            logging.info(f"Estimated time: ~{est_min:.0f} min ({est_min/60:.1f} hrs)")
        else:
            seg_min = len(jobs) * 1.5
            surf_min = len(jobs) * 55.0 / max(args.cpu_workers, 1)
            total_min = seg_min + surf_min
            logging.info(
                f"Estimated time: ~{total_min:.0f} min ({total_min/60:.1f} hrs)"
                f" [seg: {seg_min:.0f}m + surf: {surf_min:.0f}m]"
            )

        # ── Run ──
        t_start = time.time()
        results = run_batch(jobs, config)
        total_sec = time.time() - t_start

    # ── Report ──
    status_counts = {}
    for r in results:
        s = r.get("status", "unknown")
        status_counts[s] = status_counts.get(s, 0) + 1

    logging.info("=" * 60)
    logging.info("DONE")
    logging.info(f"  Total time:  {total_sec/3600:.1f} hours")
    for status, count in sorted(status_counts.items()):
        logging.info(f"  {status}: {count}")
    logging.info("=" * 60)

    # ── Save results ──
    results_path = pathlib.Path(args.subjects_dir) / f"results_{args.mode}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logging.info(f"Results saved to {results_path}")

    # ── Save dataset mapping (subject_id → metadata) ──
    if args.mode != "surf_only":
        mapping = {}
        for d in lst_dicts:
            mapping[d["study_uid"]] = {
                "keyword": d["keyword"],
                "not_healthy": d["not_healthy"],
                "study_befund": d["study_befund"],
                "seg_masks": d["seg_masks"],
                "n_t1": len(d["t1"]),
                "n_flair": len(d["flair"]),
            }
        mapping_path = pathlib.Path(args.subjects_dir) / "dataset_mapping.json"
        with open(mapping_path, "w") as f:
            json.dump(mapping, f, indent=2)
        logging.info(f"Dataset mapping saved to {mapping_path}")

    # ── Print failed ──
    failed = [r for r in results
              if r.get("status") in ("failed", "error", "timeout",
                                     "seg_missing", "seg_invalid", "input_missing")]
    if failed:
        logging.warning(f"\n{len(failed)} FAILED:")
        for r in failed:
            logging.warning(f"  {r['subject_id']}: {r.get('status')} - {r.get('error', '')[:200]}")


if __name__ == "__main__":
    main()

