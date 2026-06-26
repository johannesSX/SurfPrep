"""
run_surfprep_step_2.py — Post-FastSurfer Feature Extraction (ext datasets)
=============================================================================

Adapted from surfprep_features.py for external datasets (datasets).

Changes from surfprep_features.py:
  - Data loading: uses datasets_adapter
  - Path resolution: uses resolve_nii_path() for absolute paths
  - Dataset split: uses not_healthy/seg_masks instead of annotations/befund
  - No --src_path needed (external datasets have absolute paths)
  - No normalization is applied — features are raw, same as hospital data

Pipeline Steps (identical to surfprep_features.py):
  1. QC — Check reconstruction quality (Euler number)
  2. Register contrasts (FLAIR) → T1 space
  3. Sample contrasts onto surface at 7 cortical depths
  4. Compute gray-white contrast (pctsurfcon)
  5. Resample all features to fsaverage (163,842 vertices)
  6. Resample DKT atlas annotation to fsaverage
  7. Compute geodesic distance + vertex positions on fsaverage
  8. Stack into .npz per subject
  9. Split dataset

Usage:
  python run_surfprep_step_2.py --datasets FCDBONN
  python run_surfprep_step_2.py --datasets FCDBONN --step qc
  python run_surfprep_step_2.py --datasets FCDBONN --step features --workers 16
  python run_surfprep_step_2.py --datasets FCDBONN --modalities flair
  python run_surfprep_step_2.py --datasets FCDBONN --dry_run --limit 5
"""

import argparse
import json
import logging
import os
import pathlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional

import numpy as np

# Reuse all processing functions from the feature engine
from surfprep_features import (
    # Constants
    HEMIS, N_FSAVERAGE_VERTICES, SAMPLING_DEPTHS, MORPHO_FEATURES,
    ALL_MODALITIES, EULER_THRESHOLD,
    # Functions
    fs_cmd,
    SubjectRecord,
    get_euler_number,
    run_qc,
    register_contrast_to_t1,
    sample_contrast_to_surface,
    compute_pctsurfcon,
    resample_to_fsaverage,
    resample_all_features,
    resample_atlas_labels,
    compute_fsaverage_geodesic,
    compute_fsaverage_positions,
    stack_features_for_subject,
    process_subject,
    select_best_contrast,
)

from datasets_adapter import (
    read_data_ext,
    resolve_nii_path,
    DATASET_REGISTRY,
)


# ─────────────────────────────────────────────────────────
# Build Records (adapted for ext data)
# ─────────────────────────────────────────────────────────

def build_records_ext(
    lst_dicts: list,
    config: dict,
    contrast_selection: str = "first",
) -> List[SubjectRecord]:
    """
    Build SubjectRecord objects from datasets_adapter output.

    Paths are resolved
    via resolve_nii_path() since external datasets use absolute paths.
    """
    modalities = config.get("modalities", ALL_MODALITIES)
    records = []

    for data_dict in lst_dicts:
        study_uid = data_dict["study_uid"]
        sd_path = pathlib.Path(config["subjects_dir"]) / study_uid

        # ── T1 path ──
        t1_entries = data_dict.get("t1", [])
        t1_path = None
        if t1_entries:
            t1_path = resolve_nii_path(t1_entries[0], study_uid)

        # ── Select contrast per modality ──
        contrast_niis = {}
        contrast_par_names = {}

        for modality in modalities:
            entries = data_dict.get(modality, [])
            if not entries:
                continue

            if contrast_selection == "best" and len(entries) > 1:
                best = select_best_contrast(entries, modality)
            else:
                best = entries[0]

            if best:
                nii_path = resolve_nii_path(best, study_uid)
                contrast_niis[modality] = nii_path
                contrast_par_names[modality] = best.get("par_name", "")

        # Determine if subject has annotations (seg masks for ext data)
        has_annotations = bool(data_dict.get("seg_masks"))

        records.append(SubjectRecord(
            subject_id=study_uid,
            study_uid=study_uid,
            subjects_dir_path=str(sd_path),
            t1_nii=t1_path,
            befund=data_dict.get("study_befund"),
            city=data_dict.get("city"),
            contrast_niis=contrast_niis,
            contrast_par_names=contrast_par_names,
            has_annotations=has_annotations,
        ))

    return records


# ─────────────────────────────────────────────────────────
# Update data_dicts with status (adapted)
# ─────────────────────────────────────────────────────────

def update_data_dicts_with_status_ext(
    lst_dicts: list,
    records: List[SubjectRecord],
    modalities: list,
) -> list:
    record_map = {r.study_uid: r for r in records}

    for d in lst_dicts:
        uid = d["study_uid"]
        r = record_map.get(uid)

        if r is None:
            d["fastsurfer_status"] = "not_processed"
            continue

        if r.qc_pass:
            d["fastsurfer_status"] = "OK"
        else:
            d["fastsurfer_status"] = r.failure_reason or "Failed"

        d["euler_lh"] = r.euler_lh
        d["euler_rh"] = r.euler_rh
        d["qc_pass"] = r.qc_pass
        d["features_extracted"] = r.features_extracted

        for modality in modalities:
            d[f"{modality}_registered"] = r.registered.get(modality, False)
            d[f"{modality}_selected_par"] = r.contrast_par_names.get(modality, None)

    return lst_dicts


# ─────────────────────────────────────────────────────────
# Dataset Split (adapted for ext data)
# ─────────────────────────────────────────────────────────

def split_dataset_ext(
    records: List[SubjectRecord],
    lst_dicts: list,
    config: dict,
) -> dict:
    """
    Split dataset using not_healthy / seg_masks instead of
    annotations / befund.
    """
    # Build lookup from study_uid to data_dict
    dd_map = {d["study_uid"]: d for d in lst_dicts}

    healthy, pathological, failed = [], [], []

    for r in records:
        if not r.qc_pass:
            failed.append(r)
            continue

        dd = dd_map.get(r.study_uid, {})
        is_unhealthy = dd.get("not_healthy", False)
        has_seg = bool(dd.get("seg_masks"))

        if is_unhealthy or has_seg:
            pathological.append(r)
        else:
            healthy.append(r)

    # Print summary
    for r in healthy:
        logging.info(f"  HEALTHY     {r.subject_id}")
    for r in pathological:
        dd = dd_map.get(r.study_uid, {})
        has_seg = "seg" if dd.get("seg_masks") else "no_seg"
        logging.info(f"  UNHEALTHY   {r.subject_id}  ({has_seg})")
    for r in failed:
        logging.info(f"  FAILED      {r.subject_id}  ({r.failure_reason})")

    split_info = {
        "healthy_train": [r.subject_id for r in healthy],
        "pathological_test": [r.subject_id for r in pathological],
        "failed": [
            {"subject_id": r.subject_id, "reason": r.failure_reason}
            for r in failed
        ],
        "counts": {
            "healthy": len(healthy),
            "pathological": len(pathological),
            "failed": len(failed),
            "total": len(records),
        },
    }

    out_file = pathlib.Path(config["subjects_dir"]) / "dataset_split.json"
    with open(out_file, "w") as f:
        json.dump(split_info, f, indent=2)

    logging.info(
        f"Dataset split: {len(healthy)} healthy, {len(pathological)} pathological, "
        f"{len(failed)} failed -> {out_file}"
    )

    return split_info


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Post-FastSurfer feature extraction (ext datasets)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available datasets: {', '.join(DATASET_REGISTRY.keys())}

Examples:
  python run_surfprep_step_2.py --datasets FCDBONN
  python run_surfprep_step_2.py --datasets FCDBONN --step qc
  python run_surfprep_step_2.py --datasets FCDBONN --step features --workers 16
  python run_surfprep_step_2.py --datasets FCDBONN --modalities flair
  python run_surfprep_step_2.py --datasets FCDBONN --dry_run --limit 5
        """,
    )

    # Dataset selection
    parser.add_argument("--datasets", nargs="+", default=None,
                        help=f"Which datasets to process (default: all)")

    # Output
    parser.add_argument("--subjects_dir", type=str,
                        default=os.environ.get("SURFPREP_SUBJECTS_DIR", "data/fastsurfer_subjects"),
                        help="FastSurfer output / SUBJECTS_DIR")

    # Pipeline control
    parser.add_argument("--step", type=str, default="all",
                        choices=["all", "qc", "contrasts", "features", "resample", "geodesic", "split"],
                        help="Which pipeline step(s) to run")
    parser.add_argument("--workers", type=int, default=16,
                        help="Parallel workers")
    parser.add_argument("--geodesic_max_dist", type=float, default=100.0)

    # Modality control
    parser.add_argument("--modalities", nargs="+", default=["flair"],
                        choices=ALL_MODALITIES,
                        help="Which modalities to process (default: flair only for ext)")
    parser.add_argument("--contrast_selection", type=str, default="first",
                        choices=["best", "first"])

    # Options
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip", action="store_false", dest="skip_existing")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--euler_threshold", type=int, default=-200)

    args = parser.parse_args()

    # Override the engine's QC Euler threshold from the CLI
    import surfprep_features
    surfprep_features.EULER_THRESHOLD = args.euler_threshold

    # FreeSurfer needs SUBJECTS_DIR to find fsaverage
    os.environ["SUBJECTS_DIR"] = args.subjects_dir

    # ── Logging ──
    log_level = logging.DEBUG if args.verbose else logging.INFO
    os.makedirs(args.subjects_dir, exist_ok=True)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(
                pathlib.Path(args.subjects_dir) / "run_2_ext_pipeline.log",
                mode="a",
            ),
        ],
    )

    # ── Config ──
    config = {
        "subjects_dir": args.subjects_dir,
        "skip_existing": args.skip_existing,
        "dry_run": args.dry_run,
        "geodesic_max_dist": args.geodesic_max_dist,
        "modalities": args.modalities,
        "steps": ["contrasts", "pctsurfcon", "resample", "stack"],
    }

    # ── Load external data ──
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

    for mod in args.modalities:
        n_with = sum(1 for d in lst_dicts if d.get(mod))
        logging.info(f"  Subjects with {mod.upper():>5s}: {n_with}")

    # ── Build records ──
    records = build_records_ext(lst_dicts, config, args.contrast_selection)
    if args.limit:
        records = records[:args.limit]
    logging.info(f"Processing {len(records)} subjects")

    # ════════════════════════════════════════════════════
    # STEP 1: Quality Control
    # ════════════════════════════════════════════════════
    if args.step in ("all", "qc"):
        logging.info("=" * 60)
        logging.info("STEP 1: Quality Control")
        logging.info("=" * 60)

        for i, rec in enumerate(records):
            records[i] = run_qc(rec, config)
            status = "PASS" if rec.qc_pass else rec.failure_reason
            if not rec.qc_pass:
                logging.warning(f"  [{i+1}/{len(records)}] {rec.subject_id}: {status}")
            elif args.verbose:
                logging.info(
                    f"  [{i+1}/{len(records)}] {rec.subject_id}: PASS "
                    f"(euler: lh={rec.euler_lh}, rh={rec.euler_rh})"
                )

        n_pass = sum(1 for r in records if r.qc_pass)
        n_fail = sum(1 for r in records if not r.qc_pass)
        logging.info(f"QC: {n_pass} passed, {n_fail} failed out of {len(records)}")

        # Save QC report
        qc_report_path = pathlib.Path(args.subjects_dir) / "qc_report_ext.json"
        qc_data = []
        for r in records:
            entry = {
                "subject_id": r.subject_id,
                "qc_pass": r.qc_pass,
                "failure_reason": r.failure_reason,
                "euler_lh": r.euler_lh,
                "euler_rh": r.euler_rh,
            }
            for mod in args.modalities:
                entry[f"{mod}_available"] = mod in r.contrast_niis
            qc_data.append(entry)
        with open(qc_report_path, "w") as f:
            json.dump(qc_data, f, indent=2)
        logging.info(f"QC report saved to {qc_report_path}")

        if args.step == "qc":
            return

    # ════════════════════════════════════════════════════
    # STEP 2: Geodesic Distance + Positions (shared)
    # ════════════════════════════════════════════════════
    if args.step in ("all", "geodesic"):
        logging.info("=" * 60)
        logging.info("STEP 2: Geodesic Distance + Vertex Positions (fsaverage)")
        logging.info("=" * 60)

        compute_fsaverage_geodesic(config)
        compute_fsaverage_positions(config)

        if args.step == "geodesic":
            return

    # ════════════════════════════════════════════════════
    # STEP 3: Per-Subject Feature Extraction
    # ════════════════════════════════════════════════════
    if args.step in ("all", "contrasts", "features"):
        qc_passed = [r for r in records if r.qc_pass]
        n = len(qc_passed)

        logging.info("=" * 60)
        logging.info(f"STEP 3: Feature Extraction — {n} subjects (QC passed)")
        logging.info(f"  Modalities: {', '.join(m.upper() for m in args.modalities)}")
        logging.info("=" * 60)

        if args.step == "contrasts":
            config["steps"] = ["contrasts"]
        elif args.step == "features":
            config["steps"] = ["contrasts", "pctsurfcon", "resample", "stack"]

        t_start = time.time()

        if args.workers <= 1:
            for i, rec in enumerate(qc_passed):
                logging.info(f"[{i+1}/{n}] {rec.subject_id}")
                qc_passed[i] = process_subject(rec, config)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                future_map = {
                    executor.submit(process_subject, rec, config): idx
                    for idx, rec in enumerate(qc_passed)
                }
                done_count = 0
                for future in as_completed(future_map):
                    idx = future_map[future]
                    done_count += 1
                    try:
                        qc_passed[idx] = future.result()
                        mods_ok = [m for m in args.modalities
                                   if qc_passed[idx].registered.get(m)]
                        logging.info(
                            f"[{done_count}/{n}] {qc_passed[idx].subject_id}: "
                            f"registered {','.join(m.upper() for m in mods_ok) or 'none'}"
                        )
                    except Exception as e:
                        logging.error(
                            f"[{done_count}/{n}] {qc_passed[idx].subject_id}: ERROR {e}"
                        )

        elapsed = time.time() - t_start
        logging.info(f"Feature extraction: {elapsed/60:.1f} min for {n} subjects")

        for mod in args.modalities:
            n_reg = sum(1 for r in qc_passed if r.registered.get(mod))
            logging.info(f"  {mod.upper()} registered: {n_reg}/{n}")

        # Update records list
        passed_map = {r.subject_id: r for r in qc_passed}
        for i, rec in enumerate(records):
            if rec.subject_id in passed_map:
                records[i] = passed_map[rec.subject_id]

    # ════════════════════════════════════════════════════
    # STEP 3b: Resample atlas labels only
    # ════════════════════════════════════════════════════
    if args.step == "resample":
        logging.info("Running QC check...")
        for i, rec in enumerate(records):
            records[i] = run_qc(rec, config)

        qc_passed = [r for r in records if r.qc_pass]
        n = len(qc_passed)

        logging.info("=" * 60)
        logging.info(f"Resampling DKT atlas annotations to fsaverage — {n} subjects")
        logging.info("=" * 60)

        for i, rec in enumerate(qc_passed):
            ok = resample_atlas_labels(rec, config)
            status = "OK" if ok else "FAILED"
            logging.info(f"  [{i+1}/{n}] {rec.subject_id}: {status}")

        return

    # ════════════════════════════════════════════════════
    # STEP 4: Dataset Split
    # ════════════════════════════════════════════════════
    if args.step in ("all", "split"):
        logging.info("=" * 60)
        logging.info("STEP 4: Dataset Split")
        logging.info("=" * 60)

        lst_dicts = update_data_dicts_with_status_ext(lst_dicts, records, args.modalities)
        split_info = split_dataset_ext(records, lst_dicts, config)

        final_path = pathlib.Path(args.subjects_dir) / "data_dicts_ext_final.json"
        with open(final_path, "w") as f:
            json.dump(lst_dicts, f, indent=2, default=str)
        logging.info(f"Final data dicts saved to {final_path}")

    # ════════════════════════════════════════════════════
    # Summary
    # ════════════════════════════════════════════════════
    logging.info("=" * 60)
    logging.info("PIPELINE COMPLETE")
    logging.info("=" * 60)

    n_pass = sum(1 for r in records if r.qc_pass)
    n_feat = sum(1 for r in records if r.features_extracted)
    n_fail = sum(1 for r in records if not r.qc_pass)

    logging.info(f"  Total subjects:        {len(records)}")
    logging.info(f"  QC passed:             {n_pass}")
    logging.info(f"  QC failed:             {n_fail}")
    logging.info(f"  Features extracted:    {n_feat}")

    for mod in args.modalities:
        n_reg = sum(1 for r in records if r.registered.get(mod))
        logging.info(f"  {mod.upper()} registered:      {n_reg}")


if __name__ == "__main__":
    main()