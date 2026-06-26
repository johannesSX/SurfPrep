# SurfPrep — Cortical Surface Preprocessing

SurfPrep turns raw **T1-weighted MRI** into **cortical-surface features on `fsaverage`**. It runs [FastSurfer](https://github.com/Deep-MI/FastSurfer) for the reconstruction, then extracts the features, resamples them to `fsaverage`, and packages them. The output feeds surface models such as [GLOWORM](https://github.com/johannesSX/GLOWORM).

Per subject it writes `lh/rh.morpho.npz` — five features (thickness, curvature, sulcal depth, area, gray–white contrast) on 163,842 vertices — plus the DKT atlas on fsaverage and a shared mesh.

It includes reader-scripts for three public cohorts, and you can add your own ([Section 8](#8-add-your-own-dataset)).

| Cohort                                                                        | Subjects | Ground-truth mask |
|-------------------------------------------------------------------------------|---|---|
| **IXI** ([brain-development.org](https://brain-development.org/ixi-dataset/)) | healthy controls | none |
| **FCD Bonn** ([OpenNeuro ds004199](https://openneuro.org/datasets/ds004199/)) | focal cortical dysplasia type II | `*_roi.nii.gz` |
| **IDEAS** ([OpenNeuro ds005602](https://openneuro.org/datasets/ds005602/))    | mixed epileptogenic lesions | resection mask |

> _This public release was refactored and documented with assistance from Claude Opus 4.8 for clarity. No changes were made to the scientific methods, models, or results._

## 1. What it produces

One FastSurfer subjects directory holds every subject as `<cohort>__<id>`:

```
<subjects_dir>/
├── fsaverage/                         # standard FreeSurfer fsaverage (template)
├── fsaverage_common/                  # shared mesh, written once
│   ├── lh.positions.npy   rh.positions.npy     # (163842, 3) vertex coords
│   └── lh.edge_index.pt   rh.edge_index.pt     # mesh connectivity
└── fcdbonn__sub-00055/
    ├── mri/  surf/  label/  stats/    # FastSurfer reconstruction
    ├── lh.morpho.npz  rh.morpho.npz   # (163842, 5): thickness,curv,sulc,area,wg_pct
    └── fsaverage_features/            # DKT atlas resampled to fsaverage
        ├── lh.aparc.DKTatlas.fsaverage.annot
        └── rh.aparc.DKTatlas.fsaverage.annot
```

Lesion masks stay in the raw data directory. SurfPrep only records their paths; it never changes or moves them.

## 2. Prerequisites

**1. Docker + an NVIDIA GPU** (recommended). Segmentation uses the GPU; surface reconstruction uses the CPU. The included [`run_fastsurfer.sh`](run_fastsurfer.sh) runs the official FastSurfer image.

**2. A FreeSurfer license** (needed for the surface stage). Register at <https://surfer.nmr.mgh.harvard.edu/registration.html>, save the `license.txt`, and point `FS_LICENSE` at it. The license is free for academic / non-profit research; commercial use is not allowed (see the FreeSurfer EULA).

**3. Python 3.9+:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
chmod +x run_fastsurfer.sh        # make the Docker wrapper executable
```

## 3. Configuration

Set these environment variables (each script also has a matching flag, which wins):

```bash
export SURFPREP_DATA_ROOT=/abs/path/to/data                    # raw cohorts
export SURFPREP_SUBJECTS_DIR=/abs/path/to/fastsurfer_subjects  # FastSurfer output
export FS_LICENSE=$HOME/freesurfer/license.txt                 # FreeSurfer license
export FREESURFER_HOME=/usr/local/freesurfer                   # mri_* tools for step 2
export SURFPREP_DATA=/abs/path/to/data                         # host dir mounted into Docker
```

**Use absolute paths.** FastSurfer runs inside Docker, and the wrapper mounts `SURFPREP_DATA` at the same path inside the container. So the raw data and the output subjects dir must both be absolute and live under `SURFPREP_DATA`, or FastSurfer can't see them. Easiest: one parent folder for everything, with `SURFPREP_DATA` = `SURFPREP_DATA_ROOT` = that folder.

## 4. Raw data layout

Put each cohort under `SURFPREP_DATA_ROOT`. The readers in `datasets.py` look for these paths:

```
data/
├── IXI/                                          # https://brain-development.org/ixi-dataset/
│   ├── IXI-T1/IXI002-Guys-0828-T1.nii.gz
│   └── IXI-T2/IXI002-Guys-0828-T2.nii.gz
├── FCDBONN/                                      # https://openneuro.org/datasets/ds004199/
│   └── sub-00055/anat/
│       ├── sub-00055_T1w.nii.gz
│       ├── sub-00055_FLAIR.nii.gz
│       └── sub-00055_roi.nii.gz                  # lesion mask (patients only)
└── IDEAS/                                        # https://openneuro.org/datasets/ds005602/
    ├── ds005602/sub-1/anat/sub-1_T1w.nii.gz
    └── ds005602_masks/1/1_MaskInOrig.nii.gz      # resection mask
```

Subject IDs become `ixi__IXI002-Guys`, `fcdbonn__sub-00055`, `ideas__sub-1`.

## 5. Pipeline

Consists of two scripts. Each script needs `--datasets` and `--subjects_dir`. You can try `--dry_run` / `--limit 5` first to check the paths.

### Step 1 — FastSurfer reconstruction

`--fastsurfer_dir` must point at a folder that contains an executable `run_fastsurfer.sh` — either this repo (the bundled Docker wrapper) or a native FastSurfer install. Two notes when using the wrapper:

- Run `chmod +x run_fastsurfer.sh` once.
- Give `--fastsurfer_dir` an **absolute** path.

```bash
# segmentation first (GPU, ~1–2 min/subject)
python run_surfprep_step_1.py --datasets FCDBONN \
    --fastsurfer_dir "$PWD" --subjects_dir "$SURFPREP_SUBJECTS_DIR" \
    --mode seg_only --device cuda

# then surfaces (CPU, ~45–60 min/subject)
python run_surfprep_step_1.py --datasets FCDBONN \
    --fastsurfer_dir "$PWD" --subjects_dir "$SURFPREP_SUBJECTS_DIR" \
    --mode surf_only --cpu_workers 8
```

`--mode` has three options, and the two stages can run on different machines:

- `seg_only` — the **GPU** stage (segmentation), ~1–2 min/subject.
- `surf_only` — the **CPU** stage (surface reconstruction), ~45–60 min/subject, split over `--cpu_workers`.
- `full` — both, one machine.

Since segmentation needs a GPU and surface reconstruction needs CPU, you can run `seg_only` on a GPU machine and `surf_only` on CPU machines (useful on a cluster with few GPUs). The only requirement: `--subjects_dir` is on shared storage both machines can reach. On a CPU-only machine, turn off the Docker GPU request first: `export DOCKER_GPU_FLAG=""`. Use `--filter_prefix ideas__` to limit a run to one cohort.

### Step 2 — feature extraction → fsaverage

QC (Euler number) → optional FLAIR registration → morphometric features → gray–white contrast → resample to fsaverage → resample the DKT atlas → write `lh/rh.morpho.npz`. It also writes the shared `fsaverage_common/` mesh once.

```bash
python run_surfprep_step_2.py --datasets FCDBONN \
    --subjects_dir "$SURFPREP_SUBJECTS_DIR" --workers 16
```

Run one stage at a time with `--step {qc,geodesic,features,resample,split}`.


## 6. How a subject flows through the code

```
datasets.py            datasets_adapter.py        run_surfprep_step_1/2.py
─────────────          ───────────────────        ────────────────────────
DatasetXxx.read()  →   read_data_ext()        →   FastSurfer recon + features
(globs raw files)      (uniform records,          (engines: surfprep_*.py)
                        cohort-prefixed IDs)
```

`datasets.py` finds the files. `datasets_adapter.py` turns every cohort into one record shape and assigns the `<cohort>__<id>` IDs. The step scripts call the engines (`surfprep_fastsurfer.py`, `surfprep_features.py`).

## 7. Add your own dataset

Two small edits: write a reader, then register it.

### 7a. Write a reader in `datasets.py`

Subclass `SuperDataset`, find your files, and return one record per subject. Only `t1` and `keyword` are required; add `flair`/`t2`/`swi` if you have them, `seg` for a ground-truth mask, and `not_healthy` to mark patients.

```python
class DatasetMyCohort(SuperDataset):
    def __init__(self):
        super().__init__()
        self.file_cat = "MYCOHORT"

    def read(self):
        template_dict = super().get_template_dict()
        records = []

        for t1 in glob.glob(f"{DATA_ROOT}/MYCOHORT/*/anat/*_T1w.nii.gz"):
            d = copy.deepcopy(template_dict)
            d["t1"] = [t1]                       # required (one absolute path)
            d["keyword"] = ["mycohort"]          # required (lower-case tag)

            flair = t1.replace("_T1w", "_FLAIR")  # optional extra contrast
            if os.path.exists(flair):
                d["flair"] = [flair]

            mask = glob.glob(str(pathlib.Path(t1).parent / "*_roi.nii.gz"))
            d["seg"] = mask                       # optional ground-truth mask
            d["not_healthy"] = [len(mask) > 0]

            records.append(d)
        return records
```

`{DATA_ROOT}` is `SURFPREP_DATA_ROOT`. Paths should be absolute.

### 7b. Register it in `datasets_adapter.py`

Add the class to the registry:

```python
from datasets import DatasetIXI, DatasetFCDBONN, DatasetIdeas, DatasetMyCohort

DATASET_REGISTRY = {
    "IXI":      DatasetIXI,
    "FCDBONN":  DatasetFCDBONN,
    "Ideas":    DatasetIdeas,
    "MyCohort": DatasetMyCohort,        # <- add
}
```

Then add a branch to `_derive_subject_id()` so the IDs come out as `mycohort__<id>`:

```python
elif kw == "mycohort":
    # .../MYCOHORT/sub-XXX/anat/sub-XXX_T1w.nii.gz  -> mycohort__sub-XXX
    return f"mycohort__{p.parent.parent.name}"
```

Now run the pipeline with `--datasets MyCohort`.

## 8. Notes
- **License**: this code uses the repository LICENSE. FastSurfer and FreeSurfer have their own licenses (FreeSurfer: academic / non-profit only).