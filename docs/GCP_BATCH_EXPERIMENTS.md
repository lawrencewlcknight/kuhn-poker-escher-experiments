# Running the ESCHER experiments on Google Cloud Batch

This guide explains how to run the ESCHER Kuhn poker experiments on Google
Cloud using Google Batch. The workflow is designed to be repeatable and mostly
command-line driven:

1. configure Google Cloud locally;
2. create a Cloud Storage bucket for outputs;
3. create a service account for Batch jobs;
4. create the Batch submission script;
5. run a smoke test;
6. run full experiments with configurable CPU and memory;
7. inspect logs and retrieve outputs.

The Batch job creates a temporary VM, clones this GitHub repository, creates an
isolated Python 3.9 virtual environment, installs the repo dependencies, runs
the selected experiment, copies outputs to Cloud Storage, and then exits. Batch
handles VM lifecycle management, so there is no persistent VM to shut down after
a successful job.

---

## 1. Prerequisites

You need:

- a Google Cloud project with billing enabled;
- the Google Cloud CLI installed on your local machine;
- permission to create service accounts, IAM bindings, Batch jobs, and Cloud
  Storage buckets;
- this GitHub repository available to the Batch VM.

If the repository is public, the script can clone it directly with HTTPS. If the
repository is private, adapt the `git clone` step to use an authenticated method
such as a deploy key, GitHub token, or a pre-built container image.

This repository is pinned to Python 3.9 in `pyproject.toml`, `.python-version`,
`runtime.txt`, and `project.toml`. The Batch script below uses `uv` to create a
Python 3.9 virtual environment even when the base VM image ships with a newer
system Python.

---

## 2. One-time local Google Cloud setup

Authenticate and select your project:

```bash
gcloud init
gcloud auth login

export PROJECT_ID="your-gcp-project-id"
gcloud config set project "$PROJECT_ID"
```

For UK-based use where latency is not important and cost is a consideration,
`europe-west1` is a sensible default region:

```bash
export REGION="europe-west1"
export ZONE="europe-west1-b"
```

Enable the required APIs:

```bash
gcloud services enable \
  compute.googleapis.com \
  batch.googleapis.com \
  logging.googleapis.com \
  storage.googleapis.com
```

---

## 3. Create a Cloud Storage bucket for experiment outputs

Create a regional bucket in the same region as the Batch jobs:

```bash
export BUCKET_NAME="${PROJECT_ID}-kuhn-poker-escher-results"
export BUCKET="gs://${BUCKET_NAME}"

gcloud storage buckets create "$BUCKET" \
  --location="$REGION" \
  --uniform-bucket-level-access
```

Check the bucket exists:

```bash
gcloud storage buckets describe "$BUCKET"
```

If the bucket exists and is accessible, this command will print metadata such as
the bucket name, creation time, location, storage class, and storage URL.

---

## 4. Create a service account for Batch jobs

Create a dedicated service account:

```bash
export SA_NAME="kuhn-escher-runner"
export SA_EMAIL="${SA_NAME}@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud iam service-accounts create "$SA_NAME" \
  --display-name="Kuhn poker ESCHER experiment runner" \
  --project="$PROJECT_ID"
```

Grant the service account permission to write logs:

```bash
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/logging.logWriter"
```

Grant the service account permission to report Batch agent status:

```bash
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/batch.agentReporter"
```

Grant the service account permission to write experiment outputs to the bucket:

```bash
gcloud storage buckets add-iam-policy-binding "$BUCKET" \
  --member="serviceAccount:${SA_EMAIL}" \
  --role="roles/storage.objectAdmin"
```

Allow your user account to run jobs as this service account. Replace the email
address with your Google account email:

```bash
export YOUR_EMAIL="your-email@example.com"

gcloud iam service-accounts add-iam-policy-binding "$SA_EMAIL" \
  --member="user:${YOUR_EMAIL}" \
  --role="roles/iam.serviceAccountUser"
```

If you need to inspect logs from your local account, make sure your user has
log-viewing permission:

```bash
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --member="user:${YOUR_EMAIL}" \
  --role="roles/logging.viewer"
```

---

## 5. Environment variables to set in each new terminal

Before submitting jobs from a new shell session, set:

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="europe-west1"
export BUCKET="gs://${PROJECT_ID}-kuhn-poker-escher-results"
export SA_EMAIL="kuhn-escher-runner@${PROJECT_ID}.iam.gserviceaccount.com"

gcloud config set project "$PROJECT_ID"
```

Check the values:

```bash
echo "$PROJECT_ID"
echo "$REGION"
echo "$BUCKET"
echo "$SA_EMAIL"
```

---

## 6. Create the Batch submission script

The repository already includes the maintained submission helper at
`gcp/submit_batch_experiment.sh`. Use the checked-in script rather than copying a
stale version from this guide. The helper:

- creates the Batch job JSON;
- clones the GitHub repo on the Batch VM;
- installs Python 3.9 dependencies;
- installs Debian's `time` package so `/usr/bin/time -v` diagnostics work;
- fails the Batch task if the experiment command exits non-zero;
- writes `batch_run.log`, `resource_snapshots.log`, and `batch_status.json`;
- uploads `outputs/` to Cloud Storage from a cleanup trap on success or failure;
- uses a configurable boot disk size, defaulting to `100` GiB.

If the script is missing in another checkout, restore it from git:

```bash
git checkout -- gcp/submit_batch_experiment.sh
```

Make the script executable and check its syntax:

```bash
chmod +x gcp/submit_batch_experiment.sh
bash -n gcp/submit_batch_experiment.sh
```

`bash -n` should return silently. If it prints an error, fix the script before
submitting a Batch job.

---

## 7. Run a smoke test

Before running a full experiment, submit a very small ESCHER baseline smoke test:

```bash
./gcp/submit_batch_experiment.sh \
  "escher-smoke-exp1-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.kuhn_poker.escher_multiseed_baseline.run \
    --seeds 1234 \
    --iterations 2 \
    --traversals 5 \
    --value-traversals 5 \
    --policy-network-train-steps 2 \
    --regret-network-train-steps 2 \
    --value-network-train-steps 2 \
    --evaluation-interval 1 \
    --output-root outputs/cloud/escher-smoke-exp1" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"
```

The script prints the Batch script before submission. Check that the upload line
uses `gcloud storage cp`, for example:

```bash
gcloud storage cp --recursive outputs "gs://your-project-id-kuhn-poker-escher-results/escher-smoke-exp1-.../"
```

After submission, this command can be used to check whether the experiment job is
queued, scheduled, running, or complete:

```bash
gcloud batch jobs list --location "$REGION"
```

---

## 8. Monitor a Batch job

List jobs:

```bash
gcloud batch jobs list --location "$REGION"
```

Describe one job:

```bash
gcloud batch jobs describe JOB_NAME --location "$REGION"
```

After an experiment has completed, the output from `gcloud batch jobs describe`
can also be reviewed to determine how long the Batch job took to run. This is
useful for comparing VM sizes and setting future `MAX_RUN_SECONDS` values.

Possible states include:

- `QUEUED`
- `SCHEDULED`
- `RUNNING`
- `SUCCEEDED`
- `FAILED`

If the job succeeds, list the uploaded outputs:

```bash
gcloud storage ls -r "$BUCKET/JOB_NAME/"
```

Copy outputs back to your local machine:

```bash
mkdir -p cloud_outputs/JOB_NAME
gcloud storage cp -r "$BUCKET/JOB_NAME/*" "cloud_outputs/JOB_NAME/"
```

After reviewing the downloaded outputs, promote only lightweight thesis-facing
artifacts into the tracked repo:

```bash
python scripts/promote_thesis_artifacts.py cloud_outputs/JOB_NAME --dry-run
python scripts/promote_thesis_artifacts.py cloud_outputs/JOB_NAME
```

See [THESIS_ARTIFACTS.md](THESIS_ARTIFACTS.md) for the promotion workflow and
filtering rules.

---

## 9. Read logs for a failed job

First describe the job and find its `uid`:

```bash
gcloud batch jobs describe JOB_NAME --location "$REGION"
```

Then query task logs using the helper script:

```bash
./gcp/read_batch_task_logs.sh JOB_NAME
```

You can pass text filters after the job name. The script still scopes the query
by the exact Batch `labels.job_uid` before applying those filters:

```bash
./gcp/read_batch_task_logs.sh JOB_NAME ERROR Traceback Killed maxRunDuration
```

The helper also includes a contamination guard. If any returned line contains an
`escher-exp...` job name different from the target job, the script prints a
warning and exits non-zero so you know the log extract should not be trusted.

If you need to run the raw command manually, first describe the job and use its
exact `uid`:

```bash
gcloud logging read \
  'logName="projects/YOUR_PROJECT_ID/logs/batch_task_logs" AND labels.job_uid="JOB_UID"' \
  --limit=500 \
  --format="value(timestamp,severity,textPayload,jsonPayload.message)"
```

When adding text filters manually, keep the UID clause outside and before any
text-filter expression, for example:

```bash
gcloud logging read \
  'logName="projects/YOUR_PROJECT_ID/logs/batch_task_logs" AND labels.job_uid="JOB_UID" AND (textPayload:"ERROR" OR jsonPayload.message:"ERROR")' \
  --limit=500 \
  --format="value(timestamp,severity,textPayload,jsonPayload.message)"
```

Useful things to look for:

- `All outputs saved to:` shows where the experiment wrote local VM outputs;
- `Experiment command exit code: 0` and `Experiment completed successfully.`
  mean the Python experiment returned before cleanup/upload;
- `Experiment failed with exit code ...` means the experiment command failed,
  but the cleanup trap still attempted to upload partial outputs and logs;
- `No space left on device` indicates disk pressure;
- `Killed`, `exit code 137`, or `Out of memory` usually indicates memory pressure;
- `maxRunDuration` means the job hit the time limit;
- `Invalid machine type` or resource errors usually mean the requested CPU/memory
  does not fit the selected machine type;
- Python version mismatches will usually show up during dependency installation,
  so check for `python --version` and TensorFlow wheel errors.

The submission script also writes durable job diagnostics under
`outputs/cloud/<JOB_NAME>/` and uploads them in a shell cleanup trap on both
success and failure:

- `batch_run.log` contains the generated Batch script output captured with
  `tee`;
- `resource_snapshots.log` records periodic `df`, `free`, and largest-process
  snapshots;
- `batch_status.json` records the job name, exit code, cleanup timestamp, and
  bucket destination.

If Cloud Logging is empty or incomplete, copy the job folder back from Cloud
Storage and inspect these files first.

---

## 10. Run the full Experiment 1

After the smoke test succeeds, run the full ESCHER multi-seed baseline.

A safe starting configuration is `n2-standard-4`, 4 vCPUs, 16 GiB memory, and a
12-hour runtime cap:

```bash
./gcp/submit_batch_experiment.sh \
  "escher-exp1-baseline-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.kuhn_poker.escher_multiseed_baseline.run \
    --output-root outputs/cloud/escher-exp1-baseline" \
  "n2-standard-4" \
  "43200" \
  "4000" \
  "16000"
```

To collect process-level runtime and memory diagnostics, wrap the experiment
command with `/usr/bin/time -v`:

```bash
./gcp/submit_batch_experiment.sh \
  "escher-exp1-baseline-$(date +%Y%m%d-%H%M%S)" \
  "/usr/bin/time -v python -m experiments.kuhn_poker.escher_multiseed_baseline.run \
    --output-root outputs/cloud/escher-exp1-baseline" \
  "n2-standard-4" \
  "43200" \
  "4000" \
  "16000"
```

In the logs, look for:

- `Elapsed (wall clock) time`;
- `Percent of CPU this job got`;
- `Maximum resident set size`.

These help determine whether the VM is over- or under-sized.

---

## 11. Changing CPU, memory, and boot disk

The final four arguments control the VM and Batch task resources:

```bash
MACHINE_TYPE MAX_RUN_SECONDS CPU_MILLI MEMORY_MIB BOOT_DISK_SIZE_GB
```

`BOOT_DISK_SIZE_GB` is optional; omit it to use the script default.

Examples:

| Machine type | CPU milli | Memory MiB | Approximate resources |
|---|---:|---:|---|
| `n2-standard-2` | `2000` | `8000` | 2 vCPUs, about 8 GiB RAM |
| `n2-standard-4` | `4000` | `16000` | 4 vCPUs, about 16 GiB RAM |
| `n2-standard-8` | `8000` | `32000` | 8 vCPUs, about 32 GiB RAM |

The Batch resource request must fit inside the selected machine type. For
example, do not request `4000` CPU milli and `16000` MiB memory on an
`n2-standard-2` machine.

The boot disk defaults to `100` GiB in the checked-in submission script. This is
larger than the default Batch boot disk and gives dependency installation,
TensorFlow/OpenSpiel caches, intermediate outputs, and failure logs more room.
For long debugging runs, keep the larger boot disk unless you have evidence that
disk pressure is not relevant.

### Smaller VM test

```bash
./gcp/submit_batch_experiment.sh \
  "escher-exp1-baseline-small-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.kuhn_poker.escher_multiseed_baseline.run \
    --output-root outputs/cloud/escher-exp1-baseline-small" \
  "n2-standard-2" \
  "43200" \
  "2000" \
  "8000"
```

### Larger VM test

```bash
./gcp/submit_batch_experiment.sh \
  "escher-exp1-baseline-large-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.kuhn_poker.escher_multiseed_baseline.run \
    --output-root outputs/cloud/escher-exp1-baseline-large" \
  "n2-standard-8" \
  "43200" \
  "8000" \
  "32000"
```

---

## 12. Choosing a VM size

Use evidence rather than guessing.

Start with:

```text
n2-standard-4, CPU_MILLI=4000, MEMORY_MIB=16000
```

Then compare with:

```text
n2-standard-2, CPU_MILLI=2000, MEMORY_MIB=8000
n2-standard-8, CPU_MILLI=8000, MEMORY_MIB=32000
```

A VM may be too large if:

- memory usage is far below the allocation;
- CPU utilisation is consistently low;
- doubling vCPUs does not materially reduce runtime;
- the job is bottlenecked by Python/OpenSpiel traversal rather than CPU capacity.

A VM may be too small if:

- the job fails with `Killed`, `Out of memory`, or exit code `137`;
- the job hits `maxRunDuration`;
- logs show `No space left on device`;
- the job is much slower than expected.

---

## 13. Runtime limits

`MAX_RUN_SECONDS` is a safety cap. For example:

| Seconds | Duration |
|---:|---:|
| `3600` | 1 hour |
| `21600` | 6 hours |
| `43200` | 12 hours |
| `86400` | 24 hours |

If a job exceeds `MAX_RUN_SECONDS`, Batch stops the task and marks the job as
failed. The submission script installs cleanup traps that attempt to upload
partial outputs and debug logs even on failure, but a hard VM/agent loss can
still prevent a final upload.

For full experiments, use a generous cap such as 12 or 24 hours unless you are
deliberately testing runtime behaviour.

---

## 14. Running other experiments

Use the same submission pattern but change the Python module and `--output-root`.

Template:

```bash
./gcp/submit_batch_experiment.sh \
  "JOB_PREFIX-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.kuhn_poker.EXPERIMENT_MODULE.run \
    --output-root outputs/cloud/JOB_PREFIX" \
  "n2-standard-4" \
  "43200" \
  "4000" \
  "16000"
```

Available ESCHER experiment modules:

| Experiment | Module |
|---|---|
| 1. Multi-seed baseline | `escher_multiseed_baseline` |
| 2. Intermediate policy-training ablation | `escher_intermediate_policy_training_ablation` |
| 3. Checkpoint stability | `escher_checkpoint_stability` |
| 4. Constrained hyperparameter search | `escher_constrained_hyperparameter_search` |
| 5. Warm-start fair ablation | `escher_warm_start_fair_ablation` |
| 6. Learning-rate schedule ablation | `escher_lr_schedule_ablation` |
| 7. Reach-weighting ablation | `escher_reach_weighting_ablation` |
| 8. Value-trajectory reuse ablation | `escher_reuse_value_trajectory_ablation` |
| 9. Disk-backed regret-memory ablation | `escher_disk_backed_regret_memory_ablation` |
| 10. On-policy joint-regret ablation | `escher_on_policy_joint_regret_ablation` |
| 11. Solver-parameter random search | `escher_solver_parameter_random_search` |
| 12. Diagnostic hypothesis sweep | `escher_diagnostic_hypothesis_sweep` |
| 13. Author-budget multi-seed validation | `escher_author_budget_multiseed` |
| 14. Network-size sweep | `escher_network_size_sweep` |

Example:

```bash
./gcp/submit_batch_experiment.sh \
  "escher-exp6-lr-schedule-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.kuhn_poker.escher_lr_schedule_ablation.run \
    --output-root outputs/cloud/escher-exp6-lr-schedule" \
  "n2-standard-4" \
  "43200" \
  "4000" \
  "16000"
```

Hyperparameter-search experiments can run much longer than the smaller ablations.
For initial cloud validation, reduce candidates and seeds first:

```bash
./gcp/submit_batch_experiment.sh \
  "escher-exp11-search-smoke-$(date +%Y%m%d-%H%M%S)" \
  "python -m experiments.kuhn_poker.escher_solver_parameter_random_search.run \
    --screening-seeds 1234 \
    --confirmation-seeds 1234 \
    --screening-iterations 2 \
    --confirmation-iterations 2 \
    --screening-evaluation-interval 1 \
    --confirmation-evaluation-interval 1 \
    --n-random-candidates 1 \
    --confirmation-top-k 1 \
    --traversals 5 \
    --value-traversals 5 \
    --policy-network-train-steps 2 \
    --regret-network-train-steps 2 \
    --value-network-train-steps 2 \
    --policy-network-layers 32,32 \
    --regret-network-layers 32,32 \
    --value-network-layers 32,32 \
    --all-actions true \
    --use-balanced-probs false \
    --val-bootstrap false \
    --output-root outputs/cloud/escher-exp11-search-smoke" \
  "n2-standard-4" \
  "3600" \
  "4000" \
  "16000"
```

---

## 15. Cleaning up

Batch-created VMs are temporary and should terminate when the job completes or
fails. Failed job records can be deleted from Batch if desired:

```bash
gcloud batch jobs delete JOB_NAME --location "$REGION" --quiet
```

Do not delete the Cloud Storage bucket unless you are sure you no longer need
any outputs.

To inspect bucket contents:

```bash
gcloud storage ls -r "$BUCKET/"
```

To delete a specific job output folder:

```bash
gcloud storage rm -r "$BUCKET/JOB_NAME/"
```

---

## 16. Notes on dependency installation

The script deliberately creates a virtual environment under
`/tmp/kuhn-escher-venv` rather than installing packages into the system Python.
This is important because installing experiment dependencies into the system
Python can interfere with the Google Cloud CLI on the Batch VM.

The script uses:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.10
export CLOUDSDK_PYTHON="$(uv python find 3.10)"
uv python install 3.9
uv venv --python 3.9 --seed /tmp/kuhn-escher-venv
source /tmp/kuhn-escher-venv/bin/activate
python -m pip install --no-cache-dir --no-build-isolation -r requirements.txt
python -m pip install --no-cache-dir --no-build-isolation -e .
```

The ESCHER experiment runs from the Python 3.9 virtual environment. Cloud SDK is
pointed at a separate Python 3.10 runtime with `CLOUDSDK_PYTHON`, because current
`gcloud storage` no longer supports Python 3.9. Uploading is handled after the
experiment by `gcloud storage cp --recursive`, not by `gsutil` or by Google
Python client libraries installed into the experiment environment.

The script deactivates the experiment environment after the experiment command.
Output copying is handled by the cleanup trap so it runs after both successful
and failed experiment commands:

```bash
deactivate || true
# cleanup trap uploads outputs to Cloud Storage
```

The script does not use `gsutil`; this avoids the `gsutil` Python-runtime issue
seen on some Batch images while also avoiding upload-time dependencies on the
experiment virtual environment.
