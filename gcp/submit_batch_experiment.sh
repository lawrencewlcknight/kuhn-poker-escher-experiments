#!/usr/bin/env bash
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive

# Usage:
#   ./gcp/submit_batch_experiment.sh \
#     JOB_NAME \
#     "PYTHON_EXPERIMENT_COMMAND" \
#     MACHINE_TYPE \
#     MAX_RUN_SECONDS \
#     CPU_MILLI \
#     MEMORY_MIB
#
# Examples:
#   n2-standard-2: CPU_MILLI=2000 MEMORY_MIB=8000
#   n2-standard-4: CPU_MILLI=4000 MEMORY_MIB=16000
#   n2-standard-8: CPU_MILLI=8000 MEMORY_MIB=32000

JOB_NAME="$1"
EXPERIMENT_COMMAND="$2"
MACHINE_TYPE="${3:-n2-standard-4}"
MAX_RUN_SECONDS="${4:-21600}"
CPU_MILLI="${5:-4000}"
MEMORY_MIB="${6:-16000}"

: "${PROJECT_ID:?Set PROJECT_ID first}"
: "${REGION:?Set REGION first}"
: "${BUCKET:?Set BUCKET first}"
: "${SA_EMAIL:?Set SA_EMAIL first}"

JOB_JSON="$(mktemp "/tmp/${JOB_NAME}.XXXXXX.json")"

export JOB_NAME
export EXPERIMENT_COMMAND
export MACHINE_TYPE
export MAX_RUN_SECONDS
export CPU_MILLI
export MEMORY_MIB
export BUCKET
export SA_EMAIL
export JOB_JSON

python3 <<'PY'
import json
import os

job_json_path = os.environ["JOB_JSON"]
job_name = os.environ["JOB_NAME"]
experiment_command = os.environ["EXPERIMENT_COMMAND"]
machine_type = os.environ["MACHINE_TYPE"]
max_run_seconds = os.environ["MAX_RUN_SECONDS"]
cpu_milli = int(os.environ["CPU_MILLI"])
memory_mib = int(os.environ["MEMORY_MIB"])
bucket = os.environ["BUCKET"]
service_account = os.environ["SA_EMAIL"]

script = f"""#!/usr/bin/env bash
set -euxo pipefail

export DEBIAN_FRONTEND=noninteractive

echo "Starting job: {job_name}"
echo "Experiment command: {experiment_command}"
echo "Requested CPU milli: {cpu_milli}"
echo "Requested memory MiB: {memory_mib}"

if command -v sudo >/dev/null 2>&1; then
  SUDO=sudo
else
  SUDO=
fi

$SUDO apt-get update
$SUDO apt-get install -y git curl ca-certificates python3 python3-pip python3-venv python3-dev build-essential

WORKDIR=/workspace
mkdir -p "$WORKDIR"
cd "$WORKDIR"

git clone --depth 1 https://github.com/lawrencewlcknight/kuhn-poker-escher-experiments.git
cd kuhn-poker-escher-experiments

export HOME="${{HOME:-/root}}"
export TMPDIR="/tmp"
export PIP_CACHE_DIR="/tmp/pip-cache"
export UV_CACHE_DIR="/tmp/uv-cache"
export PATH="$HOME/.local/bin:/usr/local/bin:$PATH"

mkdir -p "$HOME" "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR"

# Log basic machine information for later VM right-sizing.
echo "Machine information:"
nproc || true
free -h || true
df -h || true
lscpu | head -30 || true

# Use Python 3.9 to match the repository metadata and TensorFlow requirements.
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv python install 3.9
uv venv --python 3.9 --seed /tmp/kuhn-escher-venv
source /tmp/kuhn-escher-venv/bin/activate
python --version

python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir --no-build-isolation -r requirements.txt
python -m pip install --no-cache-dir --no-build-isolation -e .
python -m pip check || true

mkdir -p "outputs/cloud/{job_name}"

{experiment_command}

deactivate

echo "Experiment completed. Copying outputs to Cloud Storage."
gsutil -m cp -r outputs "{bucket}/{job_name}/"

echo "Done."
"""

job = {
    "taskGroups": [
        {
            "taskSpec": {
                "runnables": [
                    {
                        "script": {
                            "text": script
                        }
                    }
                ],
                "computeResource": {
                    "cpuMilli": cpu_milli,
                    "memoryMib": memory_mib,
                },
                "maxRetryCount": 0,
                "maxRunDuration": f"{max_run_seconds}s",
            },
            "taskCount": 1,
            "parallelism": 1,
        }
    ],
    "allocationPolicy": {
        "serviceAccount": {
            "email": service_account
        },
        "instances": [
            {
                "policy": {
                    "machineType": machine_type,
                    "provisioningModel": "STANDARD",
                }
            }
        ],
    },
    "logsPolicy": {
        "destination": "CLOUD_LOGGING"
    },
}

with open(job_json_path, "w", encoding="utf-8") as f:
    json.dump(job, f, indent=2)
PY

echo "Submitting Batch job: ${JOB_NAME}"
echo "Machine type: ${MACHINE_TYPE}"
echo "Max run duration: ${MAX_RUN_SECONDS}s"
echo "CPU milli: ${CPU_MILLI}"
echo "Memory MiB: ${MEMORY_MIB}"
echo "Job config: ${JOB_JSON}"

echo
echo "Script that will run inside Batch:"
echo "-----------------------------------"
python3 - "$JOB_JSON" <<'PY'
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    job = json.load(f)

print(job["taskGroups"][0]["taskSpec"]["runnables"][0]["script"]["text"])
PY
echo "-----------------------------------"
echo

gcloud batch jobs submit "${JOB_NAME}" \
  --location "${REGION}" \
  --config "${JOB_JSON}"

echo "Submitted."
echo "Monitor with:"
echo "  gcloud batch jobs describe ${JOB_NAME} --location ${REGION}"
echo "Outputs will be copied to:"
echo "  ${BUCKET}/${JOB_NAME}/"
