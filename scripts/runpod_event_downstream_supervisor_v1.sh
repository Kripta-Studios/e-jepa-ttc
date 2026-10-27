#!/usr/bin/env bash
set -eu

ops=/workspace/ejepa_event_operations
run=/workspace/ejepa_event_runs
base=/opt/ejepa_event_bundle/workspace/e-jepa-ttc-v18-joint-measurement
runtime="$ops/event_downstream_runtime_v18"
python=/opt/ejepa_env/bin/python
supplement="$ops/supplement/EVENT_DOWNSTREAM_SUPPLEMENTAL_ADMISSION.json"
raw_manifest="$ops/EVENT_RAW_DEFERRED_MANIFEST.json"
raw_receipt="$runtime/artifacts/runpod_event_migration_20261026_v4/EVENT_HF_REMOTE_IDENTITY_V4.json"

while [ ! -f "$ops/EVENT_PRODUCER_PIDS.txt" ] || [ ! -f "$supplement" ]; do
    sleep 30
done

"$python" "$ops/event_downstream_bind_v1.py" \
    --run "$run" \
    --authorization "$ops/authorization/EVENT_EXECUTION_AUTHORIZATION.json" \
    --budget-source "$ops/authorization/POD2_BUDGET_ACTIVE.json" \
    --output "$ops"

cp /workspace/ejepa_event_bundle/event_migration/EVENT_RAW_DEFERRED_MANIFEST.json "$raw_manifest"
printf '%s  %s\n' 6e766252c8e4ab90ce6c17107cdde353a18c471211fedabeaffa8efecc5fbd20 \
    "$raw_manifest" | sha256sum -c -

# Raw restoration is owned by its independently admitted bounded-cache
# supervisor.  Queues below wait on the verified coverage receipt.

mkdir -p "$ops/logs/downstream"
for rotation in 2 3 4; do
    (
        set +e
        while true; do
            PYTHONPATH="$base/src:$base" "$python" \
                "$ops/scripts/runpod_event_queue_waiter_v1.py" \
                --queue-entrypoint "$runtime/scripts/runpod_event_queue_bootstrap_v2.py" \
                --bundle /opt/ejepa_event_bundle --run "$run" --rotation "$rotation" \
                --assignment "$ops/r${rotation}_downstream/EVENT_DOWNSTREAM_ASSIGNMENT.json" \
                --budget "$ops/EVENT_BUDGET_24USD.json" \
                --exclusion "$ops/r${rotation}_downstream/LOCAL_QUEUE_EXCLUSION.json" \
                --raw-manifest "$raw_manifest" \
                --raw-coverage /workspace/ejepa_event_raw/EVENT_RAW_COVERAGE_RECEIPT.json \
                --supplemental-admission "$supplement" --poll-seconds 15
            status=$?
            [ "$status" -eq 0 ] && exit 0
            [ "$status" -eq 4 ] || exit "$status"
            sleep 60
        done
    ) >"$ops/logs/downstream/R${rotation}.log" 2>&1 &
    echo "$!" >"$ops/EVENT_DOWNSTREAM_R${rotation}.pid"
done
wait
