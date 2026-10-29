#!/usr/bin/env bash
set -eu

logical=/workspace/ejepa_event_bundle
stage=/opt/ejepa_event_bundle
runtime=/workspace/ejepa_event_operations/runtime_v12
run=/workspace/ejepa_event_runs
ops=/workspace/ejepa_event_operations
python=/opt/ejepa_env/bin/python
base="$stage/workspace/e-jepa-ttc-v18-joint-measurement"
export PYTHONPATH="$base/src:$base${PYTHONPATH:+:$PYTHONPATH}"
export EJEPA_EVENT_SUPPLEMENT_ROOT=/opt/ejepa_event_supplement
export EJEPA_EVENT_SUPPLEMENT_ADMISSION=/workspace/ejepa_event_operations/supplement/EVENT_DOWNSTREAM_SUPPLEMENTAL_ADMISSION.json
export EJEPA_EVENT_AUTHORITY_SUPPLEMENT_ROOT=/opt/ejepa_event_authority_supplement_v2
export EJEPA_EVENT_AUTHORITY_SUPPLEMENT_ADMISSION=/workspace/ejepa_event_operations/authority_supplement_v2/EVENT_AUTHORITY_SUPPLEMENT_ADMISSION.json
expected_seals=8586

while true; do
    seals=$(find "$logical" -type f -name '*.transfer.json' | wc -l)
    partials=$(find "$logical" -type f -name '*.partial' | wc -l)
    printf '%s seals=%s/%s partials=%s\n' "$(date -u +%FT%TZ)" "$seals" "$expected_seals" "$partials"
    if [ "$seals" -eq "$expected_seals" ] && [ "$partials" -eq 0 ]; then
        break
    fi
    if [ "$seals" -gt "$expected_seals" ]; then
        printf 'Transfer seal count escaped manifest\n'
        exit 2
    fi
    sleep 60
done

if [ ! -f /opt/ejepa_event_bundle.SEAL.json ] || [ ! -f /opt/ejepa_event_bundle.READY.json ]; then
    "$python" "$runtime/event_runtime_bootstrap.py" stage-bundle stage \
        --logical-bundle "$logical" --stage "$stage" --reserve-bytes 12884901888
    "$python" "$runtime/event_runtime_bootstrap.py" seal-bundle create \
        --root "$stage" --readiness /opt/ejepa_event_bundle.READY.json \
        --output /opt/ejepa_event_bundle.SEAL.json
else
    "$python" "$runtime/event_runtime_bootstrap.py" seal-bundle verify \
        --root "$stage" --seal /opt/ejepa_event_bundle.SEAL.json
fi
while [ ! -f "$EJEPA_EVENT_AUTHORITY_SUPPLEMENT_ADMISSION" ]; do sleep 5; done

# Initial transfer/staging can charge almost the whole cgroup as inactive file
# cache. At zero scientific updates, advise away only the two verified bundle
# mirrors, then wait for a real margin before constructing CUDA/source state.
cgroup=/sys/fs/cgroup/memory
current=$(cat "$cgroup/memory.usage_in_bytes")
limit=$(cat "$cgroup/memory.limit_in_bytes")
if [ $((limit - current)) -lt 8589934592 ]; then
    "$python" /workspace/ejepa_event_operations/event_cache_reclaim_v1.py \
        --logical "$logical" --stage "$stage" \
        --seal /opt/ejepa_event_bundle.SEAL.json \
        --receipt /workspace/ejepa_event_operations/EVENT_CACHE_RECLAIM.json
fi
while true; do
    current=$(cat "$cgroup/memory.usage_in_bytes")
    limit=$(cat "$cgroup/memory.limit_in_bytes")
    if [ $((limit - current)) -ge 3221225472 ]; then
        break
    fi
    sleep 15
done

"$python" "$runtime/event_runtime_bootstrap.py" prepare --bundle "$stage" --run "$run"

mkdir -p "$ops/authorization" "$ops/handoff_extracted"
for name in EVENT_EXECUTION_AUTHORIZATION.json EVENT_LOCAL_QUEUE_EXCLUSION.json POD2_BUDGET_ACTIVE.json; do
    source="$runtime/artifacts/runpod_event_migration_20261026_v4/$name"
    target="$ops/authorization/$name"
    if [ -f "$target" ]; then
        cmp -s "$source" "$target" || { printf 'Authorization differs: %s\n' "$name"; exit 2; }
    else
        cp "$source" "$target"
    fi
done
"$python" "$runtime/event_runtime_bootstrap.py" bind-assignment \
    --bundle "$stage" --run "$run" \
    --authorization "$ops/authorization/EVENT_EXECUTION_AUTHORIZATION.json" \
    --assignment "$ops/authorization/EVENT_CLOUD_ASSIGNMENT.json"

printf '%s  %s\n' 242879395bf4463a9748d26b7a70961b46535319832b5853d69fdcdfc17c557e \
    /workspace/E_A5_R2_HANDOFF.tar | sha256sum -c -
tar -xf /workspace/E_A5_R2_HANDOFF.tar -C "$ops/handoff_extracted"
"$python" "$runtime/event_runtime_bootstrap.py" install-handoff \
    --bundle "$stage" --run "$run" --handoff-root "$ops/handoff_extracted"

for rotation in 2 3 4; do
    for fit in E_A5 E_C2F; do
        "$python" "$runtime/event_runtime_bootstrap.py" preflight \
            --bundle "$stage" --run "$run" --rotation "$rotation" --fit "$fit"
    done
done
printf 'EVENT_CONTROLLER_READY_FOR_PROBES %s\n' "$(date -u +%FT%TZ)"
