#!/usr/bin/env bash
set -u

ops=/workspace/ejepa_event_operations
runtime="$ops/runtime_v12"
run=/workspace/ejepa_event_runs
python=/opt/ejepa_env/bin/python
bootstrap="$runtime/event_runtime_bootstrap.py"
bundle=/opt/ejepa_event_bundle
base="$bundle/workspace/e-jepa-ttc-v18-joint-measurement"
export PYTHONPATH="$base/src:$base${PYTHONPATH:+:$PYTHONPATH}"
export EJEPA_EVENT_SUPPLEMENT_ROOT=/opt/ejepa_event_supplement
export EJEPA_EVENT_SUPPLEMENT_ADMISSION="$ops/supplement/EVENT_DOWNSTREAM_SUPPLEMENTAL_ADMISSION.json"
export EJEPA_EVENT_AUTHORITY_SUPPLEMENT_ROOT=/opt/ejepa_event_authority_supplement_v2
export EJEPA_EVENT_AUTHORITY_SUPPLEMENT_ADMISSION="$ops/authority_supplement_v2/EVENT_AUTHORITY_SUPPLEMENT_ADMISSION.json"
assignment="$ops/authorization/EVENT_CLOUD_ASSIGNMENT.json"
pause="$ops/handoff_extracted/handoff_authority/LOCAL_EVENT_HANDOFF_PAUSE.json"
mkdir -p "$ops/logs/probes" "$ops/logs/producers"

wait_headroom() {
    local cgroup=/sys/fs/cgroup/memory current limit
    while true; do
        current=$(cat "$cgroup/memory.usage_in_bytes")
        limit=$(cat "$cgroup/memory.limit_in_bytes")
        [ $((limit - current)) -ge 3221225472 ] && return 0
        sleep 15
    done
}

reclaim_if_needed() {
    local receipt="$1" cgroup=/sys/fs/cgroup/memory current limit
    current=$(cat "$cgroup/memory.usage_in_bytes")
    limit=$(cat "$cgroup/memory.limit_in_bytes")
    if [ $((limit - current)) -lt 8589934592 ]; then
        "$python" "$ops/event_cache_reclaim_v1.py" \
            --logical /workspace/ejepa_event_bundle --stage /opt/ejepa_event_bundle \
            --seal /opt/ejepa_event_bundle.SEAL.json --receipt "$receipt"
    fi
}

while ! grep -q 'EVENT_CONTROLLER_READY_FOR_PROBES' "$ops/logs/CONTROLLER_V12.log" 2>/dev/null; do
    controller=$(cat "$ops/CONTROLLER_V12.pid")
    if ! kill -0 "$controller" 2>/dev/null; then
        printf 'Controller exited before probe admission\n'
        exit 2
    fi
    sleep 30
done
reclaim_if_needed "$ops/EVENT_CACHE_RECLAIM_BEFORE_PROBES.json"
wait_headroom

pids=()
labels=()
for rotation in 2 3 4; do
    for fit in E_A5 E_C2F; do
        label="R${rotation}_${fit}"
        extra=()
        if [ "$label" = R2_E_A5 ]; then
            extra=(--pause-receipt "$pause")
        fi
        "$python" "$bootstrap" probe --bundle "$bundle" --run "$run" \
            --rotation "$rotation" --fit "$fit" --assignment "$assignment" \
            "${extra[@]}" >"$ops/logs/probes/${label}.log" 2>&1 &
        pids+=("$!")
        labels+=("$label")
    done
done

while true; do
    alive=0
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then alive=$((alive + 1)); fi
    done
    printf '%s phase=PROBE alive=%s ' "$(date -u +%FT%TZ)" "$alive" >>"$ops/logs/EVENT_RESOURCE_TELEMETRY.log"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
        | tr '\n' ';' >>"$ops/logs/EVENT_RESOURCE_TELEMETRY.log" || true
    printf '\n' >>"$ops/logs/EVENT_RESOURCE_TELEMETRY.log"
    [ "$alive" -eq 0 ] && break
    sleep 2
done

failed=0
for index in 0 1 2 3 4 5; do
    if ! wait "${pids[$index]}"; then
        printf 'PROBE_FAIL %s\n' "${labels[$index]}"
        failed=1
    else
        printf 'PROBE_PASS %s\n' "${labels[$index]}"
    fi
done
[ "$failed" -eq 0 ] || exit 3
reclaim_if_needed "$ops/EVENT_CACHE_RECLAIM_BEFORE_SCIENCE.json"
wait_headroom

pids=()
labels=()
for rotation in 2 3 4; do
    for fit in E_A5 E_C2F; do
        label="R${rotation}_${fit}"
        extra=()
        if [ "$label" = R2_E_A5 ]; then
            extra=(--pause-receipt "$pause")
        fi
        "$python" "$bootstrap" producer --bundle "$bundle" --run "$run" \
            --rotation "$rotation" --fit "$fit" --assignment "$assignment" \
            "${extra[@]}" >"$ops/logs/producers/${label}.log" 2>&1 &
        pids+=("$!")
        labels+=("$label")
    done
done
printf '%s\n' "${pids[@]}" >"$ops/EVENT_PRODUCER_PIDS.txt"

while true; do
    alive=0
    for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then alive=$((alive + 1)); fi
    done
    printf '%s phase=SCIENCE alive=%s ' "$(date -u +%FT%TZ)" "$alive" >>"$ops/logs/EVENT_RESOURCE_TELEMETRY.log"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
        | tr '\n' ';' >>"$ops/logs/EVENT_RESOURCE_TELEMETRY.log" || true
    printf '\n' >>"$ops/logs/EVENT_RESOURCE_TELEMETRY.log"
    [ "$alive" -eq 0 ] && break
    sleep 10
done

failed=0
for index in 0 1 2 3 4 5; do
    if ! wait "${pids[$index]}"; then
        printf 'PRODUCER_NONCOMPLETE %s\n' "${labels[$index]}"
        failed=1
    else
        printf 'PRODUCER_COMPLETE %s\n' "${labels[$index]}"
    fi
done
exit "$failed"
