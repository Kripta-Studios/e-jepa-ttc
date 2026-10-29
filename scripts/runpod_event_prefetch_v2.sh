#!/usr/bin/env bash
set -u
mkdir -p /opt/ejepa_event_bundle
while ! test -f /opt/ejepa_event_bundle.SEAL.json; do
    free=$(df -B1 --output=avail /opt | tail -1 | tr -d ' ')
    if [ "$free" -lt 12884901888 ]; then
        printf 'STOP: local free space below 12 GiB reserve\n'
        exit 3
    fi
    rsync -a --ignore-existing \
        --exclude='*.partial' \
        --exclude='*.transfer.json' \
        --exclude='.*.transfer.json*' \
        --exclude='__pycache__/' \
        --exclude='*.pyc' \
        /workspace/ejepa_event_bundle/ /opt/ejepa_event_bundle/
    status=$?
    if [ "$status" -ne 0 ] && [ "$status" -ne 24 ]; then
        printf 'STOP: rsync status %s\n' "$status"
        exit "$status"
    fi
    date -u
    sleep 30
done
printf 'Local immutable stage sealed; prefetch worker finished\n'
