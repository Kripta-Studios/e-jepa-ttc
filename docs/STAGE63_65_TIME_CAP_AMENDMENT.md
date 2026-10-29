# User-authorized removal of wall-time limits

On 2026-09-05 the user explicitly requested: "a qué te refieres con plazo restante,
si son horas límite para entrenar, eliminalo".

This supersedes only the Stage 63 twelve-hour, Stage 64 seed-7 twenty-four-hour,
shared replication forty-eight-hour and Stage 65 one-hour wall deadlines. Original
budget files remain unchanged as historical evidence, including downtime.

The original TRAINING_COMMIT and TRAINING_LOCK remain immutable scientific
identities. WALL_TIME_AMENDMENT.json separately binds the reviewed execution
commit, exact implementation diff, original lock SHA-256, passing amendment QA,
user request and checkpoint anchors. A changed/absent QA receipt or unexpected
code change fails closed. Unamended invocations retain every original time cap.

No model, loss, optimizer, schedule, normalization, RNG or data implementation is
changed. Endpoints remain fixed at update 3000. RAM (20% available), disk (40 GiB)
and VRAM (2 GiB) margins remain active. Replication/fallback authorization and the
train-all/freeze-all/evaluate-all barrier remain unchanged. No outer scores were
observed before this amendment. This is not authority to rescue failed scores.

The process was stopped for this amendment; the latest complete checkpoint and
progress were copied to resume_snapshots/before_user_time_amendment. Any progress
after that checkpoint is retained as orphaned evidence on normal resume.
