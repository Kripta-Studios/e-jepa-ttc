# Scientific Recovery V8 training status

The live, signed monitor snapshot is written outside the tracked worktree at
`artifacts/scientific_recovery_v8/monitor/TRAINING_STATUS.md`. This keeps a
running queue from changing the source revision recorded by its jobs.

After the queue has stopped, an operator may explicitly run the PowerShell
orchestrator with `-PublishTrainingStatus` to copy the final signed local
snapshot here for review. That action is intentionally not automatic.

Public validation, private test, EvTTC test and CodaBench remain sealed.
