# External detection protocol evidence v2

Audit date: `2026-10-29`. Public metadata only.

The official public route is CodaBench competition `17291` with eight
classes, center-distance thresholds 0.5/1/2/4 m, and leaderboard columns AP/ATE/ASE/AOE.
The scoring program is private; only its public metadata size (9,821 bytes) was observed.
No scoring-program, reference-data, or private-label bytes were requested.

The public results-admin endpoint returned 403 and the public leaderboard endpoint returned
an empty submission list. Therefore this audit verifies the contract but not a leader, score,
rank, or equivalence of the local evaluator to the private scorer.

The official train list describes 46 IDs, while every inspected release-catalog revision
contains 40 assets. The six absent IDs are `AUIkq1ZJRv, CEOFFwY7c8, F6XHHaiwJg, OwA5qqkewc, mXRa2Ux9ok, uXfPJNOXVY`.
They have categorical `dataset_info` rows, but no catalog media URL, size, hash, label file,
or evidence of acquisition independence. They are not usable data in the current campaign.

Regenerate with bounded public GETs:

```powershell
python scripts/audit_detection_external_protocol.py
python scripts/audit_detection_external_protocol.py --check
```
