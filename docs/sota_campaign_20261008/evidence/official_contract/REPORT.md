# Official EvTTC comparison contract

Metadata review is complete. Exact official reproduction and independence certification remain blocked by six technical contracts in [the protocol](../../../data/protocols/evttc_official_comparison_v2.json), not by missing campaign authorization. No benchmark payload, Stage76, or sealed labels were opened.

Garl Table VI reports **three** scenarios: CCRs2-medium, CCRs2-high and CCRm-medium, with RTE 8.31%, 10.56% and 12.93% (mean 10.60%). The paper describes eAP training and transfer without EvTTC fine-tuning. Its pointwise RTE is absolute TTC error divided by absolute ground-truth TTC. This does not specify all query, invalid-data and aggregation choices required to reproduce the table. [Garl/eAP paper, Table VI and VII-A/C1](https://arxiv.org/html/2603.16303v1).

The competition has **ten** named sequences, including two Slider sequences absent from the local dev32 manifest. Its page distinguishes 8-mm/16-mm cameras; naming similarity does not prove which original recording or temporal clip an official file contains. [Competition metadata](https://nail-hnu.github.io/EvTTC/competition/). EvTTC describes ground-truth TTC from optical-axis depth/relative velocity, with the reference geometry expressed in the 8-mm RGB camera frame. That does not identify Garl's input camera. [EvTTC paper](https://arxiv.org/html/2412.05053v2).

REACT uses an in-domain EvTTC validation setting and full-field event input; its Table II lists Garl at 9.44%, whereas Garl Table VI lists 10.60%. These numbers cannot be merged without reconciling the split and query contracts. REACT's ROI-crop ablation is a separate setting. [REACT paper](https://arxiv.org/html/2609.19204v1).

Current public HF/GitHub revisions remain unchanged from the earlier audit. The public lists contain 40 and 46 training IDs respectively; the six extra IDs remain training-listed. HF identifies `paper_ours_full.pth` as primary, but the inspected model card and release metadata do not bind its digest to Table VI or certify complete training/selection ancestry. Hash identity is established; training exclusion is not.

The protocol records candidate dev32 matches by scenario/speed solely to expose possible overlap. It deliberately leaves physical recording overlap and independent groups unresolved. The historical local CV declares `development_model_selection`; already inspected development recordings cannot become a new blind holdout through renaming or cropping.

The executable RTE scorer is ready and retains all missing-GT rows, reports coverage and avoids prediction-based cohort selection. Its explicit local policy is not presented as an official scorer. Resolve recording/query identity, camera/ROI, temporal/GT/scorer rules, checkpoint-to-table binding, training ancestry and independence/freeze before claiming official reproduction. Independent local exploratory branches need not stop.

`AUTHORS_DRAFT.txt` contains four concise questions, adapted from the previous unsent draft. Nothing was contacted or submitted. `SOURCES.json` records retrieval dates, URLs, revisions, byte counts and hashes for public documents only.
