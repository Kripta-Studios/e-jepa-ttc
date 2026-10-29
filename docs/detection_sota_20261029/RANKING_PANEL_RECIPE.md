# Receta de paneles R0 para S2

Estado: arnés preparado; **inferencia y ajuste no ejecutados**. Debe usarse únicamente
cuando R0 tenga un checkpoint seleccionado e inmutable.

## Poblaciones congeladas

- `train4096`: los 4.096 tokens TRAIN del manifiesto D4096.
- `selection1024`: los 1.024 tokens de selección natural, ya expuestos a desarrollo.

`plan` verifica el hash del checkpoint, el manifiesto de campaña, el split, el manifiesto
de archivos públicos y `data/train.parquet`. También demuestra que los prefijos de token y
las secuencias declaradas pertenecen a su rol. En esta fase no abre etiquetas.

```powershell
python scripts/prepare_detection_ranking_panels.py plan `
  --campaign-manifest output/campaign_data/d4096_v5_readable/d4096_manifest.json `
  --population-audit output/campaign_data/d4096_v5_readable/population_audit.json `
  --split data/splits/eap_detection_candidate_v1.json `
  --public-files-manifest data/manifests/eap_detection_public_files_v1.json `
  --checkpoint <R0_SELECTED_CHECKPOINT> `
  --data-root E:/eAP_dataset `
  --output-dir output/detection_ranking/r0_panels_v1
```

El comando produce `train4096_tokens.json`, `selection1024_tokens.json` y
`ranking_panel_recipe.json`. Los JSON de tokens contienen exclusivamente `sample_tokens`,
el contrato que acepta `infer_detection_campaign.py`. La resolución fuente 1280×720 se
hereda del release y se liga al `image_geometry` de la auditoría original, que decodificó
un miembro RGB por secuencia. La admisión v5 posterior comprueba los 28.054 miembros
del conjunto train/selección completa; su recibo queda ligado al productor R0. La receta
de paneles conserva la procedencia original de geometría y no sustituye ese recibo.

## Inferencia label-free

Ejecutar cada panel en un proceso fresco. El sidecar observa únicamente tensores ya
expuestos por el forward congelado.

```powershell
python scripts/infer_detection_campaign.py `
  --checkpoint <R0_SELECTED_CHECKPOINT> `
  --data-root E:/eAP_dataset `
  --tokens output/detection_ranking/r0_panels_v1/train4096_tokens.json `
  --output output/detection_ranking/r0_train4096_predictions.json `
  --diagnostic-sidecar output/detection_ranking/r0_train4096_sidecar.json

python scripts/infer_detection_campaign.py `
  --checkpoint <R0_SELECTED_CHECKPOINT> `
  --data-root E:/eAP_dataset `
  --tokens output/detection_ranking/r0_panels_v1/selection1024_tokens.json `
  --output output/detection_ranking/r0_selection1024_predictions.json `
  --diagnostic-sidecar output/detection_ranking/r0_selection1024_sidecar.json
```

## GT independiente posterior

`build-gt` se niega a abrir etiquetas hasta verificar que predicciones y sidecar:

- usan el checkpoint fijado en la receta;
- cubren exactamente el panel;
- declaran `targets_argument=null` y cero anotaciones devueltas;
- están ligados entre sí por SHA-256;
- declaran observación externa sin acceso a GT.

Después lee solo las filas solicitadas, valida cada parquet contra el manifiesto público y
construye `phase_ttc_detection_ground_truth_bundle_v2` mediante
`build_evaluation_ground_truth`. El bundle incluye `sequence_id_by_token`, requerido para
probar propiedad TRAIN en el ajuste.

```powershell
python scripts/prepare_detection_ranking_panels.py build-gt `
  --recipe output/detection_ranking/r0_panels_v1/ranking_panel_recipe.json `
  --panel train4096 `
  --predictions output/detection_ranking/r0_train4096_predictions.json `
  --sidecar output/detection_ranking/r0_train4096_sidecar.json `
  --data-root E:/eAP_dataset `
  --split data/splits/eap_detection_candidate_v1.json `
  --public-files-manifest data/manifests/eap_detection_public_files_v1.json `
  --output output/detection_ranking/r0_train4096_gt_v2.json `
  --receipt output/detection_ranking/r0_train4096_gt_receipt.json
```

Repetir con `--panel selection1024` solo para evaluación. Ese GT no puede ajustarse: aunque
se pase literalmente `--dataset-role train`, `train_detection_ranker.py` comprueba la
propiedad canónica contra `split.assignments.train` y falla cerrado.

## Ajuste y aplicación

```powershell
python scripts/train_detection_ranker.py fit `
  --predictions output/detection_ranking/r0_train4096_predictions.json `
  --sidecar output/detection_ranking/r0_train4096_sidecar.json `
  --ground-truth output/detection_ranking/r0_train4096_gt_v2.json `
  --panel-recipe output/detection_ranking/r0_panels_v1/ranking_panel_recipe.json `
  --ground-truth-receipt output/detection_ranking/r0_train4096_gt_receipt.json `
  --split-manifest data/splits/eap_detection_candidate_v1.json `
  --dataset-role train `
  --output output/detection_ranking/r0_ranker_v1.json

python scripts/train_detection_ranker.py apply `
  --predictions output/detection_ranking/r0_selection1024_predictions.json `
  --sidecar output/detection_ranking/r0_selection1024_sidecar.json `
  --bundle output/detection_ranking/r0_ranker_v1.json `
  --output output/detection_ranking/r0_selection1024_ranked.json
```

Los controles baseline, incertidumbre escalar y temperatura monótona deben materializarse
antes de abrir las métricas de selección. Ninguno de estos comandos autoriza test,
calibración, audit, GPU durante preparación ni ajuste previo a la selección de R0.
