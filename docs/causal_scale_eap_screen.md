# Causal Scale eAP public validation screen v1

Este screen adapta el mejor brazo sintético V8 a datos públicos eAP/Garl-TTC sin
abrir test privado, CodaBench ni EvTTC test. Es evidencia exploratoria de una seed y
no autoriza un claim SOTA.

## Datos

Se usa el cache firmado
`artifacts/cache/garl_object_event_common_roi_screen_v4/manifest.json`: 2.048 muestras
train de nueve secuencias y 2.048 validation de tres secuencias disjuntas. Contiene
los cuatro buckets TTC firmados. La entrada es event-only `[3,12,128,128]` con tres
endpoints separados 0,1 s y una ROI cuadrada común que preserva escala.

El cache no contiene máscaras. Las cajas oficiales t1/t2 se rasterizan como
rectángulos de supervisión débil mediante `weak_box_masks`; no se consideran
segmentación GT y nunca entran al forward. La caja t0 proxy se excluye.

## Entrenamiento

Config:
`configs/experiment/e_jepa_garl_event_causal_scale_eap_screen_v1.yaml`.

Runner: `scripts/train_causal_scale_eap_screen.py`.

- seed 7;
- máximo 18 épocas;
- BF16, batch 32;
- warm-up foreground 3 épocas;
- CVaR top 10%, peso 2;
- selección validation por MiD macro por secuencia y failure rate;
- early stopping mínimo 8, paciencia 5;
- límite total 6 h;
- checkpoint atómico por época y resume completo.

El benchmark de throughput 128+128 con batch 8 tardó 5,289 s y usó 395,6 MiB de
VRAM. La ejecución completa aún está pendiente.

## Comparación

El baseline primario será el release oficial event-only:

```text
E:\Garl-TTC\configs\ablation\event_lhr.yaml
E:\Garl-TTC\checkpoints\paper_event_only_lhr.pth
```

Debe evaluarse sobre exactamente los mismos 2.048 sample tokens de validation. El
full multimodal se reportará aparte porque no comparte modalidad. La diferencia de
representación —Garl 2×20 canales frente a nuestro 3×12— debe permanecer visible.

## Estado

Implementación y checks verdes. Resume coincide exactamente con entrenamiento
continuo en modelo, optimizer, scheduler, RNG, historial y best, y rechaza contratos
distintos. `scripts/build_garl_validation_subset_from_predictions.py` construye y
firma los parquets exactos, validando tokens, joins, targets y roundtrip. Falta ejecutar
el entrenamiento completo, el checkpoint oficial y la comparación firmada.
