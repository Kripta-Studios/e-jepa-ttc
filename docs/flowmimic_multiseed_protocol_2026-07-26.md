# Protocolo congelado E0/E1 multisemilla — 2026-07-26

## Objetivo

Determinar si el alineamiento físico sintético de E1 mejora de forma estable al
JEPA limpio E0. Este gate selecciona una hipótesis en validación; no abre un test
final, no reproduce Garl-TTC y no autoriza una afirmación SOTA.

## Recursos cerrados

- Configuración:
  `configs/experiment/flowmimic_e0_e1_multiseed.yaml`.
- Cache físico train+validation:
  `artifacts/features/evttc_trainval_v2_voxel_160x90_b5_raw_meta_nav.npz`.
- SHA-256 del cache:
  `22d3ef27018925aae62825f0a7f51d1420ae93cacf59aeb18b04758f5a35e88a`.
- Train: 3,019 ventanas de siete secuencias completas.
- Validación: 475 ventanas de `CCRs-side-high`.
- Test: cero ventanas en el NPZ.
- `CPLA-high`: cerrado y ausente físicamente.
- Protocolo v3 SHA-256:
  `e15b81535ea4698b74f45553cd46f26986667c185015d4671ad41a13a0c3bdf1`.

## Matriz exacta

| Variante | Alineamiento FlowMimic | Auxiliary inverse-TTC |
| --- | ---: | ---: |
| E0 | `0.0` | `0.0` |
| E1 | `0.25` | `0.0` |

Semillas emparejadas: `7`, `13` y `21`. Cada semilla se usa tanto en SSL como
en su fine-tuning correspondiente; no se reutiliza un único encoder para tres
cabezas.

Parámetros comunes:

- SSL: 30 épocas, batch 12, AdamW, LR `3e-4`;
- downstream: 30 épocas, batch 24, full fine-tuning, LR `3e-5`;
- encoder: `event-tubelet-transformer`;
- predictor denso: transformer;
- máscara tubelet `0.45`, cuatro bloques;
- horizontes: `20/60/100/240/500 ms`;
- EMA `0.99`, variance regularization `1.0`, `min_std=0.05`;
- selección SSL por loss real de validación;
- selección downstream por MAE TTC de validación;
- evaluación downstream limitada a train y validation.

## Evidencia guardada

Cada `predictions.npz` nuevo incluye:

```text
<split>_pred
<split>_true
<split>_global_index
<split>_sequence_id
<split>_timestamp_us
<split>_context_start_us
<split>_context_end_us
```

Esto permite comprobar que E0 y E1 se comparan sobre las mismas ventanas y
agrupar por secuencia. El resumen final contiene:

- media y desviación de MAE entre semillas;
- diferencias emparejadas E1−E0 por semilla;
- bootstrap emparejado entre semillas;
- bootstrap emparejado por secuencia;
- salud de embeddings;
- hashes físicos de métricas, predicciones y checkpoints;
- firma del artefacto y estado explícito de apertura del test.

La validación actual solo tiene una secuencia. Por tanto, el bootstrap por
secuencia debe devolver `degenerate_single_sequence`; no se reinterpretará como
un intervalo favorable. Incluso si E1 gana en las tres semillas, esta limitación
impide que el resultado sea evidencia final promocionable.

## Robustez

Las corrupciones se aplican a los eventos HDF5 y después se reconstruye el voxel
de validación. No se corrompe un tensor voxel ya materializado. La matriz tiene
22 condiciones no limpias:

- event dropout `0.1/0.3/0.5/0.7`;
- jitter temporal `50/200/1000 us`;
- eventos de fondo `0.01/0.05/0.1`;
- hot pixels `0.001/0.005`;
- dead pixels `0.01/0.05`;
- supresión de cada polaridad;
- escala de ventana temporal `0.5/0.75/1.25/1.5`;
- crop espacial central `0.9/0.75`.

Se informa degradación absoluta y relativa para cada checkpoint. La cabeza
downstream actual es determinista, así que el comportamiento de incertidumbre
se registra como no disponible en vez de inventarse.

El gate de robustez exige que E1 conserve menor MAE medio que E0 en todas las
condiciones y que su degradación relativa no supere la de E0 en más de cinco
puntos porcentuales. Se guardan también todos los valores, no solo el booleano.

## Ejecución

Desde un commit limpio:

```powershell
uv run --no-sync python scripts/run_flowmimic_multiseed.py `
  --config configs/experiment/flowmimic_e0_e1_multiseed.yaml `
  --with-robustness
```

El runner valida el hash del cache, el protocolo, la ausencia física de test y
la secuencia de validación antes de reservar GPU. Puede reanudar etapas
terminadas solo si commit, cache, semilla, épocas, pesos y hashes de checkpoint
coinciden.

Smoke real previo al gate:

- un window de `CCRs-side-high`, y ningún sample de test;
- event dropout `0.1`, seed `20260726`;
- eventos de entrada/salida: `1,091,096 / 982,375`;
- cache v2 resultante: `[1,21,90,160]`, SHA-256
  `b4a0697c64fa71ca53f5c5888205d01047317a1e046be0486173e9c08de9ce4d`;
- evaluación CUDA del checkpoint E1 completada con
  `final_test_opened=false`;
- predicción guardada con índice, secuencia, timestamp y límites de contexto;
- bootstrap marcado correctamente como `degenerate_single_sequence`.

El MAE de ese único window no se usa como resultado de robustez; el smoke solo
demuestra que la ruta HDF5 → corrupción → voxel → checkpoint → identidad funciona.

## Criterio de decisión

E1 se considera una hipótesis estable local si:

1. reduce el MAE puntual frente a E0 en las tres semillas;
2. la diferencia emparejada media entre semillas es favorable;
3. no presenta colapso de embeddings;
4. no muestra una fragilidad material nueva en robustez.

La promoción científica requiere además varias secuencias de validación
independientes, un holdout realmente no inspeccionado y reproducción bajo el
protocolo oficial EvTTC/eAP.
