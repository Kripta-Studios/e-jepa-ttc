# Model card — E-JEPA-TTC

Actualizado: 2026-08-09.

## Candidato v5

El candidato arquitectónico vigente es `CausalScaleTTC`, primero event-only. Predice
foreground y altura visible por endpoint, forma `r=log(h_actual/h_previo)` y deriva
`1/TTC=expm1(r)/delta_t`. No acepta bbox, categoría, track o secuencia como inputs.
Un residual aprendido es antisimétrico y acotado; el readout TTC directo es auxiliar
y no alimenta la salida principal.

El artefacto sintético ideal-foreground
`causal_scale_v5_synthetic_operator_gate_v1.json` pasó los gates físicos en commit
`7945e99`. El aprendizaje sintético posterior tiene un checkpoint local seleccionado
solo por validation: Pearson `.9560`, slope `.9686`, sign `.9957`, IoU `.8640` y
TTC symmetric relative error `.2639`. No se promueve porque translation leakage
`.02399` supera el gate `.02`. El test limpio posterior confirmó falta de
generalización: Pearson `.92135` y translation `.02749`; estado
`completed_gate_failed`. No existe ninguna métrica TTC real.

El candidato espacial v6 usa una cabeza separable row/column sin strides. En nueva
validation sintética obtiene translation `.00462` e IoU `.89323`, pero Pearson
`.92042`; permanece no promovido y test 603 está cerrado.

V7 conserva esa máscara e incorpora transporte causal parameter-free entre tres
endpoints. Validation alcanza Pearson `.96126`, TTC `.24345` y translation `.00351`;
todos los gates validation pasan. El test 603 posterior obtiene Pearson `.92014` y
falla el gate `.95`; V7 queda no promovido.

## Estado

El modelo histórico activo era un candidato event-only high-resolution, no un modelo SOTA ni
un sistema de producción. `B0_HISTORICAL_BASE_EXACT` y `A0_MATCHED_GLOBAL` son
anclas EvTTC históricas; no son el checkpoint del trainer Garl nuevo.

No hay checkpoint final promovido. El único run real del trainer nuevo es un smoke
16/16 de integración con MiD macro `1868,3186`, marcado
`claim_eligible=false`.

## Arquitectura activa

```text
eventos causales [B,T,21,H,W]
-> patch embedding
-> atención espacial por ventanas con padding/máscara
-> space-to-depth 2x2 opcional
-> mixer temporal block-causal
-> query pooling
-> cabeza TTC firmada
```

Perfil screen:

```text
320x192, patch 8, dim 32, 4 heads, profundidad 1+1, batch 2
```

Perfil full candidate:

```text
320x192, patch 16, dim 192, 6 heads, profundidad 1+2,
batch 4, acumulación 6, BF16, máximo 30 épocas
```

El perfil full usa todas las filas válidas, seeds 7/13/23, exige Git limpio y
congela el mejor checkpoint únicamente con validation Garl. Entrenamiento por sí
solo no habilita un claim.

## Modalidades

- Event-only: implementada en el trainer raw cache-free.
- Event-only v5: core, dataset, loss, aprendizaje foreground y calibración validation
  implementados; test sintético y datos reales todavía cerrados.
- RGB-E: diseño/config presente, trainer no implementado; falla de forma explícita
  para impedir que RGB sea descartado silenciosamente.
- Bbox/máscaras/depth: solo supervisión u oracle en protocolos declarados; no inputs
  del candidato raw.

## Pretraining

El pretraining JEPA high-resolution compatible todavía está bloqueado. El script
`pretrain_eap_tubelet_jepa.py` rechaza el encoder pooled legacy porque sus tokens y
resolución no son compatibles con el downstream actual. Por tanto el candidato
high-resolution todavía se entrena desde cero salvo que se proporcione un
checkpoint cuya arquitectura pase la comprobación estricta de claves/shapes.

Una auditoría sintética posterior demuestra que el regularizador de varianza y
VISReg pueden conservar un shortcut lento aunque rango/varianza parezcan sanos.
R²-lite no alcanzó el gate TTC y no forma parte del modelo. El residual temporal
es solo una propuesta condicional para un canal `z_delta`; el control frame-varying
demuestra que no debe reemplazar `z_level`. Ninguna de estas pruebas demuestra que
el mismo shortcut exista en eAP.

## Salidas

El contrato general permite:

- TTC medio firmado en segundos;
- log-varianza opcional;
- logits de colisión por horizonte;
- embedding de contexto y embeddings futuros;
- diagnósticos de salud latente.

El trainer Garl nuevo optimiza actualmente la cabeza TTC mediante Smooth L1 sobre
`sign(TTC) * log1p(abs(TTC))`. La selección usa MiD macro por secuencia con targets
firmados.

## Evidencia histórica

| Modelo | Protocolo | Resultado | Decisión |
|---|---|---:|---|
| B0 historical | validation histórica | 8,1554 % RTE | ancla exacta |
| A0 global | grouped CV 5x3 | 30,25 % ± 0,52 RTE | seleccionado en esa matriz |
| A1 Dense | grouped CV 5x3 | 30,55 % ± 0,06 RTE | rechazado |
| R1 bbox-ROI | 5 folds, seed 7 | 30,99 % RTE | rechazado |
| Object-KDA | confirmación matched | 16,960 % RTE | rechazado |

Estas cifras pertenecen a protocolos EvTTC previos y no deben compararse como si
fueran el mismo entrenamiento que el modelo Garl high-resolution.

## Uso previsto

- investigación TTC con cámaras de eventos;
- screens de arquitectura y representación;
- evaluación de transferencia JEPA;
- preparación auditable de candidatos para EvTTC/eAP.

No usar para control de vehículos, decisiones de seguridad ni afirmaciones SOTA
sin evaluación externa reproducida.

## Riesgos y limitaciones

- el smoke high-resolution no aprende todavía una señal TTC competitiva;
- el gate ideal v5 usa foreground analítico; el aprendizaje sintético posterior
  demuestra señal, pero falla todavía la equivariancia de traslación congelada;
- la incertidumbre v5 usa delta method y es frágil cerca de expansión cero;
- falta JEPA denso compatible;
- el predictor SSL real tiene rango efectivo ≈1,10 sin diagnóstico semántico real;
- falta comparar nivel frente a nivel+residual con probes congelados sobre eAP;
- falta RGB-E, modalidad fuerte en Garl-TTC;
- la geometría causal bbox-free/expansión/FoE no supera A0;
- family-OOD degrada materialmente frente a validation;
- seis secuencias eAP y el protocolo oficial completo no están disponibles;
- EvTTC Tabla VI carece aún de manifest label-free real;
- no hay calibración, robustez, latencia end-to-end, ONNX o demo del checkpoint
  final.

## Reproducibilidad

Cada run guarda commit, dirty flag, hashes de config/dataset/split, seed, entorno,
GPU, timestamps, historial, criterio de selección y SHA del checkpoint. Los perfiles
full requieren tres seeds comparables antes del freeze. Predict y score EvTTC son
procesos separados.
