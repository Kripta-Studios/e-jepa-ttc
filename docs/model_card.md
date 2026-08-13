# Model card — E-JEPA-TTC

## Scientific Recovery V5

No hay modelo promovido. A8.0 usa geometry A4 congelada y un encoder transport
separado, event-only, causal-left, radius 1, τ 0,02 y residual bound 0,05. En
outer-dev de nueve secuencias obtiene MiD 197,69, failure 7,02%, Pearson 0,225 y
coverage 92,98%. Mejora A6 (211,51), pero falla el gate MiD≤175 y pierde frente a
Garl fold-local desde cero (144,35). Geometry es tensor-a-tensor idéntica a su
parent y la causalidad prefix pasa. No es un candidato final, SOTA ni apto para
seguridad. Véase [el estado canónico](SCIENTIFIC_RECOVERY_V5_STATUS.md).

## CausalScale A0/A1 public screen status

The seed-7 event-only A0 screen is a negative result, not a promoted checkpoint:
validation sequence-macro MiD is 382.19, log-ratio Pearson is 0.046 and weak-box
IoU is 0.500. It must not be described as competitive, production-ready or SOTA.
The official release result remains exposure-confounded; it is not the matched baseline.

Matched training now exists and is the primary local baseline: official Garl
event-only trained from scratch on exact 2,048/2,048 rows obtains sequence-macro
MiD 203.63 with zero failures. A0 obtains 382.19 with 12.30% failures and loses on
all three sequences. This does not make the local Garl run an official result or a
SOTA result. It is a controlled single-seed architecture screen.

Failure decomposition shows that bbox scale change contains target signal
(Pearson 0.760), while the model's analytic foreground extent change does not
(Pearson 0.037 against physical ratio; 0.015 against bbox ratio). Event support is
not limiting. The physical inverse is behaving as specified but magnifies bad
near-zero ratios, yielding 252 unknown outputs and 151 known clipped outputs. This
model should not be used where finite, calibrated TTC is required.

Actualizado: 2026-08-10.

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

V8 evalúa consenso temporal fijo sobre logits de foreground con kernel simétrico
`[w,1-2w,w]`. Es reversible, no aumenta parámetros y el endpoint actual solo usa el
contexto disponible. CVaR top 10% mejora Pearson validation multigrupo a `.94621`,
pero falla el gate `.95`; test 901/902/903 permanece sellado.

La adaptación eAP/Garl causal-scale v1 completó A0 sobre un cache público
2.048/2.048 y produjo el resultado negativo descrito arriba. Las cajas t1/t2 son weak supervision
del foreground, no inputs ni máscaras GT; t0 proxy se excluye. El baseline comparable
es Garl event-only entrenado desde cero con el mismo presupuesto; el release
oficial expuesto se conserva en una tabla distinta.

La métrica principal que un sucesor event-only debe batir en el screen 2.048 es
MiD macro `203.6341709`, con failure no mayor que `0%`. También debe abordar el
defecto que esa cifra agregada oculta: el baseline matched emite solo TTC positivo
y obtiene MiD `437.5957` en negative/receding.

El candidato A1 no es una arquitectura nueva: amplía la observación matemática del
mismo mapa para medir también anchura y centro, sin añadir parámetros ni inputs.
Solo cambia la loss de supervisión foreground. Seed 7 mejora A0 (MiD macro
`346.83` frente a `382.19`), pero sigue lejos de Garl matched (`203.63`) y no se
promueve. La correlación de altura estática sube a `.471`, mientras anchura y centros
quedan en `<=.079`; el cambio temporal de altura correlaciona solo `.059` con el
cambio bbox. Esto apunta a limitación de representación/localización event-native y
coherencia temporal, no solo a ruido del target weak-box.

La auditoría firmada por endpoint refuerza ese diagnóstico sin abrir test. La
anchura target conserva std cercana a `.095`, pero su correlación predicha es solo
`.048/.105` en t1/t2. La actividad absoluta de entrada ocupa casi todo el ROI y no
reproduce el cambio de escala. El siguiente candidato es por tanto una ablation
controlada del decoder foreground 2-D full-resolution, no un modelo promovido.

La repetición válida de esa ablation A1-FR seleccionó epoch 11/16 con cobertura
finita 3/3 y también es negativa: MiD macro `380.2202`, failure `28.7598%`, known
coverage `.7124` y Pearson log-ratio `-.0181`. Empeora A1 en `33.3908` MiD y
`18.7988` puntos de failure. La cabeza full-resolution conserva 2-D, pero consume
eventos crudos; este resultado rechaza esa intervención superficial, no el uso de
features 2-D profundas. El siguiente control aislado usará el decoder existente
`resize_conv`, alimentado por `_EndpointEncoder.features`, con la misma loss A1.
Este control se denomina A1-DF y fue preregistrado con 355.118 parámetros antes de
observar resultados.

A1-DF terminó con MiD macro `350.3020`, failure `21.0938%` y Pearson log-ratio
`.1865`. Mejora la señal geométrica temporal frente a A1, pero empeora sus métricas
de decisión y queda lejos de Garl matched. No es un checkpoint promovido ni apto
para inferencia fiable: el `21.09%` de validation queda unknown.

El sucesor A1-DF-R no cambia arquitectura ni inputs. Añade una loss training-only
sobre el cambio de altura bbox numérico después del warm-up y fue preregistrado
antes de observar métricas.

A1-DF-R terminó con macro `349.8628`, failure `19.8242%` y ratio `.1703`. La
mejora frente a A1-DF es pequeña y concentrada en una secuencia; queda no promovido
y no es adecuado para inferencia fiable.

No hay máscaras oficiales locales utilizables: una auditoría exhaustiva de train
resolvió `0/64.629` rutas únicas declaradas. SAM ViT-L y DINOv3 ConvNeXt-Tiny sí
están íntegros en caché, y todos los RGB públicos declarados están presentes. Si se
usa SAM/DINO como teacher, el modelo deja el protocolo event-only puro aunque su
inferencia solo reciba eventos; se etiquetará explícitamente como
`event-only inference with RGB distillation`. Teacher RGB, bbox prompt y máscaras
solo pueden intervenir durante train, nunca durante validation/test.

El cache SAM train-only final conserva 3.204/4.096 máscaras y 1.602/2.048 pares.
No cambia la modalidad de inferencia, pero sí la información usada para entrenar.
A3 reporta esta cobertura y usa geometry-only para las
filas filtradas. Añade BCE `1.0` y Dice `.5` sin sweep sobre A1, sin cambiar sus
344.591 parámetros. El checkpoint seed 7 existe pero no se promueve: macro
`353.6351`, failure `10.8887%`, peor que A1 en las tres secuencias. Se reporta
separado de event-only puro como `event-only inference with RGB distillation`.

## Estado

El modelo histórico activo era un candidato event-only high-resolution, no un modelo SOTA ni
un sistema de producción. `B0_HISTORICAL_BASE_EXACT` y `A0_MATCHED_GLOBAL` son
anclas EvTTC históricas; no son el checkpoint del trainer Garl nuevo.

No hay checkpoint final promovido. A0 y A1 son screens reales completos pero
negativos; el smoke 16/16 histórico sigue siendo solo integración y permanece
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

- Los runs A0 y A1 son screens completos de una sola seed/2.048 filas, no resultados
  multisemilla ni entrenamiento suficiente para promoción.
- El ROI usa cajas oficiales. A0 rasteriza cajas t1/t2 como weak supervision; A1
  usa `h,w,cx,cy` escalares. Ninguno es bbox-free ni segmentation-supervised.
- Los comparadores A0/Garl y A1/Garl sobre los mismos 2.048 tokens están completos;
  ambos favorecen claramente a Garl matched.
- Resume pasa equivalencia end-to-end; los runs completos no necesitaron reanudarse.

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
