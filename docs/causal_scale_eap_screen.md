# Causal Scale eAP public validation screen v1

## Resultado A0 seed 7

| Métrica validation pública | Valor |
|---|---:|
| MiD global | 383.8549432 |
| MiD macro por secuencia | 382.1905104 |
| failure rate | 12.3046875% |
| known coverage | 0.876953125 |
| log-ratio Pearson | 0.0456290990 |
| log-ratio MAE | 0.0289358757 |
| weak-box IoU | 0.4997858107 |
| peak VRAM | 1557.73 MiB |
| tiempo total | 541.49 s |

La cola finita top 10% contiene 180 filas: 140 `crucial`, 22 `small`, 14 `large`
y 4 `negative`; DGq aporta 91, pBq 48 y qooh 41. La señal de escala es casi nula
y la localización débil. A0 no se promueve; A1 geometry-only es la siguiente
diferencia única.

### Descomposición exacta del fallo

El artifact firmado
`artifacts/metrics/causal_scale_eap_a0_failure_decomposition_v1.json` (identidad
`75918c58cd91258fac5aac11f8d6fca00ce6cf43014e5ee19ab3a30d7c91beb7`)
evalúa el checkpoint seleccionado en BF16 y separa cada etapa:

| Relación | Pearson | Pendiente | MAE log-ratio |
|---|---:|---:|---:|
| bbox expansion vs ratio físico TTC | 0.759753 | 0.862189 | 0.013785 |
| altura predicha vs altura bbox (log) | 0.372040 | 0.170747 | 0.181595 |
| ratio analítico del mapa vs bbox | 0.014517 | 0.019786 | 0.038536 |
| ratio analítico del mapa vs TTC | 0.036820 | 0.056951 | 0.037289 |
| residual aprendido vs TTC | -0.033979 | -0.023099 | 0.029725 |
| analítico + residual vs TTC | 0.037676 | 0.033852 | 0.029968 |
| ratio efectivo tras consenso vs TTC | 0.045641 | 0.035179 | 0.028937 |

La expansión de las cajas sí es consistente con el target físico, por lo que el
fallo no se explica por ausencia de señal en la supervisión. El encoder de
foreground reproduce débilmente la altura absoluta y no reproduce su derivada
temporal. El residual tampoco recupera la dinámica; el consenso solo deja el mismo
ratio casi no correlacionado. Esto localiza el fallo observado en la etapa
`eventos -> extensión foreground temporal`, antes de la conversión física.

Los 252 `unknown` coinciden exactamente con 252 ratios de par bajo
`|r| < 0.002`. No hubo ningún caso bajo el gate de soporte: support mínimo
`0.168996`, frente al umbral `0.0001`. Por tanto no es falta de eventos. Entre las
predicciones conocidas, 151 llegan al clip de ±60 s. Esto es el efecto esperado de
invertir un ratio pequeño o erróneo: `TTC = 1 / (expm1(r) / dt)` amplifica el error
cerca de `r=0`; la singularidad agrava el fallo, pero no lo origina.

### Qué está demostrado y qué sigue siendo hipótesis

Está demostrado en validation pública que el cuello de botella es la dinámica de
extensión aprendida; no están demostradas todavía las causas internas de esa mala
extensión. La hipótesis A1 es que BCE/Dice contra una weak-box rectangular —que
incluye fondo interior— permite optimizar solapamiento sin aprender bordes y
extensiones temporales estables. Solo una ablation A1, que quite BCE/Dice y
supervise altura/anchura/centro con todo lo demás fijo, puede confirmar o refutar
esa explicación. No se atribuye causalidad a A1 antes de ejecutar ese control.

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
VRAM. Era una medición previa; A0 ya terminó y sus resultados seleccionables son
los de la tabla de apertura.

## Comparación

El baseline primario será el release oficial event-only:

```text
E:\Garl-TTC\configs\ablation\event_lhr.yaml
E:\Garl-TTC\checkpoints\paper_event_only_lhr.pth
```

Debe evaluarse sobre exactamente los mismos 2.048 sample tokens de validation. El
full multimodal se reportará aparte porque no comparte modalidad. La diferencia de
representación —Garl 2×20 canales frente a nuestro 3×12— debe permanecer visible.

### Garl event-only matched seed 7: baseline principal a batir

Se entrenó desde cero el `TTCNetwork` y la loss/optimizer oficiales del release
auditado `256661242b8a7f5e56aa3c1c02348b30f6e89de6`. No se cargó ningún checkpoint
release. Usó exactamente las mismas 2.048 filas train y 2.048 validation que A0,
con secuencias disjuntas, seed 7, batch 32, FP32, máximo 18 épocas y selección
validation-only desde época 8 por MiD macro y failure rate. Early stopping terminó
en época 16 y seleccionó época 11.

Secuencias train exactas: `2cyv0Oedzg`, `5ilM1PX2vz`, `6h5yRW2LGc`,
`OBneIVg4Cw`, `OYgB6RGWcq`, `WbCh1DRerJ`, `mHGFBekt7X`, `qGsgzl4Q8B` y
`t79dBxj1WS`. Validation exacta: `DGqicHUGWb`, `pBqGOb2vYq` y `qoohcdtLDH`.
El manifest firmado comprueba que los dos conjuntos de secuencias son disjuntos.

| Métrica validation | Garl matched | A0 |
|---|---:|---:|
| MiD global | 203.0982270 | 383.8549431 |
| MiD macro-secuencia | 203.6341709 | 382.1905103 |
| failure rate | 0% | 12.3046875% |
| known coverage | 1.0 | 0.876953125 |
| log-ratio Pearson | 0.3722129 | 0.0506560 |
| log-ratio slope | 0.2099377 | 0.0422964 |
| sign accuracy | 0.8364258 | 0.5334076 |

Garl matched por secuencia: `DGqicHUGWb=219.8851`, `pBqGOb2vYq=217.1081`,
`qoohcdtLDH=173.9093`. A0 pierde en las tres. Por bucket, Garl obtiene crucial
`219.1689`, small `107.0243`, large `176.4692` y negative `437.5957`, todos sin
failure. A0 obtiene respectivamente `529.2508`, `265.8515`, `184.5973` y
`210.1439`, pero negative tiene `20%` failures. Las 2.048 predicciones matched son
positivas (`1.0117–6.5083 s`), de modo que Garl no resuelve receding pese a su mejor
MiD general.

La diferencia A0−Garl matched es `+180.7031360` MiD, IC95% bootstrap por
secuencia `[131.7444284,215.3146093]`; A0 gana el `35.6904%` de los 1.796 pares
finitos. La identidad firmada del run es
`553904c18874b3509e10a71e5b46b33e0f5df6ddb4fec7a7e57b6abc34322937` y la del
comparador completo `e63447135e2b09c5c6a7e2afb996bb70cce8cbba4a112afc87069e2f60c254de`.
Training+validation tardó `274.9784 s`; inferencia final `4.9803 s` (411,22/s),
peak VRAM `1.317,58 MiB`, 24.674.178 parámetros.

## A1 bbox geometry-only: resultado seed 7

A1 conserva exactamente el model config de A0 y 344.591 parámetros entrenables.
No cambia encoder, foreground head, residual, consenso temporal, conversión física,
optimizer, scheduler, seed, filas, batch, warm-up, CVaR, unknown gate o clip. El
forward continúa siendo `forward(events, delta_t_s)`.

La única diferencia es la supervisión foreground. Las bbox t1/t2 se recortan al
ROI visible 128×128 y se convierten numéricamente, sin rasterización, en:

```text
h=(y2-y1)/H   w=(x2-x1)/W
cx=(x1+x2)/(2W)   cy=(y1+y2)/(2H)
```

El mismo mapa foreground produce momentos 2-D diferenciables `h_hat,w_hat,cx_hat,
cy_hat`. A1 usa Huber sobre `log h`, `log w` y centros. BCE y Dice son cero; el
pair-ratio sigue en cero. Los pesos preregistrados son `1.25,1.25,2.5`; la loss de
centro promedia x/y, dando peso efectivo 1.25 a cada una de las cuatro magnitudes y
conservando suma nominal 5 del foreground geométrico A0. La bbox es target
training-only, nunca input. t0 proxy permanece excluida.

Cada época registra global y macro por secuencia:

- Pearson, slope, MAE y `std_pred/std_target` de `log h` y `log w`;
- MAE/correlación de `cx,cy`;
- `delta log h`, `delta log w` y `r_iso=(r_h+r_w)/2` contra bbox;
- los tres ratios diferenciales contra el ratio físico TTC;
- sign accuracy y counts sin rellenar NaN con cero.

`r_iso` es solo diagnóstico; no alimenta TTC. El checkpoint sigue seleccionándose
por MiD macro y failure rate, como A0. No hay umbrales diagnósticos inventados
post-hoc. Config SHA256:
`bc3fe3daabb8f205b1dda81f6da442c2d7452253330960d0c3ff65af7795ba28`.

El run publicado desde commit `aec7b50c0595ef7715e14107921c254c2476d066`
completó 18 épocas en `cuda:0` y seleccionó la 18. No se reconfiguró tras observar
validation.

| Métrica validation pública | A0 | A1 | Garl matched |
|---|---:|---:|---:|
| MiD global | 383.8549 | 346.1117 | 203.0982 |
| MiD macro-secuencia | 382.1905 | 346.8295 | 203.6342 |
| failure rate | 12.3047% | 9.9609% | 0% |
| known coverage | .876953 | .900391 | 1.0 |
| Pearson log-ratio | .045629 | .110821 | .372213 |

A1 por secuencia obtiene `374.2571/327.7436/338.4877` en
`DGqicHUGWb/pBqGOb2vYq/qoohcdtLDH`; mejora A0 en las tres, pero pierde frente a
Garl en las tres. Por bucket obtiene crucial `481.5264`, small `224.9528`, large
`164.0883` y negative `214.5390`, con failures `8.97/9.05/10.77/11.94%`.

| Diagnóstico A1 | Pearson | slope | MAE |
|---|---:|---:|---:|
| `log h_pred` vs bbox | .470828 | .141804 | .233985 |
| `log w_pred` vs bbox | .078759 | .035893 | .206415 |
| `cx_pred` vs bbox | .063569 | .111446 | .044197 |
| `cy_pred` vs bbox | .031956 | .123819 | .037424 |
| `delta log h` vs bbox-ratio | .059130 | .041515 | .028481 |
| `delta log h` vs ratio físico | .104778 | .083482 | .026817 |
| `delta log w` vs ratio físico | -.039961 | -.085335 | .046023 |
| `r_iso` vs ratio físico | -.000826 | -.000926 | .032220 |

A1 reduce MiD macro en `35.3611` respecto de A0, pero queda `143.1953` por detrás
de Garl matched. El bootstrap de tres secuencias completas para A1−Garl da IC95%
`[115.1042,166.6705]`, siempre desfavorable; A1 gana `721/1844` pares finitos
(`39.0998%`). Hubo 204 NaN/unknown y 1.844 predicciones finitas, sin sustituir
unknown por cero. El run tardó `631.8786 s`, usó `1558.48 MiB` peak VRAM y conserva
344.591 parámetros. Checkpoint SHA256
`29ed410b39372e67cac87e5fb0e4be2b659f1a923ea1ebfff3f49e364e139e43`.

Conclusión: eliminar BCE/Dice weak-box ayuda parcialmente, sobre todo a altura,
pero no enseña anchura/centros ni cambio temporal fiable. La hipótesis de que la
weak-box era la causa suficiente queda rechazada. Como la geometría estática
completa sigue siendo mala, A1-R no es la siguiente intervención: primero se debe
probar una representación densa event-native mejor, manteniendo la misma cabeza
geométrica para aislar el cambio.

Comparador exact-token firmado:
`artifacts/metrics/causal_scale_eap_garl_event_only_a1_geometry_comparison_v1.json`,
identidad `471fa106f4137f71ecfa4165abec696e5f83644830ded14a82abff8fb7ba485d`.

### Auditoría por endpoint y observabilidad

El diagnóstico firmado `causal_scale_eap_a1_geometry_observability_v1.json`
repite inferencia BF16 en GPU sobre los mismos 2.048 tokens. Altura predicha frente
a bbox obtiene Pearson `.4783/.4931` en t1/t2; anchura solo `.0484/.1055`. La
anchura target tiene std `.0962/.0948`, luego su fracaso no se debe a target
constante. Los centroides están cerca del centro por construcción del common ROI y
tienen poca varianza, por lo que su correlación aislada es menos informativa.

Como control label-free, los momentos de `abs(events)` dan extents medios
`.9966/.9985` en t1 y `.9965/.9984` en t2, con std `~.003`: la actividad cruda es
difusa en casi todo el crop. Sus deltas de altura/anchura correlacionan
`-.0521/-.0777` con bbox. Esto no es una máscara ni un gate; demuestra que sumar
actividad no localiza el objeto.

El decoder actual aplica un stem 2-D al input y después `amax` por anchura/altura
antes de cabezas 1-D. La hipótesis mecanística siguiente es que ese máximo axial
retiene background/hot activity y pierde coocurrencia 2-D. Se probará el decoder
`equivariant_fullres` ya existente, manteniendo toda la loss/protocolo A1. No se
mezclará con pair-ratio, weak-box, teacher RGB o JEPA. Identidad del diagnóstico:
`737a3663c13dc083b918e0101f4954bcfc22b23257255e0d183f8e09f0aa635d`.

A1-FR queda congelado antes de ejecución. Solo cambia
`foreground_decoder: equivariant_separable -> equivariant_fullres`; tiene 340.870
parámetros, menos que A1. Data, training y loss son idénticos. Experiment SHA256
`7ceb114963e8aad8f4c7edeb70344759543d3ac58abc6a47b862d3acf772c42e`; model
SHA256 `97232184d7fb00520136319f5e902c726e26766ddaae236459b6d42d9596d39a`.

El primer proceso A1-FR completó nueve épocas, pero su best epoch 4 no es evidencia:
`DGqicHUGWb` tenía MiD global NaN debido a todos los casos negative unknown y el
macro omitió la secuencia no finita. El defecto era del selector, no una autorización
para reportar un macro de dos secuencias. El run se conserva y queda invalidado por
el artifact `fd5bde50328080975781a8fc2cdae1e0a198bb5a878430fde3e1f87c9be8f19b`.
La repetición correcta exige cobertura finita 3/3 y parte desde cero.

## Estado

A0, referencia release, Garl matched y A1 están terminados. El cache matched
oficial tiene identidad `92af281030170733411ef9d65b19e88ebc8019c729dd6743e02ae9c40f564b52`,
2.048/2.048 filas y preprocessing separado de `166.7501/155.3283 s`. Resume A0
coincide exactamente con entrenamiento continuo y el runner matched liga resume a
todo el protocolo congelado. El siguiente brazo requiere un preregistro nuevo.
