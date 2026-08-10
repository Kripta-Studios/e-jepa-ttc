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

## Estado

A0, referencia release y comparación firmada están terminados. El cache matched
oficial tiene identidad `92af281030170733411ef9d65b19e88ebc8019c729dd6743e02ae9c40f564b52`,
2.048/2.048 filas y preprocessing separado de `166.7501/155.3283 s`. Falta ejecutar
Garl matched desde cero y después A1. Resume A0 coincide exactamente con entrenamiento
continuo y el runner matched liga resume a todo el protocolo congelado.
