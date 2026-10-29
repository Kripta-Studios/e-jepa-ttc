# A1: profundidad de adaptación del encoder

Estado: **NO_CANDIDATE_A1**. No demuestra promoción del sistema ni SOTA.

LAST-BLOCK y FULL parten del mismo endpoint B1 de correspondencia, seed7. Cada brazo añade 12.000 updates TTC en P. LAST-BLOCK actualiza el último bloque del encoder y el predictor; FULL actualiza el encoder completo y el mismo predictor. Se conserva objetivo, sampler, batch y presupuesto. Son 24.000 updates científicos nuevos; los 14.000 históricos del padre se reutilizan y no se cuentan otra vez. Igualar updates no iguala FLOPs.

Ambos endpoints se congelaron antes de evaluar H. La comparación tiene una sola semilla, un padre compartido y H de desarrollo; no mide variabilidad entre inicializaciones independientes ni confirma una causa única. El padre ya recibió supervisión TTC en su etapa C1.

## Poblaciones completas

| Población/modelo | n | Agregado | MiDc | MiDs | MiDl | MiDn | MAE s | Signo | Cobertura |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| P/A5-P2-FROZEN | 53890 | 100.874 | 141.813 | 62.952 | 55.295 | 55.522 | 4.751 | 3.333% | True |
| P/A1-LAST-BLOCK-S7 | 53890 | 81.179 | 90.045 | 62.549 | 67.644 | 106.278 | 7.465 | 8.215% | True |
| P/A1-FULL-S7 | 53890 | 59.652 | 59.786 | 49.214 | 56.405 | 93.547 | 5.926 | 6.413% | True |
| H/A5-P2-FROZEN | 18422 | 112.285 | 157.267 | 68.257 | 66.231 | 65.520 | 5.637 | 5.694% | True |
| H/A1-LAST-BLOCK-S7 | 18422 | 90.644 | 111.034 | 61.483 | 68.861 | 97.955 | 7.114 | 7.426% | True |
| H/A1-FULL-S7 | 18422 | 78.920 | 95.846 | 53.485 | 62.426 | 87.087 | 5.446 | 5.607% | True |

## Señal del mecanismo y admisión de desarrollo

Las razones siguientes son candidato/referencia; valores menores que 1 indican menor error. La señal de LAST-BLOCK frente a FULL y la admisión de cada brazo frente a A5 son preguntas distintas. Una banda favorable no elimina las protecciones de las otras bandas, signo o cobertura.

| Contraste H | Agregado | Crítico | Pequeño | Largo | Negativo | Elegible |
|---|---:|---:|---:|---:|---:|---|
| A1-LAST-BLOCK-S7 / A1-FULL-S7 | 1.148557 | 1.158472 | 1.149532 | 1.103082 | 1.124796 | False |
| A1-LAST-BLOCK-S7 / A5-P2-FROZEN | 0.807261 | 0.706027 | 0.900756 | 1.039710 | 1.495049 | False |
| A1-FULL-S7 / A5-P2-FROZEN | 0.702849 | 0.609447 | 0.783584 | 0.942550 | 1.329174 | False |

Decisiones completas, incluidas las protecciones que fallan:

### A1-LAST-BLOCK-S7 / A1-FULL-S7

```json
{
  "checks": {
    "bands": false,
    "coverage": true,
    "sign": false
  },
  "cost_route": "requires separate paired latency evidence",
  "eligible": false,
  "overall_ratio": 1.1485566338592017,
  "ratios": {
    "crucial": 1.1584715982674456,
    "large": 1.1030822009808314,
    "negative": 1.1247955500105884,
    "small": 1.1495323231313672
  },
  "signals": {
    "aggregate": false,
    "crucial": false
  },
  "status_scope": "DEVELOPMENT_ONLY_NOT_SYSTEM_PROMOTION"
}
```

### A1-LAST-BLOCK-S7 / A5-P2-FROZEN

```json
{
  "checks": {
    "bands": false,
    "coverage": true,
    "sign": false
  },
  "cost_route": "requires separate paired latency evidence",
  "cost_route_pass": false,
  "cost_scope": "prepared event tensor resident on GPU through signed TTC output",
  "eligible": false,
  "overall_ratio": 0.8072614051593169,
  "paired_cost_ratio": 0.6166367692031665,
  "ratios": {
    "crucial": 0.7060270722215537,
    "large": 1.0397101293080677,
    "negative": 1.495049480190436,
    "small": 0.9007555341157051
  },
  "signals": {
    "aggregate": true,
    "crucial": true
  },
  "status_scope": "DEVELOPMENT_ONLY_NOT_SYSTEM_PROMOTION"
}
```

### A1-FULL-S7 / A5-P2-FROZEN

```json
{
  "checks": {
    "bands": false,
    "coverage": true,
    "sign": true
  },
  "cost_route": "requires separate paired latency evidence",
  "cost_route_pass": false,
  "cost_scope": "prepared event tensor resident on GPU through signed TTC output",
  "eligible": false,
  "overall_ratio": 0.7028485852255124,
  "paired_cost_ratio": 0.6204021420736633,
  "ratios": {
    "crucial": 0.609447027684972,
    "large": 0.942550000701294,
    "negative": 1.3291744265669987,
    "small": 0.7835843464253487
  },
  "signals": {
    "aggregate": true,
    "crucial": true
  },
  "status_scope": "DEVELOPMENT_ONLY_NOT_SYSTEM_PROMOTION"
}
```

Orden de admisión registrado: `[]`.

La admisión incorpora el coste preparado. Ninguna decisión de esta tabla es una promoción del pipeline completo: ésta requiere las comparaciones registradas frente a E_CTX, coste E2E comparable y decisión posterior. No se ha evaluado V en este informe.

## Intervalos por grupos

2.000 remuestreos emparejados de grupos de adquisición. Cada diferencia es candidato menos referencia, según los nombres del contraste; no es un bootstrap de ventanas independientes ni un intervalo entre semillas.

| Contraste | Métrica | IC95 si todos los remuestreos están definidos |
|---|---|---|
| H/A1-FULL-S7_vs_A5-P2-FROZEN | delta_crucial | [-79.41853346520453, -42.13949948662585] |
| H/A1-FULL-S7_vs_A5-P2-FROZEN | delta_large | [-5.394236336644744, -2.0736783588435923] |
| H/A1-FULL-S7_vs_A5-P2-FROZEN | delta_negative | [8.401004477102708, 36.60492581581435] |
| H/A1-FULL-S7_vs_A5-P2-FROZEN | delta_overall | [-41.2543429390309, -24.34303744061468] |
| H/A1-FULL-S7_vs_A5-P2-FROZEN | delta_small | [-18.792437254734555, -9.730881185871539] |
| H/A1-LAST-BLOCK-S7_vs_A1-FULL-S7 | delta_crucial | [7.331923792559364, 23.385443153835126] |
| H/A1-LAST-BLOCK-S7_vs_A1-FULL-S7 | delta_large | [1.4373320682656576, 10.410874917425847] |
| H/A1-LAST-BLOCK-S7_vs_A1-FULL-S7 | delta_negative | [-2.7689024283750068, 23.656730906613934] |
| H/A1-LAST-BLOCK-S7_vs_A1-FULL-S7 | delta_overall | [7.602715908142306, 15.964872946470951] |
| H/A1-LAST-BLOCK-S7_vs_A1-FULL-S7 | delta_small | [5.50415367223249, 10.679134988617772] |
| H/A1-LAST-BLOCK-S7_vs_A5-P2-FROZEN | delta_crucial | [-64.1014515604166, -30.88549627555825] |
| H/A1-LAST-BLOCK-S7_vs_A5-P2-FROZEN | delta_large | [-1.2779655971316404, 5.728646435962212] |
| H/A1-LAST-BLOCK-S7_vs_A5-P2-FROZEN | delta_negative | [27.252379776877458, 36.86123204514452] |
| H/A1-LAST-BLOCK-S7_vs_A5-P2-FROZEN | delta_overall | [-30.280629218828242, -13.889692133433162] |
| H/A1-LAST-BLOCK-S7_vs_A5-P2-FROZEN | delta_small | [-10.726620715678102, -2.802525321302709] |
| P/A1-FULL-S7_vs_A5-P2-FROZEN | delta_crucial | [-95.36479272000241, -68.56941142146924] |
| P/A1-FULL-S7_vs_A5-P2-FROZEN | delta_large | [-2.42953659004855, 4.76231769279318] |
| P/A1-FULL-S7_vs_A5-P2-FROZEN | delta_negative | [29.926222555854373, 47.85736258988664] |
| P/A1-FULL-S7_vs_A5-P2-FROZEN | delta_overall | [-48.62941595251079, -33.815658203339396] |
| P/A1-FULL-S7_vs_A5-P2-FROZEN | delta_small | [-18.20613382431687, -9.862126240071888] |
| P/A1-LAST-BLOCK-S7_vs_A1-FULL-S7 | delta_crucial | [25.269538465981196, 35.11357760827205] |
| P/A1-LAST-BLOCK-S7_vs_A1-FULL-S7 | delta_large | [8.593510383913232, 14.22039051603412] |
| P/A1-LAST-BLOCK-S7_vs_A1-FULL-S7 | delta_negative | [6.153369603080603, 18.293263549449122] |
| P/A1-LAST-BLOCK-S7_vs_A1-FULL-S7 | delta_overall | [18.949044202548297, 24.255148361942773] |
| P/A1-LAST-BLOCK-S7_vs_A1-FULL-S7 | delta_small | [11.419027428590903, 15.390294955101421] |
| P/A1-LAST-BLOCK-S7_vs_A5-P2-FROZEN | delta_crucial | [-63.37459843121666, -40.36321306106713] |
| P/A1-LAST-BLOCK-S7_vs_A5-P2-FROZEN | delta_large | [9.177781760548053, 15.714980910404783] |
| P/A1-LAST-BLOCK-S7_vs_A5-P2-FROZEN | delta_negative | [45.05936063365296, 56.69990047043311] |
| P/A1-LAST-BLOCK-S7_vs_A5-P2-FROZEN | delta_overall | [-25.858296616926875, -13.504602175802628] |
| P/A1-LAST-BLOCK-S7_vs_A5-P2-FROZEN | delta_small | [-3.582485137639878, 2.5143193718021504] |

## Correspondencia conocida

Las mismas 24 consultas P y transformaciones de textura conocidas se usan para padre y endpoints. No son etiquetas de movimiento real ni de TTC. La media excluye únicamente valores matemáticamente indefinidos y muestra su denominador; el soporte de caja es sólo diagnóstico.

| Modelo | Soporte | Tipo | Casos definidos | Exactitud no nula media |
|---|---|---|---:|---:|
| A1-FULL-S7 | interior | warped | 48 | 0.762153 |
| A1-FULL-S7 | interior | permuted | 48 | 0.048878 |
| A1-FULL-S7 | weak_bbox | warped | 48 | 0.769908 |
| A1-FULL-S7 | weak_bbox | permuted | 48 | 0.048112 |
| A1-LAST-BLOCK-S7 | interior | warped | 48 | 0.801816 |
| A1-LAST-BLOCK-S7 | interior | permuted | 48 | 0.051149 |
| A1-LAST-BLOCK-S7 | weak_bbox | warped | 48 | 0.810131 |
| A1-LAST-BLOCK-S7 | weak_bbox | permuted | 48 | 0.049198 |
| B1-CORRESPONDENCE-STAGE1-S7 | interior | warped | 48 | 0.829594 |
| B1-CORRESPONDENCE-STAGE1-S7 | interior | permuted | 48 | 0.051683 |
| B1-CORRESPONDENCE-STAGE1-S7 | weak_bbox | warped | 48 | 0.833874 |
| B1-CORRESPONDENCE-STAGE1-S7 | weak_bbox | permuted | 48 | 0.048757 |

## Coste y trazabilidad

| Ruta | Mediana ms | p95 ms |
|---|---:|---:|
| A1-FULL-S7 | 16.065 | 31.791 |
| A1-LAST-BLOCK-S7 | 15.967 | 29.335 |
| A5-P2-FROZEN | 25.894 | 53.176 |

Coste preparado: 128 consultas P emparejadas, 20 warmups, FP32 y batch 1. Empieza con tensores residentes en GPU; excluye preparación sensorial y transferencias. No es E2E. La aceleración del entrenamiento mediante grafos CUDA es otra medición, descrita en A1_LOSS_GRAPH_20261016.md.

Los recibos de endpoints verifican los estados de pesos, Adam, scheduler y muestreo; la política congelada se contrasta con el padre. Las tablas P/H tienen recálculo independiente y métricas por grupo. Se exportan ambos modelos, incluidos los rechazados, sobre cinco ejemplos P. Los exports empiezan en tensores P2 y no incluyen el sensor crudo.

Fuentes: `adaptive/A1_adaptation_depth/evidence/COMPLETE.json`, `predictions/{P,H}/{fit}/GROUP_METRICS.json`, `bootstrap/`, `exports/` y los recibos durables de `fits/`. La contabilidad acumulada y el coste de entrenamiento, con su alcance de scopes, están en CURRENT_CONCLUSION.md.

Registro científico SHA-256: `d3a459fd8d1422c4317fd1b9d31d5b868c40f865399ff24ed5f0da8dbf207fa9`.

Evidencia completa SHA-256: `7492d350ba9a06b69481c3ab9bca74d88501f5bad1c5056d59ee4e66e33e2f31`.
