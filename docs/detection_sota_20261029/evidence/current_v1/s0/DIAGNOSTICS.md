# Diagnóstico S0 del detector congelado

Este informe usa exactamente las predicciones label-free del piloto y etiquetas raw leídas después, en un proceso separado. El perfil v2 es local y no oficial.

- Checkpoint: `dc0ce304ad1c8a791ef66395be85b09f22526bc11a834472dcb18ce16384e338`.
- Predicciones: `25cc2f8069a93e9f9be86e5664af04b786fd97cdab0763ab9c65d48fc83b6ff2`.
- Protocolo: `10da30cf8f08247d48d975f3650cb02f1b43375fe0e841a5389dfb3b5d7b4a5e`.
- Población raw/elegible/ignorada: 1936 / 1035 / 901.
- Recall class-agnostic a 2 m: 0.431884 (447/1035).
- AP101 v2 sobre clases con soporte: 0.01982172.
- Error medio de profundidad de cámara en correspondencias class-agnostic: 1.0334 m.
- Error vertical ego Z-up medio en las mismas correspondencias: 0.3613 m.

## Oráculos de desarrollo (pool fijo, rematching completo)

| Sustitución | AP101 clases con soporte |
|---|---:|
| class | 0.08653597 |
| center | 0.02673897 |
| geometry | 0.01982172 |
| ranking | 0.07633992 |

## Observador externo con paridad exacta

El sidecar aporta logits completos, query ID y el segundo componente de `pred_depth`, entrenado como log-varianza, sin cambiar las predicciones. Su correlación con el error absoluto de profundidad óptica de cámara es -0.116968; no demuestra calibración de incertidumbre de profundidad.

La box2D decodificada por query y la descomposición interna del modelo entre prior y residuo siguen sin estar disponibles. El diagnóstico geométrico `fy·H_GT/h_proyectada` es externo, usa GT y no se atribuye al modelo. No se infieren ni se fabrican internos ausentes.
