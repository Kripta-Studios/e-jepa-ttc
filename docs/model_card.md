# Model card — E-JEPA-TTC / OGE-JEPA-TTC

Actualizado: 2026-07-30.

## Estado

El único modelo con resultado local reproducido exactamente es
`B0_HISTORICAL_BASE_EXACT`. OGE-JEPA-TTC y Garl-TTC local son candidatos en
selección; no existe aún una configuración final ni un claim SOTA.

## BASE histórico

Arquitectura:

```text
21 canales
→ EventTubeletTransformer, dim 192, depth 6, 6 heads, patch 16
→ mean pooling de tokens finales
→ LN → Linear 192/96 → GELU → Dropout 0,1 → Linear 1
→ log-TTC
```

Resultado validation histórica, seed 7:

| MAE | RMSE | Error relativo |
|---:|---:|---:|
| 0,322892 s | 0,584432 s | 8,1554 % |

Predicciones y métricas tienen paridad exacta con el checkpoint original.

## Candidatos Core

- `A0_MATCHED_GLOBAL`: control entrenado en la misma matriz.
- `A1_MATCHED_DENSE_BLOCK`: interacción espacial antes de causalidad temporal.
- `A2_MATCHED_DENSE_ATTNRES`: combinación por tarea a través de capas.
- `K1_OBJECT_KDA`: recurrencia delta temporal de tokens de objeto/región.
- `A4_GT_GEOMETRY`: oracle sin entrenamiento basado en bbox GT.

Los cuatro candidatos aprendidos comparten inicialización, cabeza común,
selección de muestras y trainer. El oracle geométrico se reporta separado.

## Réplica Garl-TTC

Implementa:

- tres timestamps a 100 ms;
- dos RGB endpoints;
- dos event volumes de 20 planos;
- ROI 128x128;
- ResNet-50 separados;
- direct regression y Learned Height Ratio;
- early/late fusion;
- decoder foreground solo durante training.

EvTTC no proporciona el target 3D de altura usado en eAP. La adaptación local
supervisa altura visible en la ROI y se declara explícitamente.

## Geometría y navegación

Expertos:

- height ratio;
- area rate;
- expansión afín;
- event contrast.

La navegación se transforma físicamente a la cámara de eventos. La de-rotación
por yaw es causal. La compensación traslacional requiere profundidad:

- distancia EvTTC → oracle/teacher;
- profundidad predicha → candidata futura de inferencia.

No se permite concatenar `velocity_x` directamente como closing speed.

## Entradas y salidas previstas

Entradas:

- eventos causales;
- RGB opcional en el track RGBE;
- bbox GT en el track assisted;
- navegación causal;
- máscara/bbox predicha en el track FULL futuro.

Salidas:

- TTC continuo;
- inverse-TTC geométrico y diagnósticos;
- riesgo e incertidumbre solo si pasan sus gates;
- máscara visualizable en la variante bbox-free.

## Uso previsto

Investigación de TTC object-centric y representación predictiva de eventos.

No usar para:

- control real de vehículos;
- decisiones de seguridad;
- afirmar SOTA antes del benchmark oficial;
- inferencia con distancia ground truth.

## Riesgos y limitaciones

- EvTTC tiene pocas secuencias para routers complejos;
- el bbox-assisted no demuestra detección bbox-free;
- la navegación puede convertirse en shortcut;
- KDA, AttnRes y Dense Patch aún necesitan screen/CV;
- Garl local no reproduce el ground truth eAP privado/no disponible;
- latencia p95 end-to-end no está cerrada;
- Benchmark-10 permanece sin evaluar.

## Reproducibilidad

Cada run guarda configuración, seed, hashes, commit, selección de muestras,
época, latencia, VRAM, `best`, `last` y `weights_only`. Las métricas smoke no
son promocionables.
