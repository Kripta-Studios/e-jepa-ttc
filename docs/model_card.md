# Model card — E-JEPA-TTC / OGE-JEPA-TTC

Actualizado: 2026-07-30.

## Estado

`B0_HISTORICAL_BASE_EXACT` conserva paridad exacta con el resultado histórico.
El grouped CV de cinco folds × tres seeds selecciona `A0_MATCHED_GLOBAL` como
arquitectura final y rechaza A1 Dense para promoción. El perfil final de
precisión es `matched`; validation selecciona seed 13 antes del diagnóstico
family-OOD. Garl-TTC local sigue siendo una réplica arquitectónica y no existe
claim SOTA.

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

Confirmación matched:

| Candidato | Error relativo macro | MAE macro | Decisión |
|---|---:|---:|---|
| A0 global | 16,129 % | 0,701 s | control |
| A1 Dense | **15,210 %** | **0,628 s** | promover |
| A2 AttnRes | 16,136 % | 0,653 s | rechazar |
| K1 Object-KDA | 16,960 % | 0,731 s | rechazar |

Decisión final tras grouped CV multisemilla:

| Candidato | Score ± sd seeds | Error relativo ± sd | MAE ± sd | Decisión |
|---|---:|---:|---:|---|
| **A0 global** | **0,58452 ± 0,00853** | **30,25 % ± 0,52** | 1,011 ± 0,039 s | seleccionar |
| A1 Dense | 0,59312 ± 0,00349 | 30,55 % ± 0,06 | **1,007 ± 0,013 s** | rechazar |

El checkpoint operativo congelado es A0 matched seed 13. Obtiene score
`0,28992`, error relativo macro `14,46 %` y MAE macro `0,541 s` en validation.
En family-OOD reutilizado obtiene `0,53784`, `30,56 %` y `0,805 s`,
respectivamente. Benchmark-10 permanece sin abrir.

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
- Dense Patch gana un split histórico, pero pierde grouped CV multisemilla;
  A0 global es el candidato final y KDA/AttnRes permanecen rechazados;
- STRTTC tiene cobertura incompleta y la geometría bbox usa más contexto que
  el neural;
- Garl local no reproduce el ground truth eAP privado/no disponible;
- latencia p95 end-to-end no está cerrada;
- family-OOD duplica aproximadamente el error relativo de validation;
- Benchmark-10 permanece sin evaluar.
- CARLA JEPA solo ha superado smoke/contrato; su mejora TTC cross-domain no se
  conoce hasta completar la transferencia grouped-CV.

## Reproducibilidad

Cada run guarda configuración, seed, hashes, commit, selección de muestras,
época, latencia, VRAM, `best`, `last` y `weights_only`. Las métricas smoke no
son promocionables.

El pretraining CARLA guarda `best` por validation loss, `last`, `resume.pt`
atómico, optimizador/scheduler/scaler/RNG, `history.jsonl` y evaluaciones de
validation/test. El cargador EvTTC rechaza el checkpoint si declara uso de TTC,
colisión, velocidad, diámetro, Benchmark-10, un split distinto o una
arquitectura incompatible de 21 canales.
