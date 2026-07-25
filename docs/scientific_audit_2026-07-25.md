# Auditoría científica y comparación SOTA — 2026-07-25

## Dictamen

La arquitectura tiene una señal de investigación interesante, especialmente en
régimen de pocas etiquetas, pero el repositorio **no demuestra SOTA**. Ninguna
cifra local puede compararse de forma válida con Garl-TTC o con la tabla oficial
de EvTTC porque cambian las secuencias, la unidad de evaluación, la asistencia
por ROI/modalidad y el protocolo de test. Además, la evidencia histórica tiene
fallos materiales de procedencia.

La decisión recomendada es continuar, pero como proyecto de investigación con
un nuevo protocolo limpio. No se debe optimizar más contra `CPLA-high` ni
presentar `6.42%` local frente a `10.60%` de Garl-TTC como una victoria: son
métricas y poblaciones diferentes.

## Alcance auditado

- Rama: `scientific-recovery-v3-hardening`.
- Revisión inicial: `e2803d651432a7b06eed790d0ab53ebbba437260`.
- Historial reciente, scripts de ejecución y exportación, modelos, pérdidas,
  loaders, manifests, splits, caches, checkpoints, resúmenes y registro de
  artefactos.
- Datos locales: nueve secuencias EvTTC del starter y ocho secuencias eAP de
  entrenamiento, aproximadamente 117 GB en total.
- Literatura primaria de TTC con eventos, JEPA/world models y los artículos
  MVA y FlowMimic indicados por el usuario.

## Hallazgos de integridad

### Críticos

1. El protocolo congelado contenía hashes literales de prueba (`fake_*`) y el
   orquestador/selector ONNX admitían fallbacks no físicos. Se sustituyeron por
   hashes SHA-256 de recursos reales y validación fail-closed.
2. El registro JSONL pasaba su validación de forma, pero no verificaba que los
   ficheros existieran ni que sus hashes coincidieran. La comprobación física de
   45 registros encuentra 116 incidencias: 85 referencias ausentes, 26 hashes
   distintos y 5 hashes ausentes. Algunas rutas aparecen repetidas en varios
   campos, por lo que estas cifras son incidencias, no 116 artefactos únicos.
3. El cache global usado por los mejores resultados es formato v1. Su array
   físico `x.npy` tiene forma `[3494, 21, 90, 160]`, mientras el sidecar declara
   `shape=[3972, 21, 90, 160]` y, contradictoriamente, `window_count=3494`.
   Tampoco contiene `sample_id`. No puede sostener un resultado promocionable.
4. `CPLA-high` se ha consultado repetidamente durante el desarrollo. Es un test
   diagnóstico reutilizado, no un holdout final.
5. El evaluador final termina explícitamente con
   `FINAL_TEST_EVALUATOR_NOT_IMPLEMENTED`; la suite de robustez devuelve estados
   `not_implemented_cache_bypass_required`.

### Altos

1. La matriz object-JEPA usa cajas futuras y profundidad futura para targets
   geométricos. No usa TTC como etiqueta durante pretraining, pero tampoco es
   SSL puramente no etiquetado de eventos.
2. Con acciones activadas, el predictor recibe acciones futuras del ego. Esto
   solo es causal si son una trayectoria planificada disponible en inferencia;
   no lo es si proceden de la navegación observada a posteriori.
3. La rama object-JEPA es asistida por detección/ROI. No debe compararse como
   modelo event-only contra métodos full-frame.
4. El TTC de eAP local se reconstruye a partir de tracks 3D y usa un split local;
   no reproduce todavía las etiquetas Garl-TTC ni el split oficial de 46/12
   secuencias.
5. Los manifests low-label podían aceptar índices ajenos a train y registraban
   IDs de split como si fueran IDs de secuencia. El productor y consumidor
   ahora validan hash, unicidad, pertenencia a train y secuencias reales.

### Completitud del contrato

De una comprobación de 80 rutas exactas exigidas por `AGENTS.md`, faltan 64.
Parte de la funcionalidad existe con otro nombre, pero siguen ausentes, entre
otros, la configuración principal completa, DSEC, varios módulos de pérdidas y
evaluación, el runner de robustez real, el generador del informe, dos workflows
de CI y los targets `finetune-ttc`, `evaluate`, `demo` y `report` del Makefile.
No hay Pyright ni mypy instalado.

## Verificación tras el endurecimiento

- `pytest -q`: 191 tests superados.
- `ruff check src tests scripts`: sin errores.
- Auditor del cache object-centric: 576 muestras; train 320, validation 64,
  calibration 64 y test 128; cero solapamientos context→future,
  future→target o de secuencias entre splits. Los 1,152 solapamientos entre
  frames históricos ocurren dentro del contexto causal y son intencionales.
- El protocolo congelado declara ahora `claim_level=diagnostic`,
  `test_status=reused_test_diagnostic` y `cache_format_version=1`.
- El gate de smoke actual falla correctamente: detecta commits/protocolos
  antiguos, resúmenes sin tipo/firma, metadatos ONNX incompletos y los hashes
  literales `fake_hash`/`env_fake_hash_for_now` dentro de artefactos históricos.

Que los tests pasen significa que los guardarraíles actuales son coherentes; no
convierte automáticamente los artefactos históricos en evidencia válida.

## Resultados locales recuperables

| Evidencia | JEPA | Scratch/referencia | Lectura correcta |
| --- | ---: | ---: | --- |
| Global, 100% etiquetas, test diagnóstico reutilizado | `0.312 ± 0.044 s` | token+nav scratch `0.465 ± 0.021 s` | Prometedor, pero cache v1 inconsistente, un solo seed SSL y test reutilizado |
| Global, 10% etiquetas | `0.460 ± 0.029 s` | `1.327 ± 0.104 s` | Mejora local de 65.4%; es la señal científica más interesante, no una cifra oficial |
| Global, 5% etiquetas | `0.636 ± 0.109 s` | `1.382 ± 0.044 s` | Mejora local de 53.9%; misma limitación de procedencia |
| Object-JEPA, acciones activadas, 3 seeds | `0.349 ± 0.029 s` | `0.453 ± 0.168 s` | Mejora media de 22.8%, pero JEPA pierde en seed 17 y la media scratch está dominada por seed 42 |
| Object-JEPA, acciones desactivadas, seed 17 | `0.305 s` | `0.543 s` | Ablation incompleta; no admite conclusión multi-seed |
| Geometría causal bbox, 81 frames CPLA-high | `0.157 s` | — | Baseline muy fuerte, pero detection-assisted y en test reutilizado |

La cobertura conformal nominal del 90% del object-JEPA con acciones fue
`0.768 ± 0.227`: está mal calibrada y es demasiado variable. Los agregados
Garl ponderados son `NaN` porque el test local carece de los cuatro buckets
necesarios. Esto impide comparar el object-JEPA con la tabla oficial.

## Comparación correcta con SOTA

La referencia más fuerte y directamente relevante encontrada es
[Garl-TTC/eAP](https://arxiv.org/abs/2603.16303). En las tres secuencias reales
de EvTTC que reproduce su tabla, Garl E+V reporta RTE medio `10.60%` y latencia
`13 ms`; STRTTC reporta `11.96%` y `25 ms`. En eAP, Garl E+V reporta RTE por
bucket `16.6/20.0/34.1/28.2%` para crucial/small/large/negative.

El `6.42 ± 0.45%` local es error relativo medio sobre `CPLA-high`, no el RTE de
Garl en sus tres secuencias. Tampoco usa la misma población de frames ni la
misma asistencia E+V/ROI. Por ello:

- no se puede decir que `6.42 < 10.60` implique superar SOTA;
- tampoco se puede concluir que la arquitectura sea peor;
- la única respuesta científica actual es **no comparable / no demostrado**.

EvTTC compara STRTTC, CMax, ETTCM, FAITH, AEB-Tracker e Image FoE bajo su propio
protocolo ([paper de EvTTC](https://arxiv.org/abs/2412.05053)). El repositorio
solo cubre tres de diez filas/secuencias de la tabla localmente. El método
geométrico de [Event-Aided TTC](https://arxiv.org/abs/2407.07324) también parte
de un vehículo/ROI identificado, por lo que es el baseline conceptual apropiado
para la rama object-centric, no para el modelo full-frame.

## ¿Hay futuro en la arquitectura?

Sí, pero la hipótesis defendible no es aún “mejor MAE absoluto que SOTA”. Es:

> El pretraining latente multihorizonte aporta eficiencia de etiquetas y
> robustez a domain shift frente a la misma arquitectura entrenada desde cero.

La señal low-label es grande y consistente en las tres seeds downstream
históricas. La escala mayor y la deep supervision empeoraron, mientras que la
estructura causal, los tubelets y la geometría ayudaron. Esto sugiere invertir
en inductive biases de movimiento/expansión y mejor protocolo, no en aumentar
parámetros.

## Artículo 2607.19343 — Masked Visual Actions (MVA)

[MVA](https://arxiv.org/pdf/2607.19343) representa la acción como la trayectoria
espacio-temporal enmascarada de una entidad en píxeles. Usa la acción del robot
para predecir la escena pasiva (forward) y el movimiento deseado del objeto para
inferir el robot (inverse). Sus resultados son de reconstrucción/robotics sobre
DROID, RoboCasa y BEHAVIOR, no de TTC. El sistema completo ajusta un modelo de
vídeo de 14B con ocho H200; no es una receta viable para este MVP.

Lo aprovechable:

1. Reemplazar el vector global de movimiento del objeto por un **tubo visual
   causal** calculado solo con eventos/caja del contexto.
2. Proyectar la acción planificada del ego a un **campo de flujo espacial** en
   vez de concatenar ocho números globales a todos los tokens.
3. Separar dos modos: estimación observacional event-only y evaluación
   contrafactual condicionada por una trayectoria planificada.

No se deben introducir máscaras, cajas o acciones ground-truth futuras. MVA
reconoce que aprende correlación y no causalidad; su máscara debe tratarse como
un inductive bias, no como demostración causal.

## Artículo 2607.18227 — FlowMimic

[FlowMimic](https://arxiv.org/pdf/2607.18227) crea pares de edición de vídeo
aplicando el mismo warp temporal a source y target de pares de imagen. Incluye
pan, zoom, rotación, stretch local/global, deformación elástica y composiciones;
su “modality mimicry” alinea tareas de imagen y vídeo. Se evalúa en edición y
generación de vídeo, no en cámaras de eventos ni TTC.

Es el artículo con mayor utilidad inmediata para E-JEPA-TTC:

1. Generar aproximaciones frontales con escala física
   `s(t)=TTC/(TTC-t)` en el plano de imagen.
2. Aplicar el warp a una escena de log-intensidad y **después** simular eventos;
   deformar directamente un voxel grid produciría eventos no físicos.
3. Añadir una pérdida de equivariancia de flujo:
   el embedding futuro predicho tras el warp debe corresponder al embedding
   target warpado.
4. Añadir una cabeza auxiliar de expansión o `inverse TTC`, manteniendo TTC fuera
   del pretraining autosupervisado real.
5. Usar modality mimicry solo para alinear count/time-surface/voxel en componentes
   globales. Una igualdad fuerte entre representaciones puede borrar la señal
   temporal que interesa.

## Arquitectura propuesta para la siguiente iteración

```text
eventos de contexto
  -> voxel/tubelets multiescala
  -> encoder causal
  -> tokens de objeto + fondo
  -> predictor por horizonte condicionado por:
       a) tubo de movimiento observado en contexto (MVA)
       b) flujo ego planificado opcional, separado del modo event-only
  -> embedding futuro + inverse-TTC + riesgo + incertidumbre

target EMA <- eventos futuros reales, sin TTC, sin cajas/acciones futuras

regularización adicional:
  FlowMimic físico -> simulador de eventos -> consistencia/equivarianza latente
```

## Experimento mínimo decisivo

No ejecutar una búsqueda cartesiana. Congelar en validación y probar:

1. `E0`: tubelet-JEPA limpio sobre cache v2 reproducible.
2. `E1`: `E0` + aproximación física FlowMimic.
3. `E2`: `E1` + pérdida de flujo/inverse-TTC.
4. `E3`: `E2` + tubo causal de objeto MVA.
5. `E4`: `E3` + campo de flujo de ego planificado, solo donde esté disponible
   también en inferencia.

Para cada fila: scratch con arquitectura idéntica, tres seeds, 5/10/100% de
etiquetas, bootstrap por secuencia, robustez y calibración. Abrir el test una
sola vez después de congelar arquitectura e hiperparámetros.

Gates recomendados para seguir invirtiendo:

- mejora media de al menos 10% en MAE low-label frente a scratch;
- beneficio presente en las tres seeds, no solo en la media;
- intervalo bootstrap por secuencia favorable;
- no degradar más de 5% el régimen 100% etiquetas;
- incertidumbre que aumente bajo corrupción y cobertura 90% entre 85–95%;
- reproducción oficial de EvTTC/eAP antes de cualquier texto “SOTA”.

## Trabajo restante para una afirmación SOTA

1. Construir cache v2 desde cero y hacer coincidir sidecar, arrays y manifests.
2. Crear un nuevo holdout no inspeccionado o usar el test oficial sellado.
3. Completar las secuencias oficiales EvTTC y el split/labels oficiales eAP.
4. Reproducir Garl/STRTTC o evaluar con su código y el mismo protocolo.
5. Completar robustness, final evaluator, ONNX real y reporte regenerable.
6. Ejecutar tres seeds de pretraining, no solo tres fine-tunings de un encoder.
7. Publicar predicciones por muestra y artefactos firmados con hashes físicos.

## Continuación FlowMimic

La adaptación física propuesta ya se está implementando y tiene un protocolo
separado en `docs/flowmimic_experiment_2026-07-25.md`. Esa bitácora distingue
explícitamente error TTC de latencia: `0.312 s` es MAE de estimación, mientras
que los `13 ms` de Garl-TTC son tiempo de inferencia. No se puede dividir una
magnitud por la otra para ordenar modelos.

El gate inmediato es una ablación E0/E1/E2 solo en validación, sobre cache v2 y
con control emparejado. No hay todavía resultados FlowMimic que reportar.
