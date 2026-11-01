# Guía para ejecutar el plan sin mezclar evidencia

Este documento especifica entregables futuros. Los nombres C-* y los esquemas siguientes son propuestos; no se afirma que exista una CLI nueva. Esta tarea crea documentación y un inventario de fuentes, sin lanzar los experimentos.

## Primera sesión de ejecución

1. Leer [identidad R0](00_RELEASE_Y_PUNTOS_DE_INTEGRACION.md), el estado de la extracción concurrente y sus fuentes congeladas. Verificar procesos antes de reutilizar una salida en curso.
2. Localizar pesos y normalizadores mediante configuración; verificar sus hashes frente al sello. No copiar rutas absolutas de una máquina a la implementación.
3. Construir un manifest de consultas con identidad física, aliases, secuencia/adquisición, track, cámara y límites temporales. Registrar roles y exposición anterior.
4. Reproducir R0 con la misma preparación y política de fallback; guardar recibo de paridad y diferencias de precisión numérica.
5. Consumir resultados sellados C-TIME cuando existan. Antes de otro brazo, fijar hipótesis, control, presupuesto, métricas y regla de parada.

## Mapa de código existente inspeccionado

| Función | Entrada del repositorio |
|---|---|
| Cabeza final y contratos de entrenamiento | [operational/final_refit/training.py](../../operational/final_refit/training.py) |
| Conversión fase/TTC y saturación | [simplex_t/phase.py](../../src/e_jepa_ttc/simplex_t/phase.py) |
| Construcción de features finales | [final_refit/features.py](../../operational/final_refit/features.py) |
| Unión física OOF | [scientific_oof_features.py](../../operational/final_refit/scientific_oof_features.py) |
| Cadencia canónica | [phase_timewarp/canonical_pair.py](../../operational/phase_timewarp/canonical_pair.py) |
| Evaluación del screen/población | [evaluate_phase_canonical_screen.py](../../scripts/evaluate_phase_canonical_screen.py) |
| Normal flow y fallback histórico | [normal_flow_routing_v10.py](../../src/e_jepa_ttc/models/normal_flow_routing_v10.py) |
| Caché de geometría histórica | [event_geometry_cache_v10.py](../../src/e_jepa_ttc/data/event_geometry_cache_v10.py) |
| Solver local de expansión | [phase_event_expansion/model.py](../../operational/phase_event_expansion/model.py) |
| Controles temporales existentes | [simplex_t/controls.py](../../src/e_jepa_ttc/simplex_t/controls.py) |

Son puntos de inspección y reutilización, no una orden de editar todos esos archivos. Un nuevo módulo de investigación debe quedar aislado hasta superar los controles.

## Registro mínimo por experimento

```yaml
experiment_id: C-OBS-screen-v1
status: PROPOSED
hypothesis: "La evidencia dirigida a q anticipa correcciones perjudiciales mejor que el soporte actual"
baseline_id: R0
baseline_seal_sha256: cf5f4eac22e572679c77cd0f6a16425f978600597d4be77f0b1e576c226165a4
git_commit: null                 # completar antes de ejecutar
source_sha256: {}
parents_sha256: {}
normalizers_sha256: {}
dataset_manifest_sha256: null
split_manifest_sha256: null
exposure_role: development       # no cambiar a independiente por congelar pesos
query_identity_sha256: null
seed: 7
changed_components: [proposed_correction_gate]
frozen_components: [R0]
updates_max: 1000
primary_metric: macro_acquisition_mae_seconds
decision_rule: "08_EVALUACION_Y_PROMOCION.md, versión sellada antes del run"
future_inputs_at_inference: false
```

Los null son campos requeridos pendientes, no valores válidos para lanzar un run. Validar datos, grupos, fuentes y presupuesto antes de reservar GPU. Registrar entorno, GPU, versiones, inicio/fin, estado, checkpoint y métricas según AGENTS.md.

## Artefactos esperados

```text
<run>/
  PREREGISTRATION.md
  INPUT_FREEZE.json
  LINEAGE.json
  CONFIG.yaml
  PREDICTIONS.parquet o .csv
  DIAGNOSTICS.jsonl
  METRICS.json
  PAIRED_COMPARISON.json
  RUNTIME.json
  DECISION.md
  COMPLETE.json o FAILED.json
```

Usar los formatos y sellos del runner existente si ya cubren el contrato. No introducir otra convención solo por seguir esta lista. Una carpeta y un checkpoint parcial no equivalen a COMPLETE. Conservar todos los intentos fallidos y cualquier cambio de protocolo.

Predicciones deben incluir IDs, tiempos, target cuando su lectura esté autorizada, salida original/candidata, estado previo a emisión, rechazos y fallback. Separar extracción ciega de features de evaluación con targets cuando el protocolo lo requiera. Reutilizar cachés solo si coinciden fuentes, inputs y todos los padres relevantes.

## Pruebas de implementación que sí aportan información

Antes de ejecutar un candidato, verificar: identidad con intervención desactivada; signo y dominio en aproximación/recesión; causalidad por truncamiento del futuro; consistencia de unidades; reconstrucción de descendientes; comportamiento al faltar historia; independencia de grupos de entrenamiento/calibración; y paridad de reanudación.

Para observabilidad, fixtures con q identificable y otros con ambigüedad que la pseudoinversa oculta. Para reloj, aceleración y cruce de singularidad. Para sensor, referencias antes de ventana y dropout distinguible de silencio. Para visibilidad, máscaras vacías no deben obtener una pérdida artificialmente perfecta.

Estas son pruebas futuras del código experimental. No deben contarse como ejecutadas porque estén descritas aquí. La verificación de esta entrega se limita a integridad documental, enlaces, estructura y fuentes registradas.

## Plantilla de decisión

```text
Hipótesis y población:
Qué cambió frente a R0:
Qué control recibió el mismo reajuste/coste:
Resultado primario, efecto y CI:
Bandas, cobertura, signo y colas:
Mecanismo esperado: observado / no observado / inconcluso:
Latencia y memoria:
Exposición y límites:
Decisión: cerrar / diagnosticar / confirmar / candidato especializado:
Condición concreta para el siguiente paso:
```

No concluir «mejora el release» a partir de loss auxiliar, un subset aceptado o una sola semilla favorable. Un informe negativo con mecanismo identificado puede ahorrar una campaña completa y debe conservar el mismo nivel de trazabilidad.
