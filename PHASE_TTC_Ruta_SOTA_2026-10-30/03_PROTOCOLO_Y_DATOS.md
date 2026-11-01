# 03 — Protocolo de datos, genealogía e independencia

## Objetivo y salida de esta fase

Antes de entrenar, producir un registro verificable de qué datos y decisiones puede haber visto cada componente, qué grabaciones pueden usarse para desarrollo y qué evaluación permanece realmente independiente. La salida es un manifest sellado y un informe de exposición; no un reparto automático de carpetas.

El [anexo revisado](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/docs/superpowers/plans/evidence/2026-10-30-evttc-audit.json) registra 32 grabaciones locales, ocho relacionadas con competición y 24 candidatas. No se reabrieron sus HDF5 en esta revisión. El [README histórico](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/README.md) también declara 468 consultas etiquetadas de transferencia sobre las 32 secuencias y campañas anteriores. Por tanto, ausencia de solapamiento entre los futuros roles e independencia respecto a toda la investigación previa son propiedades diferentes.

## Paso 1. Reconstruir los nueve componentes y sus ancestros

El [loader existente](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/scripts/final_train40_transfer.py) identifica:

| ID de inventario | Función que debe auditarse | Alias de ejecución cuando existe |
|---|---|---|
| `E_A5` | Productor de eventos | `E_A5_MATCHED` |
| `E_C2F` | Productor de eventos | `E_C2F_MATCHED` |
| `R_A5` | Productor RGB | Igual |
| `R_C2F` | Productor RGB | Igual |
| `PAIR_E` | Combinación por par de eventos | `PAIR_E_MATCHED` |
| `PAIR_R` | Combinación por par RGB | Igual |
| `E_CTX_MATCHED` | Contexto temporal de eventos | Igual |
| `R_CTX` | Contexto temporal RGB | Igual |
| `PHASE_ONLY_EVENT_S7_TRAIN40` | Cabeza final | Igual |

Para cada uno, registrar hash, checkpoint padre, inicialización, datos de entrenamiento, objetivos, normalizadores, teacher si lo hubo y conjuntos que intervinieron en selección. Seguir todos los ancestros: un ajuste final con datos limpios no elimina exposición de una inicialización anterior. Incluir cachés aprendidas y reglas de calibración, aunque no se guarden como pesos de red.

**Esquema propuesto; todavía no es un archivo del repositorio:**

```yaml
component_id: E_A5
artifact_sha256: PENDIENTE
parent_artifact_hashes: []
training_manifest_hashes: []
training_groups: []
supervision: PENDIENTE
normalization_fit_groups: []
checkpoint_selection_groups: []
architecture_selection_groups: []
teacher_artifact_hash: null
exposure_status: unresolved  # clear | exposed | unresolved
evidence_refs: []
```

Una lista vacía significa «verificado que no aplica» solo cuando existe evidencia; mientras se investiga, debe figurar pendiente. Un hash demuestra identidad del artefacto, no que nunca haya visto determinado dato.

**Cierre:** todos los componentes y sus ancestros tienen estado documentado. Un componente `unresolved` puede seguir utilizándose para diagnóstico, pero limita las afirmaciones de transferencia. Si un componente estuvo expuesto, elegir una inicialización limpia o declarar el protocolo correspondiente. El SSL sobre entradas del test cuenta como exposición para una afirmación inductiva, aunque no use TTC.

## Paso 2. Reproducir la identidad y excluir grabaciones completas

El [plan existente](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/docs/superpowers/plans/2026-10-30-evttc-transfer-diagnosis.md) exige reproducir las ocho correspondencias, offsets y 365 frames. El CLI debe leer por frame, comparar arrays y hashes, y distinguir coincidencia única, ambigüedad y ausencia de datos. Incluir casos de timestamps periódicos con imágenes distintas, un píxel diferente y dos candidatos duplicados.

Excluir de ajuste, selección y calibración las grabaciones completas:

- `CCRs-1-low-100-overlap-100`, `CCRs-1-medium-100-overlap-100`, `CCRs-1-high-100-overlap-100`.
- `CCRs-2-low-100-overlap-100`, `CCRs-2-medium-100-overlap-100`, `CCRs-2-high-100-overlap-100`.
- `CCRm-low-100-overlap-100`, `CCRm-medium-100-overlap-100`.

La exclusión abarca otras cámaras, recortes, alias y derivados. Si una adquisición enlazada comparte condiciones relevantes, decidir si pertenece al mismo grupo antes de repartir. No inferir independencia porque un archivo tenga otro hash; tampoco inferir duplicación por la palabra `overlap` del nombre, que debe interpretarse con documentación de adquisición.

## Paso 3. Revisar las 24 candidatas y su historial

Este es el inventario documental, **no una asignación de grupos independientes**. [Fuente](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/docs/superpowers/plans/evidence/2026-10-30-evttc-audit.json).

| Familia descriptiva | IDs candidatos | Cantidad |
|---|---|---:|
| CCRm | `CCRm-low-0-overlap-0`, `CCRm-low-50-overlap-50`, `CCRm-medium-0-overlap-0`, `CCRm-medium-50-overlap-50` | 4 |
| CCRs-1 | `CCRs-1-low-0-overlap-0`, `CCRs-1-low-50-overlap-50`, `CCRs-1-medium-0-overlap-0`, `CCRs-1-medium-50-overlap-50`, `CCRs-1-high-0-overlap-0`, `CCRs-1-high-50-overlap-50` | 6 |
| CCRs-3 | `CCRs-3-low-100-overlap-100`, `CCRs-3-medium-100-overlap-100` | 2 |
| CCRs-side | `CCRs-side-low`, `CCRs-side-medium`, `CCRs-side-high` | 3 |
| CPLA | `CPLA-low`, `CPLA-medium`, `CPLA-high` | 3 |
| CPNA | `CPNA-low`, `CPNA-medium`, `CPNA-high` | 3 |
| CPNAO | `CPNAO-low`, `CPNAO-medium`, `CPNAO-high` | 3 |
| Total | Pendientes de revisión por adquisición | 24 |

Por grabación, registrar sesión, sensores, escenario, fuente, hashes y relaciones conocidas. Contrastar metadatos, timestamps y contenido; un cambio de recorte o compresión puede ocultar igualdad exacta. Las relaciones inciertas se investigan, se agrupan conservadoramente o se excluyen de la afirmación que exige independencia.

Añadir un historial distinto del linaje de pesos: entrenamiento, selección de checkpoint, selección de arquitectura, etiquetas inspeccionadas, scores inspeccionados, acceso solo a sensores y usos desconocidos. Buscar las campañas anteriores y sus recibos, no confiar en que el nuevo agente no las recuerde.

**Cierre:** cada grupo tiene identidad, evidencia de relación y exposición histórica. Reservar un grupo ya observado crea una reserva relativa a la campaña nueva, no un test históricamente virgen. Si todos los grupos útiles estuvieron expuestos, usarlos para desarrollo y reservar datos externos nuevos para la confirmación fuerte.

## Paso 4. Elegir el reparto según la evidencia disponible

El 60/20/20 del plan inicial es una opción, no una garantía. Tras agrupar puede dejar una reserva demasiado pequeña. Fijar el esquema sin mirar scores de candidatos:

| Situación | Decisión |
|---|---|
| Grupos suficientes y soporte adecuado de dominios | Reserva completa por grupos; selección agrupada dentro del resto. |
| Pocos grupos y datos externos limpios | Validación cruzada agrupada para desarrollo; externo reservado para confirmar. |
| Pocos grupos y sin colección nueva | Campaña preliminar agrupada; publicar cada grupo y limitar la conclusión. |
| Todos los datos influyeron en diseños anteriores | Desarrollo expuesto; confirmación posterior con adquisiciones nuevas. |
| Una familia queda sin grupos independientes | No atribuir generalización a esa familia. |

No dividir ventanas de una grabación ni sensores de una misma sesión entre roles. Documentar soporte por escenario, velocidad y TTC; no forzar porcentajes que rompan grupos. Los normalizadores y decisiones aprendidas se ajustan únicamente en el entrenamiento de cada fold. El detalle de validación anidada está en [10](10_CONFIRMACION_ESTADISTICA_Y_SOTA.md).

## Paso 5. Conseguir una evaluación Slider independiente

El inventario auditado no contiene Slider independiente. No se asume acceso actual al testbed ni a cámaras. Las opciones son solicitar una colección adicional a los organizadores, colaborar con un laboratorio o realizar nuevas capturas si se dispone del equipo. Preparar la solicitud y el protocolo; el envío a terceros requiere autorización expresa del autor.

**Plan de captura propuesto si se obtiene acceso:**

1. Definir el TTC físico de referencia y sincronización antes de grabar; medir movimiento y distancia mediante un sistema independiente de las imágenes utilizadas por el predictor.
2. Calibrar sensores y guardar intrínsecos, extrínsecos, resolución, exposición y sesgos de eventos.
3. Variar distancia, velocidad, textura, iluminación, cadencia y arranque en sesiones distintas. Incluir alejamiento o no aproximación si pertenecen al contrato elegido.
4. Separar sesiones completas para desarrollo y confirmación; no crear la reserva seleccionando ventanas fáciles.
5. Verificar precisión temporal y geométrica de la referencia antes de evaluar modelos.
6. Mantener una sesión reservada bajo custodia independiente cuando sea posible; registrar cuándo se abre.

La simulación permite comprobar expansión, paralaje y arranque. Debe separar también semillas y parámetros por roles, pero no sustituye evidencia real de transferencia al sensor. El tamaño de la colección se decide por diversidad de sesiones y precisión estadística buscada; no se inventa un número de frames suficiente.

## Entrega y condición para continuar

Completar [la plantilla de auditoría](plantillas/AUDITORIA_DATOS.md) y producir manifests de componentes, relaciones y roles, con hashes y excepciones. Estos son entregables propuestos, no archivos ya implementados. Puede iniciarse diagnóstico con medios existentes mientras se resuelve acceso adicional. El entrenamiento empieza cuando su conjunto de ajuste es elegible y la afirmación que podrá evaluarse está escrita; la falta de test nuevo limita la confirmación, no impide investigar.

Siguiente documento: [04 — Evaluador y baselines](04_EVALUADOR_Y_BASELINES.md).
