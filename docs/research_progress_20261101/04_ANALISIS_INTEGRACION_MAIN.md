# EvTTC: análisis de integración en main

Fecha de inspección: 1 de noviembre de 2026. Referencias obtenidas tras
`git fetch origin`:

```text
origin/main: 30b28b5c14eb9ed60067ed2f1dc262d7b60f2654
rama auditada: research/evttc-transfer-v2
HEAD auditado: fe49a96a5b733251bd1bd2ea89f2b17d8a87d789
```

El dictamen se refiere a ese HEAD, antes de añadir estos documentos. Los commits
de documentación posteriores no cambian los resultados de los experimentos.

## Dictamen

El trabajo EvTTC puede formar parte de `main` como investigación reproducible,
incluidos los resultados negativos y los límites de comparación. No hace falta
demostrar SOTA ni recibir una evaluación oficial del adaptado para publicar ese
código y su evidencia con el alcance correcto.

No recomendamos integrar la rama completa en su estado auditado. Hay un fallo
confirmado de CI, la documentación de entrada contiene estados superados y aún
falta validar todos los pasos del workflow sobre el cambio completo. La rama es
integrable por historial; eso no significa que esté lista para aprobar el merge.
No se ha hecho merge, cambio de rama ni sustitución del modelo entregable.

## Historial, tamaño y superficie del cambio

`origin/main` es antecesora de la rama: `git rev-list --left-right --count`
devuelve 0 commits exclusivos de main y 59 de investigación. El merge-base
coincide con main. Con esas referencias cabe un avance directo sin conflicto de
historial; cualquier cambio futuro en main exige repetir la comprobación.

El diff contiene 826 archivos, 524.326 líneas añadidas y 15 eliminadas. Los blobs
modificados suman 21.777.180 bytes, aproximadamente 20,77 MiB sin compresión.
La distribución incluye 505 archivos en `docs`, 77 en `operational`, 132 en
`scripts` y 77 en `tests`, además de planes, configuración y documentación raíz.
Es una revisión de varias campañas, no una modificación pequeña del predictor.

No hay cambios en `src/`, `uv.lock` ni `.github/` entre las dos referencias.
`pyproject.toml` añade exclusiones de Ruff para documentos recibidos y snapshots
de fuentes ejecutadas; las implementaciones mantenidas siguen bajo comprobación.
El README añade enlaces y descripciones de la investigación.

La inspección de las rutas cambiadas no encontró archivos `.pt`, `.pth`, `.ckpt`,
`.h5`, `.hdf5`, `.bag` o `.zip`. Esto comprueba esas extensiones, no constituye una
auditoría exhaustiva del contenido de cada archivo. Las mayores incorporaciones
son inventarios y JSON de evidencia; la mayor ronda 3,37 MB. El ZIP renombrado
para el correo permanece bajo `output/`, excluido de Git.

## CI remota: fallo confirmado

La ejecución [41295557687](https://github.com/Kripta-Studios/phase-ttc/actions/runs/41295557687)
corresponde a un push del HEAD auditado. Terminó en `failure`:

| Paso | Resultado |
|---|---|
| Instalación con `uv sync --locked --all-groups --no-editable` | Correcto. |
| `ruff check .` | Correcto. |
| `ruff format --check .` | Falló en dos archivos. |
| Pyright, sintaxis PowerShell, suite completa, build y CLI | Omitidos tras el fallo de formato. |

Archivos señalados por el log remoto y reproducidos en la comprobación local:

```text
scripts/report_phase_producer_timewarp.py
tests/unit/test_phase_canonical_pair.py
```

Ambos son incorporaciones de esta rama respecto a main. Que estuvieran ya
publicados en commits anteriores de investigación no elimina el fallo del
workflow. Antes de formatear fuentes relacionadas con experimentos, deben
conservarse sus bytes ejecutados y los hashes históricos; no hay que modificar
recibos para que parezcan describir código distinto.

El conector GitHub devolvió listas vacías para estados de commit y workflows
asociados a pull requests. Esas respuestas no demostraban éxito: comprobamos
también los workflows de push mediante `gh`, que mostraron el fallo anterior.
No se encontró una PR abierta de esta rama en la consulta realizada.

## Comprobaciones locales de esta revisión

- `ruff check .` pasa.
- `ruff format --check .` reproduce los mismos dos archivos pendientes;
  3.170 archivos ya tienen el formato esperado.
- Pyright con la configuración por defecto devuelve doce errores de resolución
  de `torch.Tensor` en `garl_input_contract.py`. El proyecto apunta a `.venv`.
  Repetimos el alcance de la configuración con un archivo local que selecciona
  `.venv-transfer`: cero errores. No cambiamos fuentes ni configuración compartida.
- El type-check específico de `scripts/ci_paper_validation.py` pasa.
- El análisis sintáctico de los 80 scripts PowerShell termina sin errores.
- `uv build --wheel --no-sources` construye el wheel en un directorio local
  de salida. Los comandos instalados `phase-ttc --help` y `e-jepa-ttc --help`
  terminan con código cero. Un intento previo con `python -m phase_ttc` falló:
  ese nombre no es el módulo Python; el namespace conservado es `e_jepa_ttc`.
- `git diff --check origin/main...HEAD` señala veinte líneas con espacios finales
  en tres documentos de planificación. No es el fallo que detuvo CI.

La configuración global de Pyright solo incluye cinco archivos mantenidos;
pasarla no prueba por sí sola el tipado de todo el código nuevo de investigación.
Los informes de las campañas conservan sus validaciones de alcance específico.

La suite completa local se lanzó con `pytest -q --maxfail=3` y se detuvo al
encontrar tres fallos de integración:

```text
test_final_test_guards.py::test_cache_requires_lock
test_onnx_provenance.py::test_onnx_export_rejects_altered_record
test_onnx_provenance.py::test_onnx_export_rejects_final_test_checkpoint
```

Los tres invocan `uv run --no-sync python` en un subproceso. Este seleccionó la
`.venv` del proyecto y falló al arrancar con `UnicodeDecodeError`, antes de llegar
a las guardas que cada test esperaba verificar. Ambos archivos son idénticos a
main. Repetimos solo esos tres casos con `UV_PROJECT_ENVIRONMENT` apuntando a
`.venv-transfer`: **3 passed in 24.35s**, sin editar tests ni implementación.

Esta repetición identifica el problema de entorno de esos tres casos; no acredita
el resto de la suite completa, que quedó sin terminar. No se ha medido aquí una
regresión de EvTTC por esos fallos. El éxito del build y las comprobaciones
locales tampoco sustituye una ejecución completa del workflow Linux.

## Documentación que necesita consolidación

El README raíz todavía enlaza como estado actual un informe con 423 targets
locales, once ausencias y discrepancias en CCRs1-low/medium. La auditoría posterior
corrigió la asignación de sus fuentes y dispone de 434 targets; solo Slider-750
mantiene la discrepancia de evaluación. Debe actualizarse el punto de entrada,
conservando el informe antiguo como registro histórico.

También conviven el texto del router robusto como default, el paquete adaptado
de `FINAL_MODEL` y los candidatos canónicos posteriores. El README debe indicar
qué ruta es la entrega original, cuál es el adaptado experimental y cuáles fueron
rechazadas, sin presentar estas últimas como cambios del modelo publicado.

Los borradores antiguos del correo contienen una afiliación distinta de la
condición académica que el usuario indicó ahora. Debe confirmarse el texto de
la ficha pública al consolidarlo. El cambio local de nombre a
`PHASE-TTC-Adapt.zip` también necesita una nota de ubicación para no confundir
los comandos históricos que buscan `results.zip`.

## Qué puede integrarse y qué no se puede afirmar

| Contenido | Tratamiento propuesto en main |
|---|---|
| Auditoría del evaluador y corrección de fuentes | Integrar con pruebas y procedencia. Mantener Slider-750 como pendiente. |
| Runners, controles causales, recuperación y evidencias | Integrar como herramientas de investigación con sus dependencias y alcance documentados. |
| Adaptación de la fusión | Mantenerla como versión experimental con exposición histórica declarada. |
| Resultados negativos y fallos de paridad | Integrarlos junto a las mejoras; conservar los recibos originales. |
| Métricas locales del adaptado | Publicarlas como locales, con el mismo evaluador para ambas entregas. |
| Sustitución automática del modelo o afirmación SOTA | No respaldadas por la evidencia actual. |

La integración de código en main y la promoción científica del modelo son
decisiones distintas. Las etiquetas privadas de Test12 y la respuesta sobre
Slider-750 condicionan las afirmaciones de precisión/comparabilidad, no la
posibilidad de publicar una investigación que reconoce esas limitaciones.

## Camino recomendado para integrar

1. Corregir los dos archivos de formato mediante un commit separado, respetando
   snapshots y sellos de los experimentos. Resolver también los espacios finales
   de los documentos de planificación o documentar su tratamiento.
2. Actualizar el README y el índice de resultados para reflejar 434 targets,
   la discrepancia única pendiente y la identidad de cada modelo.
3. Conseguir una CI completa en verde sobre el HEAD propuesto: tipos, PowerShell,
   tests, build y ambos nombres de CLI. Los pasos omitidos no cuentan como aprobados.
4. Abrir una PR hacia main con el alcance de investigación y revisar por bloques:
   evaluador/protocolo, adaptación, diagnósticos y evidencia. Conservar los hashes
   y las referencias de las ejecuciones al decidir la estrategia de merge.

Esta revisión no modifica las fuentes para corregir CI, no abre una PR y no
autoriza un cambio de modelo. El trabajo solicitado en este turno es documentar,
publicar la documentación en investigación y analizar la preparación para main.
