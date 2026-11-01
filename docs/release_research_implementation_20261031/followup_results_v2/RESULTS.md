# Seguimiento experimental: localización y controles

Estos 10 contrastes están terminados y sellados. Las tablas se copian de sus
resultados regenerables; no se redactan métricas a mano.

La corrección de cadencia tiene una señal favorable en el desarrollo de 149
consultas. Los controles de observabilidad y lectura robusta delimitan por qué
sus mediciones actuales no justifican un gate mayor. Los bancos de reloj y
sensor comprueban mecanismos y limitaciones, no precisión del release.

La ampliación de 2608 y la conservación nativa son etapas distintas. Este
informe no las declara cerradas ni usa estos contrastes para cambiar su candidato.

## Índice de evidencia

- [cadence_localization](cadence_localization/RESULTS.md); origen `output\release_research_20261031\cadence_localization_analysis149_v1`.
- [observability_failure](observability_failure/RESULTS.md); origen `output\release_research_20261031\observability_failure_audit_v1`.
- [robust_measurement](robust_measurement/RESULTS.md); origen `output\release_research_20261031\robust_measurement_analysis149_v1`.
- [sensor_identifiability](sensor_identifiability/RESULTS.md); origen `output\release_research_20261031\sensor_identifiability_v2`.
- [clock_regimes](clock_regimes/RESULTS.md); origen `output\release_research_20261031\clock_regimes_v1`.
- [native_original_drift](native_original_drift/RESULTS.md); origen `output\release_research_20261031\native_original_drift_profile_v1`.
- [sensor_proxy](sensor_proxy/RESULTS.md); origen `output\release_research_20261031\sensor_proxy_analysis149_v1`.
- [sensor_reference_control](sensor_reference_control/RESULTS.md); origen `output\release_research_20261031\sensor_reference_analysis149_v1`.
- [photometric_q](photometric_q/RESULTS.md); origen `output\release_research_20261031\photometric_q_analysis149_v1`.
- [clock_cadence](clock_cadence/RESULTS.md); origen `output\release_research_20261031\clock_cadence_analysis149_v1`.

## Qué permiten decidir

- Cadencia: atribución separada a eventos/RGB; candidato ampliado intacto.
- Observabilidad: estimabilidad algebraica y residuo bajo no bastan; la lectura robusta histórica tampoco recupera una mejora en esta cohorte.
- Sensor: resolver referencia/escala antes de ajustar el productor; los eventos idénticos del banco delimitan un problema de identificación.
- Reloj: medir error y retraso ante maniobras; la ley constante no predice cambios futuros arbitrarios.
- Nativo: la deriva ya existe en el original, antes del candidato; se conserva la comprobación CUDA y las tolerancias fijadas.

No se reabren D4, O1/O2, S2/S3, V2/V3 ni JEPA mediante estos fixtures.
El usuario confirma que no hay nuevas adquisiciones/calibración y pide usar
los datos disponibles. S1 se ha ejecutado como proxy exploratorio en 72 consultas
de 24 adquisiciones; se ajusta escala excluyendo cada adquisición evaluada.
El control posterior de brillo RGB observado ayuda a separar un fallo de
referencia/alineación de una limitación del movimiento radial modelado.
Estos residuos de sensor no equivalen a mejora TTC ni calibración física conocida.
La lectura fotométrica posterior permite traslación, deformación y luz para
comprobar esa diferencia sobre TTC. El contraste del reloj se repite con
PAIR canónico para no atribuir a la dinámica un posible error de cadencia.

## Referencia RGB frente a todos los controles S1

Diferencias de MSE macro (RGB observado menos control), mismo soporte y
24 adquisiciones. Intervalos descriptivos individuales, sin corrección
por multiplicidad; no son confirmación independiente ni resultados TTC.

| Control | Delta MSE | CI95 por adquisición |
|---|---:|---|
| activity | -39.801318 | [-88.7004822920064, 27.10349312267338] |
| texture | -48.087011 | [-83.98933139583518, 2.105499440121994] |
| physical | -48.991273 | [-83.93149591787848, -0.2561229282542121] |
| permuted | -49.359675 | [-84.56412070076489, -0.04389766736511059] |

## Reproducción

```powershell
python scripts/report_release_research_followup.py --output DIRECTORIO_NUEVO
```

Las copias COMPLETE de cada experimento se refieren a su carpeta original:
sus artefactos completos, incluidas trazas y fuentes ejecutadas, permanecen allí.
El COMPLETE de este informe autentica solo sus propias copias y fuentes.
