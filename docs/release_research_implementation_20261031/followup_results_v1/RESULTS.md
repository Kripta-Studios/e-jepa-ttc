# Seguimiento experimental: localización y controles

Estos seis contrastes están terminados y sellados. Las tablas se copian de sus
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

## Qué permiten decidir

- Cadencia: atribución separada a eventos/RGB; candidato ampliado intacto.
- Observabilidad: estimabilidad algebraica y residuo bajo no bastan; la lectura robusta histórica tampoco recupera una mejora en esta cohorte.
- Sensor: resolver referencia/escala antes de ajustar el productor; los eventos idénticos del banco delimitan un problema de identificación.
- Reloj: medir error y retraso ante maniobras; la ley constante no predice cambios futuros arbitrarios.
- Nativo: la deriva ya existe en el original, antes del candidato; se conserva la comprobación CUDA y las tolerancias fijadas.

No se reabren D4, O1/O2, S2/S3, V2/V3 ni JEPA mediante estos fixtures.
S1 real sigue necesitando referencia anclada y escala identificada.

## Reproducción

```powershell
python scripts/report_release_research_followup.py --output DIRECTORIO_NUEVO
```

Las copias COMPLETE de cada experimento se refieren a su carpeta original:
sus artefactos completos, incluidas trazas y fuentes ejecutadas, permanecen allí.
El COMPLETE de este informe autentica solo sus propias copias y fuentes.
