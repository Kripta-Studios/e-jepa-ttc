# Continuación después de V16: candidato real y mejora de representación

Base documental: `71ec9d900189a120b06c3956029496e6fd704f6f`.
Fecha: 12 de octubre de 2026. No incluye resultados posteriores de V18.

## Decisión

Conservar `PHASE_ONLY_EVENT_S7` como candidato oficial ya congelado y `F_TRUE`
como referencia multimodal distinta, no sustituida universalmente. Obtener una
puntuación oficial del ZIP existente cuando el usuario autorice expresamente el
envío y se verifique el contrato del benchmark. Continuar la investigación de
representación en V18 sin relanzar la búsqueda de 32 cabezas V16.

Este paquete NO crea una campaña V19 ni amplía el presupuesto V18. No contiene
un entrenador nuevo ni una solución TTC nueva. Incluye revisión, coordinación,
plan condicionado de entrenamiento y scripts CPU para reproducir diagnósticos.

## Orden

1. `docs/01_REVISION.md`
2. `docs/02_PLAN_OPERATIVO.md`
3. `docs/03_REFIT_Y_EVALUACION.md`
4. `PROMPT_CONTINUAR_V16.md` para cierre, comparación y preparación oficial.
5. `MENSAJE_V18.md` para la sesión que entrena la representación.
6. `AUTORIZACION_SUBMISSION_OPCIONAL.md` sólo si se decide autorizar ese acto.

Los archivos de evidence son recálculos propios desde predicciones o recibos de
verificación. No son nuevos entrenamientos. La revisión de seis cabezas usa
las entradas V cacheadas, no sensores. No se redistribuyen los pesos ni datasets
originales en este paquete.

## Reproducción

El ZIP del usuario tiene dos niveles. Extraer el exterior y el ZIP conservado en
`evidence/`; llamar ROOT a la carpeta interior con `operational/`, `src/` y
`artifacts/`. Con el entorno del proyecto:

```powershell
python scripts/analyze_saved_predictions.py --root $ROOT --output ./recomputed
python scripts/replay_phase_only.py --root $ROOT --output ./head_replay
python scripts/replay_published_metrics.py --root $ROOT --output ./metric_replay
python scripts/check_submission.py --zip "$ROOT/artifacts/multimodal_research_20261012/test12_phase_only/PHASE_ONLY_S7_GarlTTC_submission.zip"
python -m unittest discover -s tests -v
```

No lanzar los antiguos módulos `train` o `test12_infer` para leer estos informes.
Los scripts de replay escriben fuera de ROOT, sin alterar la evidencia histórica.
