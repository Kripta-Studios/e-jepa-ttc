# Continuación de los tres requisitos científicos

Documentación de conjunto solicitada tras publicar los experimentos:

- [01: trabajo realizado](01_TRABAJO_REALIZADO.md): continuación desde el handoff,
  implementación, ejecuciones, recuperación, controles y publicaciones.
- [02: resultados y pendientes](02_RESULTADOS_Y_PENDIENTES.md): cifras verificadas,
  logros, resultados negativos y requisitos que faltan para aprobar un modelo.

El [informe v1](v1/README.md) registra experimentos terminados, no el cumplimiento
del objetivo SOTA. No hay nueva versión aprobada ni evaluación oficial Test12.
La investigación requiere acceso a esa evaluación y aclaración del protocolo
Slider-750, además de un candidato que supere todos los controles.

## Runs conservados

- Recuperación y cadena nativa:
  `output/release_research_20261031/native_platform_followup_v2`.
- Router entregable congelado:
  `output/phase_timewarp/eap_router_sensitivity_v1`.
- Reajuste emparejado rechazado:
  `output/phase_timewarp/canonical_router_refit_v1`.
- Disponibilidad pública y plantilla Test12:
  `output/phase_timewarp/external_dependencies_20261101/VERIFIED.json`.

El diagnóstico del router completó su inferencia y sellado en 62.36 segundos;
ese tiempo excluye la verificación previa de todos los padres y no es un
benchmark de inferencia de extremo a extremo. La cobertura de 6762 aliases
representa 5805 consultas físicas; no son 6762 muestras independientes.

## Reproducción

Los protocolos de [reajuste](../research_log_canonical_population2608/CANONICAL_ROUTER_REFIT_PROTOCOL.md)
y [router eAP](../research_log_canonical_population2608/EAP_ROUTER_SENSITIVITY_PROTOCOL.md)
contienen las condiciones previas. No reutilizar un directorio de salida
sellado. El informe puede regenerarse en un directorio nuevo con:

```powershell
$env:PYTHONPATH='src;.'
$env:PYTHONUTF8='1'
.venv-transfer/Scripts/python.exe scripts/report_phase_three_requirements.py `
  --native output/release_research_20261031/native_platform_followup_v2 `
  --router output/phase_timewarp/eap_router_sensitivity_v1 `
  --refit output/phase_timewarp/canonical_router_refit_v1 `
  --external output/phase_timewarp/external_dependencies_20261101/VERIFIED.json `
  --output output/phase_timewarp/three_requirements_report_reproduction
```

El generador verifica los informes contra sus recibos COMPLETE, la genealogía
del router, la auditoría del evaluador publicada y la identidad y componentes
del entregable. Copia los bytes de los doce JSON fuente, sin normalizarlos.
`v1/STATUS.json` registra sus hashes y mantiene `goal_achieved: false`.

Validación del código nuevo: ocho pruebas del reajuste y nueve de la auditoría
del router pasan; Ruff y Pyright scoped sin errores. La ejecución real de la
auditoría y del generador terminó con código cero. No se modificaron los
umbrales tras observar los resultados ni se seleccionó una semilla favorable.

## Dependencias que siguen abiertas

Test12 conserva sus targets privados en la
[revisión oficial verificada](https://huggingface.co/datasets/NAIL-HNU/GarlTTC-dataset/blob/53fbaade17f3b76aa56dc7df2b527ce279062f59/README.md)
y requiere [evaluación oficial](https://www.codabench.org/competitions/17289/).
La plantilla coincide exactamente con los 6762 tokens locales. No hubo envío.
El acceso solicitado al usuario sigue pendiente.

Slider-750 mantiene 283.931058% local frente a 132% en el resultado recibido;
nueve de diez filas concuerdan al redondear. El
[borrador para los organizadores](../research_log_20261031_sign_safe/organizer_evaluator_question.txt)
solicita el alineamiento y la versión del evaluador. No se envió: el usuario
exigió pedir permiso antes de contactar y la pregunta sigue pendiente.

Las poblaciones de competición EvTTC, transferencia Garl-TTC y validación
in-domain REACT no son intercambiables. Ninguna de estas comprobaciones acredita
una comparación completa contra SOTA. El fallo del candidato nuevo también
permanece abierto como resultado científico; no se convierte en un éxito por
haber terminado el entrenamiento.
