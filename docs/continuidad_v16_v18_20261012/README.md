# Continuidad V16–V18: ejecución y evidencia

**[Informe de ejecución](INFORME_EJECUCION.md)** · **[Guía de envío manual a CodaBench](EVALUACION_OFICIAL.md)** · **[Resultados de coste](COSTES.md)**

PHASE_ONLY_EVENT_S7 permanece congelado, con 6.762 predicciones y sin puntuación oficial. F_TRUE conserva su función como referencia de transferencia. Esta continuación no entrenó modelos ni modificó V18.

- [Registro de candidatos](REGISTRO_CANDIDATOS.csv) y [diagnóstico de signo/rango](DIAGNOSTICO_SIGNO_RANGO.csv).
- [Descomposición de la diferencia FCWD](diagnostics/GAP_DECOMPOSITION.csv): la regresión en 3–6 segundos explica más que el déficit neto frente a F_TRUE.
- [128 pares CPU con paridad exacta](cost/full_dense/RESULT.json), [intento descartado por diferencias numéricas](cost/pilot/PARITY_FAILURE.json) y [comparación de latencias](COMPARACION_LATENCIA.csv). No acreditan paridad GPU ni una aceleración universal.
- [Recálculo de 46.778 comprobaciones escalares](metric_replay/METRICS_REPLAY_VERIFICATION.json) y [reproducción de las seis cabezas](head_replay/HEAD_REPLAY.json).
- [17 pruebas pasadas](verification/QUALITY.json), repetidas desde la extracción independiente según [el recibo de entrega](DELIVERY_VERIFICATION.json). Las dos pruebas del lector V18 usan fixtures sintéticos.
- [Protocolo geométrico](geometry/PROTOCOLO_ANOTACION.md): siguen pendientes las anotaciones independientes. No se inventaron correspondencias.
- [Estado V18 observado](V18_LIVE_STATE.json) y [comandos de continuación](COMANDOS_REANUDACION.md). El estado es una instantánea del 12 de octubre, no un monitor vivo ni resultados finales de V18.
- [Submission original](submission/PHASE_ONLY_S7_GarlTTC_submission.zip), [verificación](submission/INDEPENDENT_VERIFICATION.json) y [ficha del modelo](submission/MODEL_CARD.md). Publicar este archivo en GitHub no lo envía al benchmark.

## Publicación y alcance

El usuario autorizó este push después del cierre local. Las frases «sin push» o «repositorio limpio en 71ec9d9» de los documentos y recibos originales describen aquel cierre anterior a esta publicación. Se conservan sus bytes y fechas; no son una prohibición vigente del push solicitado. La instrucción de no enviar a CodaBench sigue vigente.

Esta carpeta contiene una selección de informes, tablas, predicciones, fuentes y recibos del ZIP verificado. Se omiten los paneles de sensores y su ZIP ciego, el array binario del replay y el inventario inicial de procesos del equipo. Los archivos omitidos se enumeran en [PUBLICATION_MANIFEST.json](PUBLICATION_MANIFEST.json). Los datos sensoriales, pesos y datasets no se añaden a Git.

Las fuentes ejecutadas se conservan como `*.py.txt` en `scripts/`, `package/` y las carpetas `cost/*/source/`: son archivos históricos para inspección, no nuevos módulos instalados. Sus bytes coinciden con el ZIP. Para ejecutarlas, se usa la extracción original con extensiones `.py` y el entorno y distribución de directorios descritos en los comandos; no son un instalador portable. Los recibos pueden contener rutas locales de ejecución. El manifiesto relaciona cada nombre publicado con su ruta original.

El ZIP completo local `E_JEPA_TTC_CONTINUIDAD_EJECUTADA_20261012.zip` tiene 38.849.993 bytes y SHA-256:

```text
bc28a8054fb89d3cdaa874344238d1768031f1c976700a2a338e93305966a103
```

`DELIVERY_MANIFEST.json` inventaría el ZIP completo; `PUBLICATION_MANIFEST.json` inventaría esta selección para GitHub. No son el mismo conjunto. Desde esta carpeta, `python VERIFY_PUBLICATION.py` comprueba los tamaños y SHA-256 publicados. La verificación de publicación no repite entrenamiento, inferencia ni benchmarks.

El [paquete de instrucciones original](package/README.md) y sus permisos son históricos; la evaluación oficial queda exclusivamente en manos del usuario. No hay score oficial ni SOTA demostrado.
