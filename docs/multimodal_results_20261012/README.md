# Resultados multimodales V16, 12 de octubre de 2026

**[Leer el informe detallado](INFORME_DETALLADO.md).** Explica el trabajo ejecutado, los 32 fits y 80.000 updates, la comparación Garl FCWD, el candidato PHASE_ONLY, los resultados negativos, la atribución, los costes y los límites de las conclusiones.

- [Tablas completas de entrenamiento y transferencia](artifacts/multimodal_research_20261012/MULTIMODAL_REVIEW_FINAL.md).
- [Comparación congelada Garl visual/full, cobertura y geometría](artifacts/multimodal_validation_20261012/MULTIMODAL_REVIEW_FINAL.md).
- [Resultados por estudio](artifacts/multimodal_research_20261012/studies/).
- [Predicciones test12: 6.762 tokens, sin puntuación oficial](artifacts/multimodal_research_20261012/test12_phase_only/TEST_PREDICTIONS.csv).
- [Contabilidad de los 32 fits](artifacts/multimodal_research_20261012/ACCOUNTING.json).
- [Coste GPU con registro y paridad](artifacts/multimodal_research_20261012/FCWD_GPU_ROUTE_COST.json).
- [Recibo de extracción independiente y recálculo](receipts/E_JEPA_TTC_RESEARCH_RESULT_20261012T164206Z.independent-verification.json).
- [Inventario de esta publicación](PUBLICATION_MANIFEST.json).

PHASE_ONLY pasa los criterios internos originales del 2 % en tres semillas de cabeza. F_TRUE mantiene mejor agregado y MAE en FCWD. No hay score oficial test12 ni SOTA demostrado. El ZIP de formato oficial está preparado y no enviado.

## Qué contiene cada entrega

GitHub contiene esta explicación, los informes históricos de ambas etapas, tablas CSV, predicciones, protocolos, diarios y recibos. Las copias de artefactos mantienen sus bytes originales. Algunas rutas absolutas de los recibos describen la máquina de ejecución; no son rutas portables ni credenciales.

El ZIP de entrega `E_JEPA_TTC_INFORMES_Y_RESULTADOS_20261012.zip` contiene esta carpeta y, dentro de `evidence/`, el paquete completo anterior `E_JEPA_TTC_RESEARCH_RESULT_20261012T164206Z.zip` sin modificar. Este último incluye 47.842 payloads: fuentes, pesos nuevos, doce endpoints V13, estados, predicciones y recibos. Su SHA-256 es `a59bac60a1a97c4c5e0b279247c98b96b1b4ce1ff3dc8db7781b4cf01372bab9`.

El paquete completo no contiene los sensores HDF5/TAR ni todas las dependencias públicas Garl. Conserva las identidades y rutas necesarias para prepararlas. No representa un instalador autónomo del runtime sensorial.

## Verificación

Desde esta carpeta, `python VERIFY_PUBLICATION.py` comprueba tamaños y SHA-256 del inventario. Desde la extracción del paquete completo, `python VERIFY.py` verifica sus payloads. Con las dependencias del proyecto instaladas, el recálculo de métricas se ejecuta en la raíz de esa extracción:

```powershell
$env:PYTHONPATH = 'src;.'
python -B -m operational.multimodal_research.research_metrics_replay
```

El recálculo previo pasó 46.778 comprobaciones escalares, trece tablas internas, 34 externas y 32 endpoints. Reutiliza el entorno Python existente y no repite inferencia desde sensores. El CSV de componentes positivos también se regeneró byte a byte.

Los documentos históricos conservan decisiones de su fecha. La etapa inicial tenía cero updates y prohibición de push. La autorización posterior permitió entrenamiento y la petición de publicación permite este push. Ninguna de esas ampliaciones autoriza por sí sola una submission oficial.
