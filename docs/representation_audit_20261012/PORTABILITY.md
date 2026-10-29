# Qué puede reproducirse desde el bundle

El bundle separa `source/` —código y documentación— de `campaign/` —pesos, resultados y evidencia—. `MANIFEST.json` enumera cada miembro con tamaño y SHA-256. El ZIP tiene además hash exterior y una verificación independiente de CRC y de todos los hashes.

Desde `source/`, usando el entorno científico registrado y `PYTHONPATH=src;.`:

```powershell
python scripts/verify_v17_exports.py ../campaign/exports
python scripts/replay_v17_bundle_metrics.py ../campaign
python scripts/verify_v17_bundle.py /ruta/al/bundle.zip
```

El primer comando carga pesos y ejemplos de entrada incluidos. Reproduce exactamente las salidas CPU guardadas. Un audit hook bloquea accesos a las raíces originales de artefactos V13/V15; la comprobación no depende de sus checkpoints ni de las cachés completas. Los ejemplos no contienen claves de targets o del teacher DINO.

El segundo comando recalcula las métricas desde predicciones y targets guardados. No realiza nueva inferencia. La cuenta de filas incluye tablas de controles reutilizados; no representa nuevas observaciones independientes.

Esto no es una reconstrucción completa desde eventos crudos. Para repetir fits hacen falta las dependencias externas enumeradas y verificadas en `BUNDLE_DEPENDENCIES.json`, incluidas las cachés congeladas. Para volver a extraerlas hacen falta los datos e índices originales. La ruta nativa es la de consultas de objeto admitidas; no se añade ni se mide un detector o tracker externo.

Los exports incluyen `WEIGHTS.pt`, `MODEL_METADATA.json`, `EXAMPLE_INPUT.npz`, `EXAMPLE_OUTPUT.npz` y receipts. La metadata distingue cabezas, residual equal-slope, calibración de altura y lectura espacial. Los pesos del optimizador y estados RNG para continuidad permanecen en los checkpoints de entrenamiento.

Se reutilizó el entorno compartido de este equipo. `ENVIRONMENT.json` y `ENVIRONMENT_PACKAGES.json` registran el runtime observado. No se ha validado una instalación limpia del bundle en otra máquina; el lockfile heredado no sustituye ese registro de ejecución. La igualdad bit a bit no se garantiza al cambiar hardware o versiones.

Los tiempos CPU de cabeza o readout sobre características preparadas y los tiempos GPU de entrenamiento/extracción tienen scopes distintos. No se presentan como latencia completa del sistema.

`PORTABLE_CPU_FORWARD_PROFILE.json` conserva las 200 duraciones por export, después de 20 warmups, con batch1 y un hilo CPU. Excluye carga del modelo, lectura de entradas, encoder, transporte, construcción del historial y normalización. Son tiempos de pared sobre un equipo compartido, no costes de kernels ni garantías para otro hardware. El perfil se regenera con `python scripts/profile_v17_exports.py ../campaign/exports`.
