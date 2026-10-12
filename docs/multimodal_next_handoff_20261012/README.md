# Continuación multimodal tras V13 — 12 de octubre de 2026

## Decisión

Conservar los doce fits V13 completos y el cierre negativo V14. No repetir RGB-PORT, la matriz de masas ni los arreglos de continuidad. Coordinar V15 RANGE-READOUT, si fue iniciado, sin duplicar su autorización o alterar sus padres.

La siguiente extensión ejecutable propuesta es **comparación multimodal, atribución y perfilado con pesos congelados**. Cero actualizaciones de optimizador. Su objetivo es cerrar la comparación que falta contra Garl full/visual-only en FCWD, distinguir información RGB de su disponibilidad, y medir cuándo la fusión ayuda respecto al RGB propio. Las alternativas de entrenamiento de `docs/03_NEXT_TRAINING.md` son propuestas posteriores, no una cola automáticamente autorizada.

## Lectura

1. `docs/01_REVIEW.md`: resultados, reproducción y nuevas inferencias descriptivas.
2. `docs/02_EXECUTION.md`: tareas concretas, controles y entregables.
3. `docs/03_NEXT_TRAINING.md`: decisiones de investigación después del cierre.
4. `PROMPT_CODEX_START.md`: mensaje operativo para la sesión local.
5. `execution_policy.json`: alcance, permisos y coordinación.

Base revisada: `62cf4dbe914ea077c68e42983f869cd5422ab71c`, rama histórica `scientific-recovery-v13-rgb-port`.

No es un trainer nuevo, ni una reparación ya aplicada al repositorio. Los scripts de este paquete sólo calculan estadísticas sobre los CSV publicados y verifican archivos. La red y los medios no están incluidos.

## Ejecutar los recálculos del paquete

```powershell
python -B scripts/verify_package.py
python -B -m unittest discover -s tests -v
python -B scripts/analyze_public.py --output E:/EJEPA_results/multimodal_public_review_20261012
```

Use un directorio de salida nuevo o vacío. No hay downloads, CUDA, lecturas de checkpoints ni entrenamiento en esos scripts.

## Qué se verificó aquí

ZIP de entrada: 22.483.682 bytes, SHA-256 `22fabc28119c21a314ebf1f857f78e5439282dda8d552bd54d0626e81aea35b2`. CRC y 6.459 entradas del manifiesto verificados; 6.460 miembros incluyendo manifiesto. Nueve documentos contrastados con sus SHA-1 de blobs Git. Replay publicado: 29 fuentes, 9.471 comprobaciones escalares, sin inferencia nueva.

Clonación fallida por DNS. El árbol textual extraído no incluye `.git`, datos ni pesos, y no se presenta como clon completo. Se revisaron los ocho informes, código de fusión, entrenamiento de cabezas, registro y adaptación; comprobar hashes de miles de archivos no equivale a leer científicamente todas sus líneas.

`evidence/` separa resultados publicados, recálculos propios y bootstrap post hoc. Los CSV originales se copian sin cambios. No son nuevos resultados del modelo.

Reconciliación de las cohortes históricas usando el ZIP original extraído:

```powershell
python -B scripts/reconcile_cohorts.py --snapshot-root RUTA_AL_ZIP_EXTRAIDO --output E:/EJEPA_results/cohort_identity.json
```

El bootstrap descriptivo adicional usa NumPy y Pandas del entorno existente:

```powershell
python -B scripts/paired_bootstrap.py --output E:/EJEPA_results/multimodal_paired_bootstrap
```

Son 4.000 remuestreos de secuencias completas, semilla20261012, media micro de sumas de pérdida/sumas de consultas. No corrige multiplicidad, no aporta seeds de entrenamiento y no convierte FCWD en más de tres secuencias. No instalar/actualizar bibliotecas globales para regenerarlo.
