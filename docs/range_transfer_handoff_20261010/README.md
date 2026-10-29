# RANGE-TRANSFER — continuación propuesta, 10 de octubre de 2026

**Decisión:** conservar V12 como referencia y terminar V13 sin modificar su entrenamiento. Investigar después el sesgo por rango con un experimento pequeño y factorial sobre cabezas del sistema event-only V13, productores finales congelados y roles P/H/V existentes.

Este paquete no acredita entrenamiento nuevo, una mejora RGB final ni SOTA. El código incluido implementa pesos TRAIN y una pérdida adicional, no un trainer de producción. Su envío explícito a Codex como autorización nueva activa únicamente la cola delimitada aquí. El estado vivo lo tiene el host del usuario.

## Lectura

1. [Revisión y hallazgos](docs/01_REVIEW.md).
2. [Atribución y comprobaciones antes de entrenar](docs/02_DIAGNOSTICS.md).
3. [Protocolo factorial: masa × pérdida](docs/03_EXPERIMENT.md).
4. [Integración archivo por archivo](docs/04_INTEGRATION.md).
5. [Rendimiento, RGB, datos y evaluación posterior](docs/05_ROADMAP.md).
6. [Recuperación y entrega](docs/06_EXECUTION.md).
7. [Fuentes y alcance de verificación](docs/07_SOURCES_SCOPE.md).
8. [Política prospectiva](execution_policy.json), [arranque](PROMPT_CODEX_START.md) y [reanudación](PROMPT_CODEX_RESUME.md).

## Qué hacer ahora

Mantener la sesión V13 como propietaria del entrenamiento. Usar una sesión/worktree nuevo para lectura, análisis e implementación ligera. Entrenar las nuevas cabezas sólo cuando E_A5_MATCHED, E_C2F_MATCHED, PAIR_E_MATCHED y la caché H final estén completos y admitidos. No esperar a FCWD RGB, respuesta de autores o score test12 para ejecutar las ramas independientes.

La cola nueva tiene hasta 6 fits iniciales × 2.500 updates y 4 réplicas condicionadas × 2.500: **25.000 científicos, 500 técnicos sintéticos y 4.500 de recuperación; techo físico 30.000**. No consume permiso residual de otros planes. No se reasigna ahorro a nuevas hipótesis.

## Reproducir el análisis entregado

Con el ZIP original del usuario ya extraído, `V12/` es un snapshot completo de código sin `.git`:

```bash
python scripts/verify_package.py
python scripts/analyze_public.py --v12-root /ruta/al/handoff_original/V12 --output /ruta/a/resultados_nuevos
python scripts/run_contract_tests.py --repo /ruta/al/handoff_original/V13
```

Los dos últimos comandos no ejecutan entrenamientos ni cargan los grandes checkpoints. La suite de contratos usa los módulos de fase/modelo fijados en V13 con importación aislada; no sustituye los tests de integración del repositorio.

## Qué no hacer

No añadir esta pérdida a un fit V13 a medio ejecutar. No normalizar de nuevo con FCWD. No presentar la mediana de tres cabezas como tres entrenamientos independientes de los productores. No llamar `overall_MiD` al promedio FCWD, que carece de negativos. No sustituir V13 por otra gran arquitectura antes de disponer de sus resultados.
