# Después de V14: VARIANZA Y RANGO DEL READOUT
Fecha: 12 de octubre de 2026. Propuesta nueva; V14 permanece NO_CANDIDATE.

## Decisión
No repetir el factorial M0/M1/M2 × L0/L1 ni promover una receta rechazada.
Investigar dos mecanismos separados: (S) varianza introducida por el muestreo
uniforme con pesos y clipping; (R) límite del residual de A5, con geometría congelada.
Preservar V13; sus resultados RGB se revisan cuando existan endpoints completos.

Orden: `docs/01_REVIEW.md`, `docs/02_PROTOCOL.md`, `docs/03_IMPLEMENTATION.md`,
`execution_policy.json`, `PROMPT_CODEX_START.md`.

`reference/` contiene operadores de prueba, NO un trainer integrado. Las pruebas
sintéticas no demuestran mejora TTC. El auditor de `scripts/analyze_v14.py` usa
NumPy/Pandas; los contratos usan PyTorch. No se instala o actualiza el entorno global.

La autorización propuesta se activa sólo si el usuario envía el prompt a Codex.
El saldo histórico no autoriza esta campaña. Máximo científico nuevo 56.000;
500 técnicos y 3.500 de recuperación; 60.000 físicos. Ramas condicionales, sin obligación
de consumir el techo ni ampliar estudios si fallan.

## Evidencia revisada
Commit V14 `1c251781ac845a5d661523e607e0f7c4a8d3af8a`.
ZIP fuente `c540a53a4ebdcc4dc047d6b29340e429468ce77ce5602e938ad2292081a3dc71`.
Se verificaron sus 772 miembros / 771 entradas de manifiesto. Se recalcularon métricas desde
98.592 filas de predicciones guardadas, NO las inferencias. La caché V13 no está incluida.
La clonación HTTPS falló por DNS; se contrastaron los seis documentos vía GitHub.

## Uso
1. Crear worktree separado con `scripts/Prepare-Worktree.ps1`.
2. Dar `PROMPT_CODEX_START.md` al agente.
3. No ejecutar el runner V14 cerrado. Codex debe implementar el runner V15 y
   verificar su CLI con `--help` antes de publicar comandos reales de ejecución.
4. Ante pausa, conservar fuentes y usar `PROMPT_CODEX_RESUME.md`.
