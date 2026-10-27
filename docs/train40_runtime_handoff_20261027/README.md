# TRAIN40: cierre de optimización y handoff — 27 de octubre de 2026

**Estado: PAUSADO por petición expresa del usuario. Objetivo NO conseguido.**

El usuario confirmó que la meta es **50 actualizaciones/s y menos de 20 ms por actualización completa para hasta veinte objetos simultáneos**, con pesos originales y paridad numérica. No basta una ROI, un forward preparado ni dividir el coste de un lote entre veinte. Deben medirse también cobertura, p95, p99 y acumulación de cola. Los umbrales previamente expresados fueron p95 ≤50 ms y p99 ≤100 ms; el gate implementado informa además del requisito más estricto de todas las actualizaciones por debajo de 20 ms.

El trabajo está en `perf/train40-persistent-50hz`, creada desde `2ad7a05797799caf1bcdc9202c307d244f9fe31d`. No se entrenó, no se eligieron otros pesos y no se reescribieron las predicciones congeladas. La investigación se detuvo antes de completar la validación de las 554 consultas y antes de construir un servicio multiobjeto completo.

## Leer al reanudar

1. [Estado y protocolo](01_STATE_AND_PROTOCOL.md): contratos, fronteras temporales y pendientes imprescindibles.
2. [Implementación](02_RUNTIME_ARCHITECTURE.md): módulos, interfaces, cachés, buffers y riesgos de integración.
3. [Resultados regenerados](RESULTS_GENERATED.md): cifras extraídas de los JSON publicados.
4. [Hallazgos y resultados negativos](03_FINDINGS_AND_REJECTED_PATHS.md): qué funciona, qué falló y qué no se admitió.
5. [Reproducción](04_REPRODUCTION.md): entorno Windows, comandos y compilador aislado.
6. [Plan para el siguiente swarm](05_NEXT_SWARM.md): reparto de trabajo y orden de validación.

La historia científica anterior permanece en [engineering_history_20261027](../engineering_history_20261027/README.md). Leer también su descripción del candidato y del scope de latencia. Este handoff no sustituye ni reinterpreta sus cierres.

## Qué quedó demostrado

- El piloto `integral_pilot_v1` terminó con paridad a `atol=rtol=1e-5` en las consultas originales **0, 200 y 553**, comparando PHASE_ONLY, E_CTX y R_CTX. Las representaciones de eventos fueron bit-exactas en esos tres casos. La comprobación antes/después encontró **601 archivos intactos**.
- El reloj completo de ese piloto, desde paquete RAW en RAM y frames disponibles, dio **20,323–44,124 ms para un objeto**. Incluye ROI, eventos, transferencias y toda la cadena. Excluye disco, construcción inicial del historial y priming sintético. No es una prueba sostenida ni una comparación homogénea contra los 1.612 ms offline históricos.
- Con veinte objetos de carga artificial, la ruta de productores preparada con reutilización dio aproximadamente **179–182 ms** en ejecución concurrente. Excluye entradas, H2D y cabeza de contexto. Por sí sola rebasa la meta.
- La representación GPU de veinte ROI distintas, sin modelos, dio aproximadamente **75 ms** para un paquete nuevo y su consulta. Esta prueba y la anterior tienen ámbitos diferentes; no deben sumarse como si fueran una medición conjunta.
- Inductor funciona en este Windows mediante una instalación aislada de Triton. Solo se probó un helper de correlación; no un productor compilado ni TTC final.

## Qué no está hecho

**No hay admisión completa sobre 554 consultas, no hay una prueba continua a 50 Hz, no hay paridad final para veinte objetos, ni un servicio que los lleve de sensores hasta veinte TTC.** Tampoco hay evidencia de que este hardware alcance la meta manteniendo toda la cadena y precisión. No convertir las mejoras parciales en esa afirmación.

## Evidencia y conservación

Los resultados completos siguen localmente en `artifacts/train40_runtime_20261027/`. Ese árbol está ignorado por Git y contiene además dependencias y cachés del compilador. La carpeta [evidence](evidence/MANIFEST.json) publica recibos compactos, perfiles y snapshots de scripts, con SHA-256. Los scripts exploratorios se guardan como `.py.txt`: son evidencia de lo ejecutado, no código incorporado automáticamente al runtime.

Los pesos, datos HDF5, wheel, cachés y dependencias instaladas **no se suben**. El candidato continúa en `D:/E_JEPA_POD2_RETURN_20261027/FINAL_CANDIDATE`. La rama original y los módulos científicos congelados no se alteran.

Al cerrar no queda un entrenamiento ni un benchmark GPU solicitado en ejecución. Reanudar solo cuando el usuario lo pida; el objetivo de optimización quedó pausado, no completado.

## Comprobaciones del cierre

Suite focalizada del runtime y tests existentes de transferencia final: **180 passed, 3 skipped, 2 xfailed**. Los omitidos requieren activación CUDA explícita; los xfail estrictos mantienen la evidencia negativa de deduplicación de pares. Ruff check/format pasan y Pyright devuelve cero errores con el entorno hermano. Ver los [logs de cierre](evidence/closeout/pytest.txt) y la [preservación verificada](evidence/closeout/preservation.json).

Los commits separan primitives, entradas persistentes, grafos, ring, medición y experimento de transporte. El commit documental final y el SHA publicado se consultan con `git log -1` y `git rev-parse origin/perf/train40-persistent-50hz`; no se anticipa aquí un hash del propio documento.
