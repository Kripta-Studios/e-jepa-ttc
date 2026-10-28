# Council v1 — snapshot de investigación activa

Resultados y alcance: [COUNCIL](docs/train40_runtime_council_20261028/COUNCIL.md) y
[tablas regeneradas](docs/train40_runtime_council_20261028/RESULTS_GENERATED.md).

Este paquete conserva recibos positivos y negativos, fuentes exactas disponibles,
documentación y generadores. No contiene RAW, pesos, dependencias, caches ni .pt.
Los diagnósticos cuyos recibos referencian tensores .pt necesitan esos artefactos
externos para volver a ejecutarse; aquí se pueden regenerar sus tablas JSON.
No es un cierre ni una admisión de 50 Hz/<20 ms. El objetivo permanece activo.

Desde la raíz del paquete, con Python (sólo stdlib para el informe):

```text
python scripts/report_train40_resume.py --source artifacts/train40_runtime_council_20261028 --output docs/train40_runtime_council_20261028/RESULTS_GENERATED.md
python artifacts/train40_runtime_council_20261028/frame_sealing_cpu/write_findings.py.txt
python artifacts/train40_runtime_council_20261028/rgb_exact_gpu/write_findings.py.txt
python docs/train40_runtime_council_20261028/build_linspace_attribution.py
```

Los archivos .py.txt son snapshots legibles que Python puede ejecutar con esa
extensión. Las rutas absolutas en recibos se conservan como procedencia histórica;
los generadores anteriores leen los recibos locales relativos. La reproducción de
GPU/datasets requiere el checkout y sus datos/pesos externos verificados.

MANIFEST.json vincula todos los bytes del paquete excepto el propio manifiesto.
El manifiesto de artifacts registra recuperaciones exactas desde Git y cualquier
fuente histórica no recuperada; nunca se sustituye por una versión distinta.
QA/propuestas en curso y dependencias de descarga quedaron fuera de esta ronda.
