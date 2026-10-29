# Council final delta

Requiere el directorio hermano council_v1 intacto. No duplica sus blobs; MANIFEST.json vincula cada ruta final a bytes exactos de la base o del delta.

Desde este directorio, con Python stdlib:

```text
python materialize.py verify --delta . --base ../council_v1
python materialize.py materialize --delta . --base ../council_v1 --output ../council_final_materialized
```

En el árbol reconstruido, leer docs/train40_runtime_council_20261028/HANDOFF.md. El informe se regenera con scripts/report_train40_resume.py usando --source artifacts/train40_runtime_council_20261028 --output docs/train40_runtime_council_20261028/RESULTS_GENERATED.md. No se incluyen RAW, pesos ni tensores .pt. El objetivo de 50 Hz/<20 ms no está admitido.
