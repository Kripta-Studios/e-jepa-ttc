# Unidad técnica de entrenamiento OOF: fuentes y límites

`operational/final_refit/fold_training.py` prepara y valida únicamente las cabezas
PAIR y CTX internas de una rotación. No constituye un runner OOF completo y no
autoriza actualizaciones científicas.

`fold_sources.py` añade una vista estricta del rol `producer`: valida la membresía
24/8/8, conserva el ordinal nativo usado para buscar teachers y usa los caches
TRAIN40 sólo como almacenamiento. `fold_producers.py` puede comprobar de forma
recuperable hasta diez updates técnicos con pesos nuevos de semilla 7. Su receipt
permanece `IN_PROGRESS` y no es un endpoint científico.

## Fuentes vinculadas

- `artifacts/final_refit_20261017/folds/FOLD_CONTRACT.json`.
- El manifiesto y los tres parquet `producer`, `refiner` y `generation` de la
  rotación seleccionada.
- Las fuentes ejecutadas declaradas al construir cada unidad.
- Los receipts y checkpoints de cada ancestro, junto con una admisión de endpoint
  que fija rotación, rol de entrenamiento y grupos.
- El cache local de la etapa, incluida su identidad ordenada de tokens y grupos.

La unidad admite PAIR133 sólo dentro de la base latente de su propio productor.
PAIR_E y PAIR_R usan el rol `producer_pair`; E_CTX y R_CTX usan el rol disjunto
`refiner`. `generation` queda reservado para producir salidas OOF y no entrena una
cabeza PHASE por rotación.

## Bloqueos pendientes

Las llamadas científicas a `fit_fold_stage` fallan de forma explícita. Sólo se
permite un recorrido técnico acotado a cinco updates por llamada y diez durante
toda la vida de cada fit. Un receipt técnico que declare `COMPLETE` o
`scientific_endpoint=true` se rechaza y esta ruta nunca escribe una admisión de
endpoint. Faltan:

- cierre transitivo verificable desde cada consulta hasta los grupos usados por
  productores, PAIR y CTX;
- runner científico de productores por rotación y sus endpoints transitivamente
  admitidos; el adaptador técnico acotado no satisface este requisito;
- generación sobre el bloque `generation` sin acceso a sus grupos;
- stitching únicamente de PHASE17 y PHASE20, con cobertura exacta de TRAIN40;
- normalizador global sobre PHASE17 OOF y entrenamiento de la cabeza final sobre
  PHASE20 OOF;
- refits de despliegue y registro explícito del cambio de distribución entre
  features OOF y productores finales entrenados con TRAIN40 completo;
- presupuesto, endpoints y cierre de elegibilidad científica del grafo completo.

La genealogía de entrenamiento demuestra qué grupos pudieron influir en una fila
OOF. La genealogía de despliegue identifica los modelos finales usados para generar
test12. Son contratos distintos; completar uno no demuestra el otro.
