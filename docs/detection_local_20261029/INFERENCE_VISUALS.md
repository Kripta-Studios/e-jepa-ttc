# Visualización de predicciones guardadas

Fecha: 2026-10-29. El
[renderer](../../scripts/render_eap_detection_predictions.py) se ejecutó sobre
`initial_train.json` y `final_train.json` del experimento local
`output/portable_experiments/overfit16_seed7_v1`. No carga Torch ni ejecuta el
modelo; representa predicciones ya guardadas sobre las imágenes RGB originales.

## Artefactos ejecutados

La comparación queda exclusivamente en:

```text
output/portable_experiments/overfit16_seed7_v1/visuals_comparison/
  frame_00.png
  frame_01.png
  frame_02.png
  frame_03.png
  contact_sheet.png
  training_curve.png
  visual_receipt.json
```

Son cuatro frames elegidos mediante posiciones `linspace` en el orden declarado
del manifest de overfit: índices 0, 5, 10 y 15. No se seleccionaron según el
resultado visual ni la precisión. Cada fila compara exactamente la misma imagen
y el mismo ground truth inicial/final. El código falla si difieren las etiquetas
guardadas entre ambos artefactos, el manifest de medios o el hash de los bytes RGB.

Se dibuja en verde el ground truth retenido por el adaptador, y en magenta las
predicciones con confianza al menos 0,2. El texto diferencia ambas poblaciones
y sus conteos. Ese filtro afecta únicamente al dibujo: no modifica JSON, métricas
ni la población usada por AP. Las proyecciones no dibujables quedan registradas
con su razón, sin convertirlas en cajas ficticias.

La imagen utiliza su resolución original de 1280×720 y `K_event`/`T_event_ego`
originales. Las dimensiones nativas `[l,w,h]` se convierten a la disposición
original `[x,y,z,l,h,w,yaw]` antes de usar el proyector geométrico inspeccionado.
La advertencia visible recuerda que el overlay pinhole no certifica rectificación
física RGB/eventos.

La curva contiene las 344 filas reales de `training.jsonl`, sin suavizado: pérdida
ponderada, pérdida de centro y segundos por step. Los samples alternan durante
entrenamiento, por lo que cada punto no representa una evaluación repetida sobre
la misma imagen. La gráfica de tiempo recoge la ejecución observada con su carga
concurrente; no sustituye a un perfil aislado para extrapolar presupuesto.

## Inspección visual

Se inspeccionaron `frame_00.png` y `training_curve.png`. En el primer frame la
caja GT se proyecta sobre el vehículo anotado. La salida inicial no contiene
predicciones decodificadas; la salida final contiene 50 cajas guardadas y muestra
11 por encima del umbral visual. Varias están claramente desplazadas respecto al
objeto. Esta comparación ilustra por qué el descenso de pérdida no basta para
considerar útil el detector. Las figuras no acreditan generalización ni puntuación
oficial del benchmark.

`visual_receipt.json` conserva los tokens, hashes de ambos JSON de predicciones,
manifest, imágenes RGB, proyector, renderer y PNG generados, junto con los conteos
por frame. Los medios y overlays no se incorporan a Git.

## Reproducción

```powershell
& '../phase-ttc-main-integration/.venv/Scripts/python.exe' -X utf8 `
  scripts/render_eap_detection_predictions.py `
  --eap-root E:/eAP_dataset --manifest output/eap_rgb_pilot/manifest.json `
  --role overfit16 `
  --predictions output/portable_experiments/overfit16_seed7_v1/initial_train.json `
    output/portable_experiments/overfit16_seed7_v1/final_train.json `
  --labels Initial Final --score-threshold 0.2 `
  --training-log output/portable_experiments/overfit16_seed7_v1/training.jsonl `
  --output output/portable_experiments/overfit16_seed7_v1/visuals_recheck
```

El renderer rechaza directorios de salida existentes y destinos fuera de
`output/`. Para una evaluación selection posterior basta proporcionar su JSON y
`--role selection`; no presupone que ese resultado ya exista.

Seis tests pasan: selección sin sesgo por resultado, orden de dimensiones,
clipping de líneas, umbral exclusivamente visual, preservación de entradas y
registro de cajas detrás de la cámara. Ruff y formato pasan en el script y sus
tests.

## Inferencia independiente sin abrir etiquetas

El [CLI de inferencia](../../scripts/infer_monodgp_eap.py) se ejecutó en procesos
nuevos con el checkpoint v1, carga `weights_only=True` y `load_state_dict(strict=True)`.
Su proveedor RGB sobrescribe la lectura de etiquetas y devuelve una lista vacía
sin consultar `labels_path`. La preparación recibe esa lista vacía y el forward
usa explícitamente `targets=None`; no se introduce una ROI GT ni un ID GT.

Resultados guardados, exclusivamente bajo `output/portable_experiments/overfit16_seed7_v1/`:

| Archivo | Ejecución real |
|---|---|
| `inference_label_free16.json` | 16 frames; paridad exacta con las predicciones de `final_train.json`, error máximo 0 con tolerancia absoluta 1e-4 |
| `inference_temporal7.json` | Siete frames consecutivos de `qGsgzl4Q8B`, sin ground truth de salida; destinados a construir historia de ROIs predichas |

La mediana del forward GPU sincronizado fue 49,61 ms para los 16 frames y
48,11 ms para los siete. Son tiempos del forward, separados de lectura,
preprocesado y decode. No incluyen PHASE ni representan latencia completa del
sistema. Los recibos conservan cada componente, dos forwards de calentamiento,
memoria, versiones, hash de checkpoint, fuentes, RGB y configuración de precisión.
No se emite velocidad ni se declara compatibilidad de submission oficial.

La comparación opcional usa únicamente el campo de predicciones del JSON de
referencia, cargado después de completar la inferencia. No participa en los inputs del
modelo. Un test carga RGB con una ruta de etiquetas inexistente y permite que
`read_parquet` acceda únicamente al manifest, demostrando que la ruta de lectura
de GT no es necesaria.

La fuente exacta que produjo los dos recibos anteriores está conservada en
`inference_code_v1/infer_monodgp_eap.py` dentro del run. Posteriormente se amplió
el resolvedor para cargar métricas, runtime de datos, atención portable, adaptador
y helper del detector desde sus fuentes congeladas, en ese orden y bajo nombres
canónicos de módulo. Verifica todos los hashes registrados antes de ejecutar
ningún módulo, y los vuelve a comprobar al cargarlos. Si no existe snapshot,
exige que la fuente actual coincida exactamente. Un snapshot alterado se rechaza.
La sobrescritura que impide abrir etiquetas se conserva sobre la clase del loader
realmente cargado. Esto permite reproducir v1/v2 mientras el adaptador evoluciona,
sin tratar código diferente como equivalente ni mezclar versiones entre imports.

## Comparación ejecutada tras la continuación v2

También se ejecutó el renderer para el mismo estado inicial de v1 frente al
checkpoint final de `overfit16_seed7_v2_resume`. Los cuatro tokens siguen siendo
los mismos y se verificó que el GT y los bytes RGB coinciden. Los PNG y su recibo
están en `output/portable_experiments/overfit16_seed7_v2_resume/visuals_comparison/`.
La curva de ese directorio representa los 456 pasos adicionales de la
continuación, no los 800 pasos acumulados completos.

El `report.json` de v2 registra ratio de pérdida de centro **0,345219** respecto
al inicio aleatorio v1 y aceptación funcional `PASS`. En los mismos 16 frames de
entrenamiento, la métrica de desarrollo AP101 sobre seis clases con soporte es
**0,201421**; ATE es **1,08117 m**, ASE **0,190580** y AOE **0,083536 rad**.
Estas cifras son de memorización sobre train, bajo el perfil de desarrollo
declarado, y no una evaluación oficial ni una estimación de generalización.

La inspección de `frame_00.png` de v2 sigue mostrando múltiples predicciones
desplazadas alrededor de un único GT: nueve cajas dibujadas con confianza ≥0,2
de 50 guardadas. Aprobar el criterio funcional de entrenamiento no convierte
estas figuras en evidencia de un detector listo para uso.

La inferencia independiente de v2 también se ejecutó tras cambiar el adaptador
actual del proyecto: el resolvedor cargó las cinco fuentes congeladas y verificó
sus hashes, obtuvo paridad exacta en los 16 frames (error máximo 0) y una mediana
del forward sincronizado de **44,26 ms**. El recibo es
`overfit16_seed7_v2_resume/inference_label_free16_frozen.json`; la fuente exacta
ejecutada está en `inference_code_frozen16/infer_monodgp_eap.py` del mismo run.

El CLI actual admite además los flags de construcción registrados para una
variante futura con cabeza de velocidad. La carga del checkpoint sigue siendo
estricta y no inicializa pesos externos de forma implícita. La comparación de
paridad comprueba también presencia y valores de `velocity_mps` si ese campo
existe. Cuando están registradas, también verifica y carga las fuentes congeladas
de geometría, diagnósticos de velocidad, extensión de velocidad e inicialización
de backbone. La variante de velocidad exige la extensión registrada; importar el
helper del backbone no ejecuta la inicialización ImageNet.

Este soporte de código no se presenta aquí como una ejecución del piloto
con velocidad. Ocho tests del CLI pasan, incluidos orden de carga canónico,
rechazo de snapshots alterados antes de ejecutar código, acceso RGB sin etiquetas
y comparación de velocidad.
