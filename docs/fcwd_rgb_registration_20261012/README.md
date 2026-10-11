# FCWD: adaptación RGB mediante registro local de sensores

Se ha implementado una ruta adicional para evaluar los modelos RGB y de fusión
V13 en FCWD. Usa los intrínsecos publicados y estima un desplazamiento local
con los píxeles RGB y eventos disponibles en cada consulta. No requiere entrenar
pesos ni leer etiquetas TTC. **No constituye una calibración extrínseca métrica**
y sus resultados serán exploratorios. La evaluación histórica permanece intacta.

## Problema encontrado

FCWD proporciona cajas en coordenadas de la cámara de eventos derecha, pero la
imagen Blackfly derecha tiene otra resolución y otra óptica. Escalar coordenadas
por 1920/1280 y 1200/720 no representa esa geometría. El archivo extrínseco
disponible corresponde a LiDAR→RGB, no a eventos→RGB.

Se consiguió decodificar `calibParams.mat`, que contiene objetos MCOS MATLAB,
con `mat-io==1.0.1`. Los campos nombrados `APScameraParams` y `DVScameraParams`
incluyen dos calibraciones intrínsecas:

| Campo | RGB | Eventos |
|---|---:|---:|
| Resolución | 1920×1200 | 1280×720 |
| fx, aproximadamente | 4647,49 | 3292,34 |
| fy, aproximadamente | 4649,56 | 3296,06 |
| k1 | −0,004178 | −0,069012 |
| k2 | 0,484196 | 1,486342 |

El código usa los valores completos del archivo, transpone `IntrinsicMatrix`
y convierte el centro principal de píxeles MATLAB a coordenadas de base cero.
No empareja las poses de tableros: hay 131 vistas RGB y 33 vistas de eventos,
sin una correspondencia publicada que permita deducir extrínsecos comunes.

Fuentes: https://nail-hnu.github.io/EventAidedTTC/

https://github.com/NAIL-HNU/event_aided_ttc

https://github.com/foreverallama/matio

## Transformación implementada

Código: `operational/fcwd_rgb_registration/geometry.py`.

1. Corregir distorsión y proyectar RGB al plano óptico de eventos usando K_RGB y K_E.
2. Construir conteos de eventos del intervalo causal de 20 ms que termina en el
   timestamp de la imagen RGB disponible. Separarlo en dos mitades de 10 ms.
3. Registrar el patrón `log1p` de conteos contra el gradiente RGB, con CLAHE,
   desenfoque σ=1,5 y correlación normalizada dentro de una búsqueda local.
4. Exigir al menos 100 píxeles de soporte, NCC≥0,22 en el paquete y sus dos
   mitades, desacuerdo entre mitades≤5 píxeles y solución fuera del límite de búsqueda.
5. Proyectar el contorno de las tres cajas causales de eventos a RGB original,
   incluyendo distorsión: 33 puntos por lado. Construir una ROI cuadrada común
   con margen de 25%.
6. Mantener **exactamente esa ROI y transformación para toda la historia**. Se
   conservan la expansión visual entre frames y los deltas temporales reales.

La historia procede de `blackflys/right/ts`, con aproximadamente 100 ms entre
frames y tolerancia de 1 ms, tripletas reales y soporte completo≤650 ms.
La ROI retrospectiva se declara disponible en la consulta, no en cada instante
histórico. La rama de eventos conserva cuatro observaciones a 100 ms y tres
ventanas por observación.

Un registro rechazado **no elimina la consulta**: las cabezas RGB quedan sin
predicción; F_TRUE y F_ZERO usan explícitamente E_CTX como fallback. Se publican
tanto todos los casos como el subconjunto común con RGB disponible. Garl event
y H8 se comparan en las mismas filas. No se inventa un comparador Garl multimodal
si sus predicciones históricas no están disponibles.

## Comprobaciones y experimentos negativos

- El cambio de resolución, ECC y SIFT no ofrecieron una transformación fiable.
- Un ajuste afín global sobre bordes falló el criterio geométrico definido en
  `PROTOCOL.json`; se conservan `FIT_FROZEN.json` y `PILOT_RESULT.json`. No se usa
  ese ajuste en inferencia.
- El desfase vertical depende de la proximidad: el registro local lo permite sin
  derivarlo de TTC, LiDAR ni profundidad.
- En 60 consultas elegidas por hash de ID: 47 aceptadas (16/20 FCWD1, 16/20 FCWD2,
  15/20 FCWD3), desplazamiento horizontal entre −26 y −21 píxeles, máximo
  desacuerdo entre mitades de 4,1231 píxeles. Es una revisión operativa sobre
  entradas del dominio destino, **no una validación independiente de precisión
  geométrica ni TTC**. NCC y acuerdo temporal no son error contra correspondencias GT.
- Regresiones: convenciones MATLAB, proyección óptica, geometría inválida, falta
  de textura, recuperación de una traslación sintética conocida, serialización
  y tiempos reales FCWD. Suite conjunta inicial: 26 pruebas aprobadas; Pyright
  cero errores y advertencias. `tests/test_fcwd_rgb_registration.py`.

Limitaciones: paralaje dependiente de profundidad, oclusiones, movimiento durante
20 ms, objetos recortados por el sensor, falta de correspondencias geométricas
GT, cajas de objeto proporcionadas por el dataset y adaptación sobre entradas
del dominio destino. No afirmar equivalencia con extrínsecos oficiales ni SOTA.

## Campaña completa iniciada

El registro de las **630 consultas** terminó: **495 aceptadas y 135 rechazadas**
por correlación débil o inconsistencia entre mitades. Por secuencia:

| Secuencia | RGB aceptado | Total |
|---|---:|---:|
| FCWD1 | 164 | 214 |
| FCWD2 | 173 | 218 |
| FCWD3 | 158 | 198 |

`REGISTRATION_SEALED.json` recoge cobertura y hashes de los 630 recibos.
`PUBLISHED_INTRINSICS.json` contiene los intrínsecos extraídos.
`REAL_INPUT_HISTORY_QA.json` verifica historias reales en tres consultas, incluidos
casos próximos y de confianza baja: cinco observaciones con tripletas causales,
ROI idéntica durante la historia y tensores `[3,3,128,128]`.

La inferencia CUDA ha producido y guardado resultados reales tanto para RGB
registrado como para fallback de eventos. `REAL_INFERENCE_QA.json` conserva un
ejemplo de cada ruta y comprueba salidas finitas y fallback exacto. **La inferencia
completa y sus métricas todavía estaban en ejecución al publicar esta nota.**
No se han usado esas salidas para ajustar el registro.

El estado vivo está en `campaign/STATE.json`; los fragmentos permiten reanudar.
El resultado final se escribirá en `campaign/predictions/RESULT.json`, después
de sellar las 630 predicciones. No confundir `REGISTRATION_SEALED.json` con una
evaluación TTC terminada.

## Ejecución y trazabilidad

Raíz de código: repositorio `e-jepa-ttc-v13-rgb-port`.

Datos: `E:/FCWD_dataset/inputs`.

Experimentos, dependencias aisladas, imágenes de inspección, QA y registros:
`E:/EJEPA_results/fcwd_rgb_calibration_20261012`.

La instalación aislada utiliza `opencv-python-headless==5.0.0.93` y
`mat-io==1.0.1`. No cambia el entorno compartido de entrenamiento.
Con esas dependencias y `src` en PYTHONPATH:

```powershell
python -B -m operational.fcwd_rgb_registration.campaign `
  --run artifacts/rgb_port_20261008 `
  --manifest ../e-jepa-ttc-v12-efficient-context/artifacts/sota_campaign_20261008/fcwd_population/QUERY_MANIFEST.json `
  --calibration E:/FCWD_dataset/inputs/calibration/camera_intrinsic/calibParams.mat `
  --output E:/EJEPA_results/fcwd_rgb_calibration_20261012/campaign `
  --comparator ../e-jepa-ttc-v12-efficient-context/artifacts/sota_campaign_20261008/fcwd_inference/scoring/SCORED_PREDICTIONS.csv `
  --device cuda
```

`prepare.py` sella todos los registros y sus hashes antes de inferir. `transfer.py`
reutiliza el motor de endpoints congelados, con adaptadores de entrada acotados
y las correcciones admitidas de máscara y fusión. Conserva los doce checkpoints,
los normalizadores y la receta. `score.py` abre los comparadores/etiquetas solo
tras comprobar el sellado de todas las predicciones. `campaign.py` encadena las
fases y reintenta errores de almacenamiento; errores de contrato no se ignoran.

Los resultados MiD por bandas y grupo no se denominan `overall_MiD` oficial.
Si falta una banda, no se imputa. Esta ruta no modifica ni desbloquea falsamente
el marcador histórico que exige calibración oficial; produce evidencia adicional
con otro contrato y salidas separadas.
