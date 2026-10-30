# Paquete de datos para la continuación hasta release

Preparación local solicitada el 30 de octubre. No despliega Pods, transmite datos,
ejecuta entrenamiento ni autoriza gastos. El estado de la investigación continúa
siendo el del plan S0–S6; preparar los datos no convierte el candidato en un modelo
final ni acredita generalización TTC.

## Pausa preservada

R0, R1 y R2, semilla 7, están completos. El candidato de screening es R2.
La confirmación R0/13 se detuvo durante la evaluación de 2048. Su checkpoint
completo recuperable es 1792 y el seleccionado 1024. Las 256 actualizaciones
registradas posteriores no están en ese checkpoint y deben repetirse al resumir.
Los checkpoints, curvas, recetas y receipts se incluyen en el paquete.

## Datos y particiones

El inventario público se fija a NAIL-HNU/eAP-dataset,
revisión a814a946e60d37f7242358af5f344542177e50c5:

- 40 secuencias, con roles conservados: 24 train, 8 selection, 8 calibration.
- 136 TAR RGB: 127.088.947.200 bytes.
- 40 H5 de eventos: 448.999.857.897 bytes.
- 41 archivos de metadatos/anotaciones: 132.362.021 bytes.
- Datos físicos disponibles para D4096, D16384, Dfull, historia causal y eventos.
  Solo D4096 tiene aquí su admisión de píxeles completa; ampliar el muestreo exige
  ejecutar el preflight correspondiente y conservar las exclusiones justificadas.
- No se incorpora test como entrenamiento ni se adopta un nuevo split 24/8/4/4.
  Las 40 secuencias del release público no resuelven la discrepancia con las 46
  descritas en el artículo.

Se añaden las anotaciones Garl, el INPUT cache row-local de 40 secuencias
(44.004.127.727 bytes), el teacher TRAIN24, sus pesos externos congelados, los
folds independientes, máscaras de historia válida, admisión H5, contratos y
fuentes. La presencia física de INPUT TRAIN40 no autoriza usar selection o
calibration para fit: cada consumidor debe respetar sus listas de tokens y roles.
No se adopta un productor aprendido TRAIN40 como padre de un refit independiente.

## Entrega y recuperación

`prepare_runpod_release_bundle.py` genera MANIFEST.json, archivos privados TAR
de aproximadamente 4 GiB, PARTS.json y BUILD_RECEIPT.json. Todos los privados se
hashean contra sus bindings cuando existen. Los datos públicos se inventarían
con tamaño/presencia local y SHA-256 del release: deben verificarse íntegramente
al descargarlos o subirlos al Pod. No se afirma que se rehashearon localmente los
576 GB de medios públicos durante esta preparación.

`runpod_release_transfer.py` extrae únicamente archivos regulares declarados,
rechaza traversal y cambios de hash, descarga archivos públicos con revisión
fijada y reanuda transferencias HTTP con validación del rango. El perfil `rgb`
trae metadatos y todos los RGB; `all` añade todos los eventos. Los TAR privados
se conservan hasta que se confirme la verificación del destino.

La entrega conserva los bytes originales de manifests y evidencia. Se incluye
un mapa de raíces para sus referencias Windows; los paths históricos no se
reescriben silenciosamente. Las referencias históricas no resueltas se registran
en el manifiesto y deben distinguirse de las dependencias del comando inmediato.

## Arranque previsto

La primera ejecución cloud tendrá admisión de entorno, verificación de todos
los archivos RGB, validación CPU de población/receta y un smoke de 16 updates.
El smoke es ejecución y perfil, no confirmación de precisión ni paridad bit a bit
entre Windows/Blackwell y Linux/Ada.

`runpod_release_cloud.py` conserva la receta científica y genera configuraciones
con rutas del destino. Propone seis réplicas nuevas emparejadas R0/R2 con semillas
7,13,23 para evitar mezclar silenciosamente hardware en la confirmación. Preserva
todos los resultados locales; no convierte un checkpoint Windows a otro runtime
relajando hashes. La reanudación cruzada de esos checkpoints requiere una
migración de procedencia y validación separadas, y no es el arranque automático.

Los gates de la cola comprueban config, checkpoints, evaluación completa y
población. La cola se detiene ante errores y exige un directorio nuevo; no hace
reintentos infinitos. Puede bloquearse la siguiente ejecución mediante
`cloud_runs/STOP_AFTER_CURRENT`.

Las fases geométricas, temporales, eventos y PHASE independiente tienen sus datos
y fuentes preparados, pero su selección y admisión de entrenamiento siguen
dependiendo de resultados. R_A5 aún requiere validar su primer batch real y el
refit PHASE necesita portado de rutas/admisión GPU. No existe una orden honesta
que ejecute automáticamente S0–S6 completos y produzca hoy el modelo final.

## Recursos del Pod

El usuario ha elegido preparar una estrategia de 700 GB. Se descargan primero
RGB/metadatos y luego eventos por rol: train, selection y calibration. Se conserva
una reserva de 50 GiB; si una fase no cabe, se rechaza la descarga antes de empezar.
Los duplicados TAR del Pod pueden eliminarse después de extraer y verificar sus
payloads. No se promete conservar simultáneamente todos los datos, archivos de
transporte, entornos y futuros resultados dentro de 700 GB. Para
reducir presión del evaluador, preferir al menos 48–64 GiB de RAM y CPU/disco
rápidos. Los tamaños definitivos se obtienen del manifest/receipt del paquete,
no de estimaciones de compresión. Docker/Pod, acceso SSH y velocidad de red se
verificarán después de que el usuario autorice el despliegue.

Se conservan entornos separados detector/PHASE porque sus versiones de Pillow,
pandas y PyArrow difieren. Los requisitos se fijan a las versiones locales
observadas; la instalación y compatibilidad de wheels Linux quedan por verificar
en el Pod. El bootstrap no inicia entrenamiento por sí mismo.
