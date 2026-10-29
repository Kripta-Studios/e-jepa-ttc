# Admisión acotada: detector RGB eAP con backend portable

Fecha: 2026-10-29. Esta decisión habilita experimentos internos con RGB eAP y
cajas 3D. No cambia los recibos históricos ni declara resueltos el scorer privado,
el puente GarlTTC–PHASE o la integración causal multimodal.

## Por qué se puede entrenar ahora

El contrato G0 original agrupaba requisitos del sistema completo. Para entrenar
un detector con imágenes y etiquetas de la misma publicación eAP no hacen falta
IDs GarlTTC, etiquetas TTC, eventos ni inferir el significado físico de velocidad.
La dependencia necesaria es una correspondencia fiable entre imagen, calibración
y cajas eAP, con su transformación de targets e inferencia comprobada.

La [resolución geométrica](GEOMETRY_RESOLUTION.md) aporta 17.611 pares idénticos
original/HF y reproyecciones reales. El [runtime](MEDIA_PILOT.md) verifica los 512
RGB concretos del piloto y su correspondencia con los manifests. El adaptador
[`monodgp_eap_adapter.py`](../../src/e_jepa_ttc/detection/monodgp_eap_adapter.py)
tiene casos analíticos de ejes, dimensiones no cúbicas, centro geométrico, resize,
yaw con extrínsecas inclinadas e inferencia sin etiquetas. Una revisión matemática
independiente no encontró fallos materiales en esa conversión.

## G0_RGB_BOX_V1: alcance admitido

| Parte | Decisión y límite |
|---|---|
| Población | Public train únicamente, revisión `a814a946e60d37f7242358af5f344542177e50c5` |
| Contrato 3D | Centro ego geométrico; `bbox_3d_ego=[x,y,z,l,h,w,yaw]`; `Rz(-yaw)`; largo en Y local, ancho en X |
| Target del detector | `h,w,l` absolutos, media de clase cero; profundidad Z de cámara; alpha respecto al rayo del centro 3D |
| Decode | Centro geométrico, sin desplazamiento KITTI a la base; inversión del plano ego para yaw; dimensiones internas `l,w,h` |
| Cámara | Pinhole del visualizador oficial sobre el RGB remapeado al plano evento; aproximación documentada, no rectificación perfecta |
| Resize | Bilineal Pillow; intrínseca transformada con escalas X/Y y desplazamiento de centros de píxel |
| Entrada | Imagen completa e intrínseca; inferencia con `targets=None`; ninguna ROI o identidad GT entra al modelo |
| Targets válidos | Centro de cámara en `[0.1,60)` m, todas las corners por delante de 0,1 m, centro proyectado visible, bbox proyectada >1 px |
| Capacidad | Máximo 50 objetos admitidos por frame; excederlo provoca error, nunca truncamiento silencioso |
| Velocidad/TTC | Cabezas ausentes en esta variante; no se inventan valores ni se genera una submission incompleta |

El [recibo de admisión de los 16 frames](evidence/overfit_target_admission.json)
registra 80 anotaciones originales: 47 admitidas, 29 excluidas por profundidad y
cuatro por centro proyectado fuera de imagen. Las 47 cubren seis clases. Los
conteos del piloto completo se registran con cada ejecución, después de aplicar
exactamente el mismo perfil. Las métricas describen esa población filtrada y no
la totalidad del benchmark.

Train y selection usan secuencias disjuntas fijadas antes de mirar etiquetas o
resultados. No se ha probado independencia entre sesiones de adquisición ni se
ha eliminado la exposición histórica de PHASE a TRAIN40. PHASE no participa en
estos ajustes ni proporciona pseudoetiquetas.

## G1_PORTABLE_V1: ruta de ejecución

La variante utiliza atención deformable diferenciable mediante `grid_sample`,
explícitamente seleccionada. No afirma haber construido la extensión CUDA
externa ni ser una reproducción binaria del operador original. Se exige un
caso bilineal analítico, gradcheck CPU FP64 y comparación CPU/GPU de valores y
gradientes; comparar una función con ella misma no satisface la validación.

Se fija `align_corners=False`, padding de ceros y rechazo de entradas no finitas.
El backward CUDA puede ser no determinista; la semilla registra el experimento,
pero no implica identidad bit a bit. El patch upstream conserva cada cambio
de compatibilidad y de supuestos KITTI, incluido el uso de `fy` en el prior
vertical. El prior altura/proyección sigue siendo aproximado y aprendible.

La ejecución comprobada incluye el modelo completo ResNet50, ocho clases,
42.168.788 parámetros, pérdidas, backward, actualización de una cabeza activa y
decode. La configuración del entrenamiento es 640×384, batch 1 y 11 grupos:
se midió también un grupo, pero no aportó una ventaja de tiempo en la muestra
observada ni era necesario para memoria. La configuración no se reduce por
suposición. Los recibos de cada resolución y grupo conservan su propio alcance.

`G1_extension_v1` permanece bloqueado. La admisión portable exige pruebas del
perfil efectivamente ejecutado, y sus resultados de precisión corresponden a
esa variante. El smoke sintético acredita software; no acredita precisión eAP.

## Entrenamiento y evaluación

Cada run escribe su prerregistro antes del primer paso real: tokens, hashes,
configuración, inicialización, semilla, LR, presupuesto y criterio de aceptación.
El overfit fija una reducción de al menos el 50 % de la pérdida media de centro
evaluada en los mismos 16 frames; se miden además AP, errores y recuperación de
objetos. La reducción de pérdida por sí sola no demuestra un detector útil ni
generalización. Los fallos se conservan; cualquier nuevo presupuesto constituye
otro run, con justificación anterior a sus resultados.

El piloto posterior usa dos secuencias train y dos selection, 256 frames por rol.
Se conserva checkpoint, predicciones iniciales/finales, pérdidas, tiempos,
memoria, exclusiones y semillas. El checkpoint se recarga antes de inferencia
final, sin pasar etiquetas al forward. Calibration y el test oficial no se usan
para elegir hiperparámetros ni resultados.

El [perfil de métricas](DEVELOPMENT_METRICS.md) tiene nombre propio: matching por
distancia XY, AP101 con envolvente de precisión y errores TP definidos. No se
presenta como AP oficial. La confianza es el sigmoide de clase, separado de la
varianza de profundidad; no se multiplica por un término que pueda superar uno.

Una ejecución exitosa permitirá afirmar aprendizaje y rendimiento local de este
detector RGB bajo el protocolo publicado. No bastará para afirmar robustez,
SOTA, rendimiento sobre las ocho clases en selección sin soporte, velocidad
validada ni funcionamiento conjunto con PHASE.
