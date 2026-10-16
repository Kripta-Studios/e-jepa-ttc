# Continuación condicionada después de B1

Estado: diseño de la cuarta comparación, **todavía sin registro ni updates**.
No es un candidato seleccionado, un permiso para cambiar los gates ni un
resultado. La ejecución requiere receta completa, fuentes congeladas y QA de
gradientes, invariantes y reanudación antes del primer update científico.

B1 aporta señal emparejada en H frente a su control TTC: mejora el agregado y
MiDc y conserva las protecciones de signo, bandas y población. Sin embargo,
no pasa la banda negativa frente a A5. Ambos hechos están en
[B1_RESULTS.md](B1_RESULTS.md); un fallo de admisión del sistema no cancela la
señal de un mecanismo.

La exactitud del diagnóstico de correspondencia conocida aumenta durante la
fase auxiliar y desciende parcialmente durante el ajuste TTC completo. El
control TTC ajusta mejor el agregado P, pero transfiere peor a H. Esto permite
plantear una pregunta concreta dentro de la familia A: ¿limitar la profundidad
de adaptación posterior conserva mejor la representación adquirida sin impedir
que la lectura se ajuste a TTC?

La matriz primaria no demuestra que sólo pueda funcionar un encoder entrenable:
CORR-FROZEN también mejora la región crítica. CORR entrenable mejora agregado y
MiDc respecto a FROZEN, pero empeora signo y alejamiento. No se presenta esa
comparación como victoria general ni como prueba de necesidad del fine-tuning.
La continuación utiliza la autorización del usuario para estudiar profundidad de
adaptación tras una señal de representación, y la nueva evidencia emparejada B1.

La comparación prevista es una adaptación del último bloque residual frente a
la adaptación completa, desde el **mismo endpoint B1-CORRESPONDENCE-STAGE1-S7**.
Ambos brazos entrenarán la lectura geométrica existente; no se cambiarán volumen,
temperatura, salida, pérdida, masas P, input, ROI ni reloj. Ambos tendrán seed7 y
12.000 updates nuevos, con el mismo muestreo y la misma receta de optimización.
El control completo se entrenará de nuevo dentro de esta comparación porque su
horizonte de 12.000 updates difiere de los 10.000 de B1. No se escogerá el mejor
entre ambos controles después de ver H.

El registro deberá concretar qué parámetros forman el último bloque, las
políticas train/eval y dropout, las tasas por grupo, el estado inicial de Adam y
scheduler, los hashes del padre y de las poblaciones, y la decisión cuantitativa
emparejada. Los dos endpoints se congelarán antes de abrir H de esta comparación.
Se comprobará que los parámetros excluidos no cambian y que el bloque entrenable
recibe gradientes y cambia efectivamente. La ruta debe conservar CUDA Graphs,
caché sensorial exacta y checkpoints completos con su contabilidad.

Será la cuarta comparación adaptativa tras C1, D1 y B1, con un máximo de 24.000
updates científicos. No habilita una quinta comparación, otra familia de réplica,
un sweep de pérdida o semillas, ni búsqueda en V/FCWD. H sigue siendo desarrollo
reutilizado; una mejora posterior no sería validación externa independiente.

El refit y un nuevo export test12 seguirán condicionados a una receta admitida y
congelada. No se ejecutan por el mero hecho de terminar los fits. La familia de
réplica ya utilizada es C1/fusión; no se sustituye retrospectivamente por B1 o A.
Las mediciones de coste pendientes, la entrega reproducible y su extracción
independiente siguen siendo parte del cierre, cualquiera que sea el resultado.
