# Control de referencia tras S1 exploratorio

El proxy radial ligado a q de R0 no supera actividad ni textura; su diferencia
frente a q permutado tampoco es concluyente. Antes de atribuirlo al estado de
movimiento, comprobar si la referencia RGB aproxima siquiera los cambios del
sensor. No se eligen parámetros usando TTC ni se abre un entrenamiento mayor.

Añadir un único control observado: diferencia de log-luminancia entre la misma
referencia RGB anterior al inicio y la última imagen RGB disponible antes de
la consulta, dividida por su propio intervalo real. Conservar la proyección
geométrica aproximada y las celdas 16x16 con 5x5 puntos. Exigir el mismo 80% de
muestras válidas en ambos extremos, con luminancia entre 5 y 250. Reajustar los
cuatro controles anteriores sobre la misma intersección de celdas. Quedan cinco
sondas de dos parámetros, 24 folds de adquisición excluida, sin búsqueda.

Reutilizar los recuentos RAW ya sellados. Leer únicamente las dos imágenes RGB;
verificar el hash de la referencia original. Registrar el desfase de los
extremos RGB respecto a los intervalos de eventos y el área retenida. No usar
la siguiente imagen posterior a la consulta para mejorar alineación temporal.
La variación RGB observada es una sonda reconstructiva, no una medición de TTC
ni una salida física de q. Diferencias de respuesta, paralaje y desfase temporal
siguen siendo causas alternativas. Una ventaja del control observado sobre el
radial localiza una carencia del proxy de movimiento; su ausencia indica que
no se ha validado la referencia fotométrica. Ninguna conclusión demuestra por
sí sola mejora TTC o autoriza S2.

CPU a un hilo y sin CUDA visible, con la misma excepción de recursos perfilada
en S1. La única nueva lectura es RGB; no se duplica la extracción de eventos.
