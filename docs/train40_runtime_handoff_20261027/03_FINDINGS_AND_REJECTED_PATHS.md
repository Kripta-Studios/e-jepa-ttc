# Hallazgos, límites y rutas rechazadas

## Perfil original

En la consulta densa 553, cProfile atribuyó aproximadamente 1,69 s a eventos: lectura/crop RAW, proyección y doce encodings repetidos. Los productores costaban unos 0,33 s. La consulta 200 era mucho más ligera. Los perfiles completos están publicados; no representan latencia CUDA aislada.

El caso 553 contiene aproximadamente 9,3 millones de eventos dentro de la ROI en 600 ms y 15,5 millones RAW en la ventana del probe. Evitar I/O repetido no elimina el coste de reproyectarlos. Las LUT y deduplicación CPU reducen trabajo, pero no bastan para 20 ms en ese caso.

## Mejoras verificadas parcialmente

- H4 y seis encodings únicos, lectores/cachés persistentes y LUT: pruebas de igualdad contra las implementaciones originales.
- Contexto residente: seis casos con pesos sellados dieron igualdad exacta; tramo aislado cercano a 1 ms. No incluye productores.
- Grafos completos más reutilización: reducen lanzamiento Python, viajes de fase y CNN repetidas. Cambios de batch introducen pequeñas diferencias; admisión final solo probada en tres consultas.
- Ramas E/R concurrentes: equality serial/paralelo en los tres pilotos y guards/lifetime comprobados. No equivalen a veinte objetos.
- Crop paralelo: en el caso denso la mediana RAW-cache pasó de unos 21,4 a 8,8 ms con ocho workers. El intento de copiar ROI a memoria contigua antes de PIL fue más lento y se rechazó.
- Momentos GPU: tensores exactos para 0/200/553, incluyendo ingestión nueva y ROI actual. Los veinte ROI de carga siguen costando unos 75 ms sin modelos.

Consultar [RESULTS_GENERATED](RESULTS_GENERATED.md) para cifras ligadas a recibos y alcances. No extrapolar un porcentaje de mejora entre fronteras temporales diferentes.

## Resultados negativos que hay que conservar

1. **JIT con optimización:** después de la primera admisión exacta, el executor optimizó y aparecieron diferencias pequeñas. Se desactivó la optimización durante traza/warmup/captura; `diff_probe.json` conserva el fallo.
2. **Primer piloto completo:** `pilot_v1` fue invalidado al cambiar `producers.py` durante el ensayo. La paridad pasó, pero su trazabilidad no. No promover esos tiempos.
3. **Dos objetos agrupados:** `grouped_multiobject_floor.json` contiene `serial_parallel_exact=false` para dos objetos. La diferencia no se cuantificó antes de pausar. Sigue pendiente aunque las otras cantidades coincidan serial/paralelo.
4. **Deduplicación de pares de transporte:** dos pruebas CPU del modelo completo difieren bitwise en `pair_tokens`. Las assertions exactas permanecen como `xfail(strict=True)`. No se relajó el gate de TTC. La opción sigue sin admitir y no está conectada a productores.
5. **Scatter de todo RAW en GPU:** fue exacto en tres casos, pero una representación densa podía consumir casi todo el presupuesto de 20 ms antes de modelos. Además el primer prototype precalculaba índices dependientes del ancla; no era un update completo.
6. **Compilador:** dos fallos de toolchain Windows se conservaron antes de una ejecución satisfactoria; no se instalaron dependencias en el entorno científico original.

## Transporte: pruebas y propuestas

La correlación inversa cumple `C_reverse[d,p] = C_forward[-d,p+d]`. El volumen directo permite reconstruir el inverso con desplazamientos y padding, preservando la reducción original por canal. Dieciséis casos CPU dieron igualdad exacta en campos y probabilidades.

Se pueden reutilizar denominadores de medias ponderadas, traslaciones ya calculadas y muestras inversas compartidas entre summary global/foreground. `transport.py` ofrece una vista privada del modelo sin monkeypatch global. Sus opciones por defecto pasaron pruebas CPU; **no tienen admisión CUDA ni sobre 554 consultas**.

La deduplicación de pares adyacentes reduce otra redundancia, pero cambia batch y falló igualdad bitwise en dos pruebas. Antes de conectarla, cuantificar tanto tensores como TTC, conservar las diferencias y usar el gate ya fijado. No anunciar una reducción teórica de operaciones como speedup medido.

## Límite de cómputo observado

El recuento por hooks del encoder con reutilización de endpoints dio aproximadamente 7,02 GMAC por objeto. Veinte objetos requieren aproximadamente 280,8 GFLOP solo en convoluciones (dos FLOP/MAC). Caber en 20 ms exigiría alrededor de 14 TFLOP/s sostenidos en esas convoluciones, además de todo lo restante. Es un recuento, no una medición de rendimiento máximo del hardware.

La prueba agrupada mejoró los productores preparados de veinte objetos desde unos 265 ms a unos 180 ms. Sigue lejos del objetivo. Los próximos probes razonables son fusión de transporte/encoder, layout channels-last y algoritmos cuDNN, manteniendo pesos y precisión/gate. No asumir que más CUDA Graphs solucionan el cómputo restante.

## Otra representación pendiente

El ring actual consulta 120 buckets por ROI. Una alternativa todavía **no implementada** almacena prefijos temporales acumulados de las integrales espaciales; cada cuarto de 25 ms sería diferencia de dos prefijos. Se consultarían 25 fronteras en vez de 120 buckets. Requiere revisar overflow de momentos en sesiones largas, memoria int64 y paridad; no hay tiempos medidos para esta idea.
