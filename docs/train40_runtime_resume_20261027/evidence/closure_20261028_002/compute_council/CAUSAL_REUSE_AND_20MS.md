# Reutilización causal y viabilidad de veinte objetos por debajo de 20 ms

Investigación CPU del 28 de octubre de 2026. No se ejecutaron modelos ni GPU, ni se modificaron fuentes del runtime. El único experimento nuevo lee relojes, shapes y anotaciones causales; conserva script y resultados en este directorio. La referencia congelada y el gate permanecen intactos.

## Conclusión

La caché de endpoints event entre actualizaciones no es una vía principal para este replay: la ROI cambia antes de que vuelva a aparecer la misma fase temporal. Incluso identificando proyecciones ROI equivalentes, menos del 1% de los inputs event previamente solicitados es reutilizable. La caché RGB sí elimina una fracción considerable de llamadas, pero no garantiza el plazo en actualizaciones con un frame/ROI nuevo. No hay evidencia actual de una ruta admitida que cumpla veinte objetos en menos de 20 ms.

Esto no constituye una prueba de imposibilidad para otros kernels u otro hardware. Sí descarta prometer el objetivo a partir del solapamiento del historial o de la frecuencia nominal RGB sin medir su identidad real.

## Dependencias exactas

`operational/rgb_port_revision/transfer_inputs.py:17` fija lags event de 300, 200, 100 y 0 ms. Cada observación contiene tres ventanas consecutivas de 100 ms. Para una consulta en t, las seis ventanas únicas terminan en t−500, t−400, …, t. `operational/train40_runtime/inputs.py:261` construye esas seis ventanas y las coloca en doce slots.

Con consultas cada 20 ms, ninguna ventana coincide con la actualización inmediatamente anterior. Existen cinco fases temporales. Frente a t−100 ms sí coinciden cinco de seis ventanas, y tres de cuatro observaciones completas, **solo si la ROI, fuente, representación y contrato de ejecución siguen siendo equivalentes**. Por eso hace falta conservar al menos las cinco fases, no solo el último resultado.

Para RGB, `sensor_triplets` (`operational/rgb_port_revision/transfer_inputs.py:66`) parte del último frame disponible y busca predecesores a 100 ms ±1 ms, dentro de 650 ms. Los relojes reales tienen intervalo mediano aproximado de 50 ms: 100 ms es el espaciado del modelo, no la frecuencia de la cámara. Un frame nuevo puede alternar la fase par/impar de toda la cadena. La reutilización por tripleta puede recuperar observaciones al volver a una fase, pero una ROI nueva invalida el recorte de todo el historial.

`MultiObjectTrain40Service` reaplica la ROI actual a frames RAW y eventos históricos (`multiobject.py:267`, `:498`). Ese contrato impide reutilizar features calculadas con la antigua ROI. GroupNorm en los encoders (`causal_scale_ttc.py:548` y siguientes) depende del recorte; calcular una imagen completa y luego ROIAlign no reproduce esas normalizaciones.

Además, reutilizar las features de una observación en otro slot o reducir el batch exige admisión numérica: `producers.py:344` advierte de cambios de redondeo al reducir el batch. Ya existen fallos E cuando se reduce de doce slots a seis. La igualdad del input es condición necesaria para la caché; no demuestra equivalencia de un nuevo esquema de batching.

## Experimento de metadatos reproducible

Comando:

```powershell
& '../e-jepa-ttc/.venv/Scripts/python.exe' artifacts/train40_runtime_resume_20261027/compute_council/causal_cache_metadata_probe.py
```

El script exige salida nueva y escribe `causal_cache_metadata_v1.json`. Para repetir, copiar el script y cambiar explícitamente el nombre del resultado; conservar el anterior. Usa las mismas 32 secuencias y hasta cien consultas de 20 ms por secuencia que el probe RGB anterior. Programados 3.191; aceptados 3.129; excluidos 62 (53 ambiguos/no primarios y nueve sin anotación). Un hueco reinicia las cachés. Solo inserta inputs ya solicitados, nunca queries o ROI futuras. Retención máxima de claves desde su último uso: 650 ms.

Todas las cifras siguientes proceden de `counts` y `fractions` del JSON, con hashes del script y manifiesto. No se leyeron píxeles, eventos RAW, pesos ni predicciones. Un objeto primario por secuencia; ninguna conclusión de hit-rate real para veinte tracks.

| Unidad candidata de caché | Clave ROI | Reutilizables / requeridos | Fracción |
|---|---|---:|---:|
| Ventana event de 100 ms | Cuatro floats exactos | 115 / 18.774 | 0,6125% |
| Ventana event de 100 ms | Proyección efectiva exacta | 155 / 18.774 | 0,8256% |
| Observación event de tres ventanas | Cuatro floats exactos | 69 / 12.516 | 0,5513% |
| Observación event de tres ventanas | Proyección efectiva exacta | 93 / 12.516 | 0,7430% |
| Tripleta RGB | Cuatro floats exactos | 9.286 / 15.487 | 59,9600% |
| Tripleta RGB | Crop efectivo exacto | 9.298 / 15.487 | 60,0375% |

En 2.916 pares elegibles separados 100 ms, la ROI event exacta permanece igual en 23, y su proyección efectiva en 31. La ROI RGB exacta permanece igual en 36; su crop efectivo en 43. La caché de llamada RGB completa tiene los mismos 1.719 hits usando floats o crop efectivo: 54,9377% de las consultas aceptadas, 53,8703% de las programadas. La comparación entre fracción de observaciones y de consultas tiene denominadores distintos; no equivale directamente a ahorro de tiempo.

La proyección event efectiva se identifica por las dos tablas de límites de píxeles de `integral.py:26`. El crop RGB efectivo usa clipping y conversión a entero de `rgb_port/data.py:132`. Estas equivalencias conservan los píxeles del contrato actual; no consisten en aproximar la ROI. Aun así, no aportan una mejora relevante en esta población. Mantener metadatos solicitados actuales para auditoría y contexto aunque una clave efectiva se reutilice.

## Qué se puede adelantar sin usar futuro

1. **Ingestión compartida independiente de ROI:** conteos y momentos temporales de cada paquete recibido pueden incorporarse inmediatamente. El prefix actual ya conserva esa información antes de conocer la ROI (`temporal_prefix.py:439`). Para adelantar trabajo dentro del último paquete de 20 ms, el sistema real tendría que entregar subpaquetes, por ejemplo cada 5 ms. Con el contrato actual que entrega un único paquete completo, no se puede fingir esa disponibilidad anterior.
2. **Frames RAW:** decodificar, transferir y fijar identidad al llegar un frame es causal. Los buffers deben tener propiedad y vida útil estables. Su coste sigue formando parte de la carga sostenida y de la latencia desde disponibilidad de esa observación. No incluir el acceso a futuros frames de un HDF5 como prefetch causal.
3. **Historial dependiente de ROI:** cuando se conoce la ROI actual pueden prepararse las cinco ventanas event antiguas y los productores de las tres observaciones antiguas, cuyos eventos ya terminaron. Para el query siguiente, una predicción de que la ROI no cambiará es especulativa: servirla solo si la clave real coincide; contabilizar trabajo descartado y el camino de fallo. No usar anticipadamente la anotación que el replay aún no habría entregado.
4. **Contexto final:** aunque los productores RGB hagan hit, sus edades, disponibilidad, selección y fusión se reconstruyen en el query actual (`multiobject.py:93`, `batched_context.py:180`). No cachear TTC ni extrapolarlo como equivalente.

Adelantar trabajo puede acortar el camino después de la última observación, pero no elimina demanda de cómputo. Para 50 Hz sostenidos, la GPU debe absorber en promedio menos de 20 ms de trabajo por actualización, incluido trabajo adelantado o descartado, y el camino final debe satisfacer también el plazo. Un pipeline que trabaja 80 ms antes y 15 ms después por consulta continúa acumulando cola en una sola GPU si llegan consultas cada 20 ms.

## Límites legítimos y siguiente experimento

El prefix más reciente de este frente ya mide 30,62268 ms de media para ingestión y representación event de veinte ROI, con cinco repeticiones (`../prefix/gpu_query_graph_v1/summary.json`, resultado TemporalPrefixEventHistory/20). Es un cuello de botella observado del camino actual, no un límite físico de cualquier implementación. No sumar sus percentiles con los de productores de otro run para fabricar un p95 total.

En el peor caso admitido —veinte ROI nuevas— el ahorro garantizado de caché es cero. El hit-rate conjunto de veinte objetos no puede inferirse elevando el de uno a la potencia veinte: las llegadas de frames están correlacionadas y no se conoce la dependencia entre sus ROI. El sondeo actual tampoco ofrece una distribución p99 de costes.

**Siguiente experimento prioritario:** una vez admitido el transporte compilado, medir el pipeline completo a reloj de 20 ms con veinte ROI diferentes y un perfil explícito de cambios de ROI. Separar tres poblaciones: replay causal de ROI anotada primaria más carga artificial declarada; cambio efectivo de las veinte ROI en cada update como ensayo de fallos de caché; ROI fija como techo favorable. Conservar cobertura, gates de input/TTC, misses, trabajo en background, disponibilidad real y crecimiento de cola. Esto da el presupuesto residual real, sin atribuir al cache mejoras que solo aparecen al repetir exactamente la misma consulta.

Si el objetivo es reducir directamente el cuello de botella de 30 ms de representación, el siguiente desarrollo de alto impacto es medir una consulta prefix selectiva por ventana y adelantar las cinco ventanas antiguas cuando la ROI ya está disponible, junto con ingestión de subpaquetes reales. Debe empezar con un trazado de dependencias y tiempos: disponibilidad de paquete/frame/ROI, comienzo y final de cada ventana, bytes/cómputo descartados y camino crítico. No basta con reportar únicamente el tramo de la última ventana. Si la carga total por periodo sigue excediendo 20 ms, la anticipación no demuestra 50 Hz.

El resultado negativo de la caché event hace preferible optimizar el transporte completo ya medido y la representación prefix antes que implementar una caché compleja de cinco fases. La caché RGB de llamadas enteras es útil como optimización oportunista; sus misses siguen siendo parte del requisito.
