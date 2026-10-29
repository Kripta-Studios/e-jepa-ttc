# Propuesta `train40_repeated_workload_stress_v1`

Estado: diseño prospectivo. No implementado, no medido y sin admisión de
50 Hz. El protocolo natural existente permanece intacto.

El inventario local no tiene una secuencia de 60 segundos. Esta propuesta
permite ensayar 60 segundos de carga repetida y evaluar estabilidad del
servicio, preservando la distinción entre duración de ejecución y duración
de observaciones distintas.

## Población y relojes

Usar doce episodios del manifiesto causal `CPLA-low` de 250 consultas:
3000 updates y 60 segundos de períodos ofertados. Cada episodio conserva los
timestamps de sensor originales `16750664..21730664`; no se modifica el tiempo
que recibe el modelo ni se simula continuidad entre la última observación de
un episodio y la primera del siguiente.

Cada resultado debe identificarse mediante `(episode_id, query_id, object_id)`.
El reloj host usa `scheduled_ns = first_ns + update_index * 20000000`, sin
retrasar la siguiente llegada para esperar a la inferencia. El manifiesto
declara 250 consultas de sensor distintas, 4,98 segundos entre sus anchors,
5 segundos de períodos únicos y doce repeticiones.

## Frontera entre episodios

La primera inicialización global puede cargar modelos, primar shapes
sintéticas y establecer la historia anterior a la primera llegada, como el
runner actual. Debe quedar registrada fuera del reloj.

En cada frontera posterior, la primera llegada del nuevo episodio incorpora
un marcador explícito de reinicio. Desde su deadline hasta todos sus TTC
deben contabilizarse:

1. invalidación de historia, identidad de fuente y cachés temporales;
2. adquisición del prefijo RAW de 600 ms hasta `primer_anchor−20000 µs`;
3. adquisición del paquete nuevo y de los frames RGB causalmente necesarios;
4. reset, transferencias e inicialización de la representación de historia;
5. preparación de la consulta, modelos, contexto y copias finales a host.

No insertar una pausa en la grilla entre episodios. No precalcular crops,
representaciones, features o TTC del episodio siguiente. No conservar features
o predicciones del anterior para responder al siguiente. Los mismos modelos,
pesos, normalizadores y grafos de shapes pueden persistir durante los 60 s.
Las cachés del sistema operativo pueden favorecer lecturas repetidas y deben
declararse como límite del ensayo; no se afirma medir almacenamiento frío.

La implementación deberá utilizar el reset explícito de historia con IDs de
época y limpiar todos los estados que dependen de timestamps. No basta con
aceptar un rollback en la API actual. Los payloads de inicio tienen un tamaño
mayor y deben respetar presupuestos de bytes explícitos, incluyendo historia,
frames, cola y lifetime de objetos retenidos.

El coste de reinicialización probablemente hará fallar 20 ms. Ese resultado
es informativo y debe conservarse. Excluir las fronteras permitiría describir
episodios rápidos, pero no admitir una ejecución sostenida a 50 Hz.

## Gate separado

Un resultado positivo requiere los 3000 updates programados, todos los objetos
y tres TTC host por objeto, cero errores o descartes, todas las respuestas
deadline→TTC estrictamente por debajo de 20 ms y ningún trabajo anterior
pendiente a una llegada. También requiere preservación de pesos,
normalizadores y fuentes, y paridad independiente de todas las salidas con
`atol=rtol=1e-5`.

El oracle puede preparar las referencias después del tramo cronometrado.
Aunque una referencia por consulta sea idéntica entre episodios, la comparación
debe cubrir cada resultado de cada episodio para detectar estado residual y
fallos intermitentes. La validación de entradas debe cubrir al menos el primer
update de cada episodio, los cambios de ROI y todas las formas ejecutadas.

Publicar distribuciones globales y vistas de transiciones/interior de episodio.
La distribución global decide el gate; ninguna vista permite retirar outliers.
Mantener todos los errores, IDs ausentes, descartes, tiempos de fuente, bytes,
uso de memoria y preservación. La frecuencia se verifica con las llegadas y
terminaciones reales, nunca con el inverso de una media.

## Afirmación permitida

Campos prospectivos mínimos:

```json
{
  "schema": "train40_repeated_workload_stress_v1",
  "repeated_workload": true,
  "episodes": 12,
  "unique_sensor_queries": 250,
  "unique_sensor_anchor_span_s": 4.98,
  "unique_offered_query_periods_s": 5.0,
  "offered_updates": 3000,
  "offered_rate_hz": 50.0,
  "offered_duration_s": 60.0,
  "episode_transitions_inside_deadline_clock": true,
  "natural_contiguous_60s_demonstrated": false,
  "physical_goal_demonstrated": false
}
```

Estos valores describen el diseño; un runner futuro deberá calcularlos desde
su manifiesto y recibos, sin rellenar resultados por adelantado. Un pass
demostraría capacidad sostenida para esta carga de replay repetida. No
demostraría 60 segundos de observaciones nuevas ni validez con sensores reales.
Tampoco cambia automáticamente el gate natural de uno, dos o veinte objetos.
