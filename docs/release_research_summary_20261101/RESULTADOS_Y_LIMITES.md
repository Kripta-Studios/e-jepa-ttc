# Resultados conseguidos y límites de la investigación

Fecha de cierre experimental: 1 de noviembre de 2026. El
[documento de trabajo realizado](TRABAJO_REALIZADO.md) detalla implementación,
ejecuciones, correcciones y recuperación. Este documento interpreta los
resultados de los informes sellados; las cifras se transcriben de esos
artefactos, que conservan las tablas regenerables.

**Hay una mejora de desarrollo de la corrección de cadencia en Slider. No se
ha demostrado una mejora oficial del release ni una ganancia generalizable.**
Los restantes controles ejecutados no justifican abrir los entrenamientos
condicionales definidos en el dossier. Eso no demuestra que cualquier variante
futura de observabilidad, reloj, sensor, visibilidad o JEPA sea inútil.

## 1. Cadencia: señal favorable localizada

La evaluación completa comprende 2608 consultas. La tabla utiliza solamente
pares disponibles en ambos brazos y mantiene la ausencia original de Slider.

| Dominio | Pares / población | MAE original (s) | MAE canónico (s) | Cambio relativo |
|---|---:|---:|---:|---:|
| Carretera | 2531 / 2531 | 0,810620018 | 0,810489612 | −0,0161 % |
| Slider | 76 / 77 | 1,178155898 | 1,120808950 | −4,8675 % |

F0 supera el criterio de desarrollo registrado: ausencia de regresión en MAE
y error relativo en ambos dominios, al menos una mejora y conservación de
cobertura y signo. El intervalo del cambio de MAE en carretera es
**[−0,000504700; +0,000093150] s**, con bootstrap por adquisición. Incluye cero:
el pequeño cambio medio no demuestra una mejora de carretera.

Slider procede de una sola adquisición ya utilizada en el screen. Las 2459
consultas nuevas de la ampliación son de carretera; no convierten la mejora
de Slider en una replicación independiente.

La localización por modalidad mostró que corregir la cadencia RGB explica la
intervención de carretera del screen. En Slider contribuyen ambas modalidades.
El caso nominal de eventos conserva identidad exacta. Este resultado permite
conservar un candidato concreto y acotado, sin reconstruir toda la representación.

El router también pasó el criterio puntual de la ampliación, pero había
fallado su screen anterior y permanece diagnóstico. El candidato considerado
en esta campaña es F0. Fuente:
[evaluación de población](../release_research_implementation_20261031/campaign_results_v1/POPULATION.md).

## 2. Observabilidad: los indicadores actuales no predicen mejor cuándo corregir

El ensayo con adquisición excluida comparó la predicción del beneficio de
corregir RGB. Menor MSE es mejor:

| Control | MSE macro |
|---|---:|
| Constante | 0,457493 |
| Soporte | 0,492044 |
| Físico | 0,919267 |
| Permutado | 0,756780 |

El brazo físico empeora frente a soporte en aproximadamente 0,427224, con
CI95 [0,080551; 1,010417]. No aporta la señal incremental exigida para admitir
una cabeza de observabilidad mayor.

La auditoría encontró q disponible en 63 casos; ocho tenían cero grados de
libertad, 26 signo incorrecto y diez quedaban fuera del dominio canónico.
La lectura robusta redujo mucho la cobertura: ocho lecturas de q/TTC,
cuatro de ellas con signo incorrecto. Su MAE con fallback fue 0,779543 s,
frente a 0,766025 s de R0 en esa cohorte.

Se consiguió localizar problemas de cobertura, identificación y sesgo. No se
consiguió un gate físico útil. Fuentes:
[continuación RAW](../release_research_implementation_20261031/continued_results_v1/RESULTS.md)
y [auditorías posteriores](../release_research_implementation_20261031/followup_results_v2/RESULTS.md).

## 3. Reloj causal: mecanismo correcto, mejora real no demostrada

Después de corregir PAIR, el MAE de `EVENT_CONTEXT` fue aproximadamente
1,941890 s. EWMA obtuvo 1,952822 s; el observador, 1,981365 s; persistencia
y transporte, 3,104232 s. Estas dos últimas rutas coinciden cuando el último
ancla ya está en el instante de consulta. Los intervalos de los contrastes
pertinentes no acreditan una ventaja de los filtros sobre el contexto.

El banco de mecanismos sí verificó transporte exacto a velocidad constante e
invariancia causal. Mostró también que reducir ruido mediante un observador
puede introducir retraso al frenar. Una ecuación correcta no basta para que
la medida que la alimenta sea mejor que el contexto aprendido.

No se admite D4 a partir de estas pruebas. Fuente:
[reloj con PAIR original y canónico](../release_research_implementation_20261031/followup_results_v2/clock_cadence/RESULTS.md).

## 4. Sensor: explicar eventos no equivale a recuperar TTC

El banco S0 mostró ambigüedades: distintas combinaciones de movimiento y
fotometría pueden producir los mismos eventos dentro de su precisión temporal.
Con una referencia conocida se recupera información en ese banco limitado;
eso no supone disponer de calibración física real.

En S1, sobre 72 consultas de carretera y 24 adquisiciones, los modelos
sensoriales basados en textura y física no superaron el control de actividad.
En el control posterior con máscara común, el cambio RGB observado obtuvo
MSE 96,657290 frente a 145,648563 del modelo físico. La diferencia favorece
RGB observado: CI95 [−83,9315; −0,2561]. Sin embargo, frente al control de
actividad, el intervalo [−88,7005; +27,1035] incluye cero.

Esto encuentra información relevante para explicar la formación de eventos,
pero no demuestra que esa información se traduzca en una expansión útil.
Las lecturas radial, con traslación y afín produjeron TTC en las 72 consultas,
pero introdujeron errores grandes. Al aplicar fallback en la cohorte completa:

| Lectura | MAE (s) |
|---|---:|
| R0 | 0,766025 |
| Radial | 13,2410 |
| Con traslación | 12,0502 |
| Afín | 9,1685 |
| Permutación espacial | 45,6296 |

La lectura afín supera a la permutada y a las otras lecturas físicas, pero
empeora ampliamente respecto a R0. Entrenar el productor para imitar esa q
podría imponer un objetivo sesgado. No se admiten S2/S3 con este proxy.
Fuentes: [referencia RGB](../release_research_implementation_20261031/followup_results_v2/sensor_reference_control/RESULTS.md)
y [recuperación de q](../release_research_implementation_20261031/followup_results_v2/photometric_q/RESULTS.md).

## 5. Superficies y visibilidad: medición disponible insuficiente

La medición con fallback obtuvo MAE macro 3,103983 s frente a 0,766025 s de
R0. El intervalo del empeoramiento fue [0,934231; 4,198245] s. Hubo medición
en 133 de 149 consultas y fallback en 16.

No se detectaron cambios de crop en las 146 historias auditables; 53 tenían
redimensionado anisótropo constante. Esto acota una hipótesis concreta sobre
recortes, sin descartar oclusión, fondo o paralaje. El resultado no admite
V2/V3 a partir de esta medición. Fuente:
[ensayo de superficies](../release_research_implementation_20261031/continued_results_v1/RESULTS.md).

## 6. Comparación nativa: conservación local de cobertura, fallo histórico

Se compararon original y candidato en 5805 consultas físicas del mismo equipo,
tras exigir repetibilidad exacta de la ruta original local.

| Comprobación local | Resultado |
|---|---:|
| Consultas emparejadas | 5805 / 5805 |
| Nuevas ausencias | 0 |
| Cambios de signo | 0 |
| Predicciones idénticas | 86 |
| Diferencia absoluta mediana | 0,000014305 s |
| Diferencia absoluta p95 | 0,000076771 s |
| Diferencia absoluta p99 | 0,000121040 s |
| Diferencia absoluta máxima | 0,006496429 s |

El contexto de eventos permanece idéntico. Son medidas de sensibilidad de
predicción, no errores respecto a ground truth.

Por separado, el original local falló la paridad intermedia histórica en las
5805 consultas y la tolerancia de TTC en 504. La diferencia histórica de TTC
tuvo mediana 0,000185490 s, p95 0,001094055 s y máximo 0,496791840 s.
El máximo pertenece al estrato de predicciones de magnitud superior a 16 s.
Las cuatro políticas numéricas ensayadas en el perfil inicial no recuperaron
la paridad histórica.

Por tanto, no está demostrada la conservación de Overall/Crucial/Negative ni
la no inferioridad del score oficial. No se cambió la tolerancia para admitir
el candidato. Fuentes: [comparación local](../release_research_implementation_20261031/campaign_results_v1/RESULTS.md)
y [deriva histórica completa](../release_research_implementation_20261031/native_parity_distribution_v1/RESULTS.md).

## 7. Latencia: presupuesto descriptivo cumplido en ambas fronteras

Ambos estudios utilizaron 120 consultas y la GPU RTX 5070 Ti Laptop. Las cifras
son tiempos de pared de la pasada principal, sin hooks de instrumentación.

| Frontera | Brazo | p50 (ms) | p95 (ms) | p99 (ms) |
|---|---|---:|---:|---:|
| NPZ preparado + RGB | Original | 329,344 | 1603,654 | 1737,175 |
| NPZ preparado + RGB | Canónico | 346,744 | 1658,066 | 1791,578 |
| RAW + RGB | Original | 532,024 | 963,353 | 1014,155 |
| RAW + RGB | Canónico | 534,662 | 1008,124 | 1208,804 |

El aumento de p95 fue **3,39 %** desde NPZ y **4,65 %** desde RAW, dentro del
presupuesto descriptivo de +10 %. La reconstrucción RAW fue exacta en 120/120
observaciones y todas las salidas cronometradas coincidieron con el replay.
El pico CUDA asignado fue aproximadamente 66,03 MiB; la RAM máxima del ensayo
RAW fue 2 433 994 752 bytes.

Las dos fronteras se midieron en ejecuciones separadas. No se pueden restar
sus percentiles para calcular coste de voxelización ni afirmar que RAW es
intrínsecamente más rápido. El caché del sistema operativo puede estar caliente,
se usan prefijos de secuencia y las cajas ya vienen dadas. No se midieron
detector, captura física o transporte. Tampoco se ha demostrado servicio
sostenido a 50 Hz. Fuente:
[suplemento RAW y mediciones](../release_research_implementation_20261031/raw_latency_results_v1/RESULTS.md).

## 8. Qué está terminado y qué sigue sin demostrar

| Estado | Alcance |
|---|---|
| Ejecutado y cerrado | Cadencia 2608; controles de observabilidad, reloj, sensor y superficies; comparación local 5805; distribución de deriva histórica; latencias NPZ y RAW |
| Conseguido | Candidato de cadencia con mejora de desarrollo localizada; cobertura y signo conservados en la comparación local; costes medidos; fallos físicos y numéricos documentados |
| Implementado como auxiliar | Componentes físicos, predictor temporal y mecanismos de admisión, con tests; no son nuevos modelos de producción |
| No entrenado | O2, D4, S2/S3, V2/V3 y JEPA condicional; los controles previos no cumplen sus condiciones de admisión |
| No demostrado | Generalización de la mejora de Slider, confirmación independiente, conservación del score nativo y mejora oficial del release |
| Conservado | Release oficial, versiones sustituidas, fallos, fuentes ejecutadas y trazabilidad de los resultados |

No quedan los ensayos admitidos de esta campaña esperando a ser lanzados. La
validación local pendiente se completó con los datos que el usuario autorizó.
El dossier completo contiene propuestas condicionales más amplias que esta
campaña: no se declara que todas hayan sido entrenadas ni que la investigación
sobre esos mecanismos esté agotada.

Para proponer un nuevo release habría que resolver o explicar de forma
verificable la réplica histórica y evaluar precisión bajo un protocolo válido
de conservación; para afirmar generalización haría falta confirmación que no
dependa de la adquisición ya utilizada. Repartir de nuevo los mismos datos no
crea esa independencia.

Con los datos actuales puede formularse otra hipótesis acotada sobre sesgo,
cobertura o identificación, fijando controles antes de evaluarla. Reabrir un
brazo aprendido requiere evidencia incremental respecto a los controles que
han fallado aquí, su genealogía de entrenamiento y la evaluación multisemilla
correspondiente. La señal de cadencia se conserva como candidata de desarrollo;
el release oficial permanece sin cambios.
