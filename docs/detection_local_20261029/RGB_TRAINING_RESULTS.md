# PHASE-TTC con detección 3D: resultados locales

Fecha: 2026-10-29. Rama: `dev/eap-detection-local-20261029`.

Documentación ampliada: [implementación](IMPLEMENTATION_REPORT.md),
[experimentos y hallazgos](EXPERIMENTS_AND_FINDINGS.md),
[reproducción y continuación](REPRODUCTION_AND_HANDOFF.md).
El [índice de documentación](README.md) reúne los informes por componente.

La viabilidad de ingeniería ya está demostrada: el detector completo aprende
sobre imágenes eAP reales en la GPU local, guarda y recupera sus pesos y produce
cajas 3D sin recibir ROIs ni etiquetas como entrada. Existe además un replay
causal que pasa ROIs detectadas a los pesos finales de PHASE y conserva TTC
firmado, fase y límites de fase. Esto permite desarrollar el sistema propuesto;
todavía no acredita robustez de la cadena ni un resultado oficial en eAP.

**El piloto terminó y muestra señal útil, pero la precisión sigue siendo baja.**
Completó 512 updates y 2.048 exposiciones en 18,28 minutos de entrenamiento
(21,93 minutos desde preregistración hasta el cierre). La AP101 de desarrollo
en selección fue 0,020641 sobre cinco clases con GT admitido; el prior constante
alcanzó 0,004126 y la permutación prefijada 0,003728. Esta evidencia permite
continuar el desarrollo; no justifica presentar el nuevo detector como robusto.

## Evidencia que cambia el estado anterior

- **Geometría RGB/3D admitida para este protocolo:** 17.611 correspondencias
  original/HF exactas en una secuencia train; proyección contrastada con el
  visualizador original y tres imágenes reales. La aproximación pinhole conserva
  el límite de la RGB remapeada al plano de eventos. Véase
  [GEOMETRY_RESOLUTION.md](GEOMETRY_RESOLUTION.md).
- **Detector completo en GPU:** MonoDGP con ocho logits de clase y backend
  deformable PyTorch, probado con forward, pérdidas, backward, actualización,
  decode y recarga estricta. El operador CUDA externo no se compiló; no se
  presenta este resultado como equivalencia con aquella implementación.
- **Sobreajuste real:** el primer presupuesto terminó sin superar su criterio.
  Se conservó ese fallo y se declaró una continuación con el mismo umbral
  original; después de 800 updates acumulados, la razón de pérdida de centro fue
  0,3452, menor que el máximo fijado de 0,5. Las predicciones finales se
  reprodujeron exactamente en un proceso de inferencia separado sin etiquetas.
- **Velocidad aprendida:** tres cabezas por decoder reciben supervisión de los
  dos componentes publicados, con máscara de validez y pérdida explícita. No se
  inventan ceros ni se usa esta velocidad para calcular TTC.

Las [tablas regenerables](TRAINING_TABLES.md) conservan los resultados completos
de cada run terminado. El [perfil y las condiciones](PORTABLE_DETECTOR.md)
explican la selección del backend, inicialización y batch. Los gates amplios
del handoff original permanecen como evidencia histórica: las admisiones
`G0_RGB_BOX_V1` y `G1_PORTABLE_V1` tienen un alcance más concreto.

## Piloto RGB y velocidad

Se fijaron antes de observar resultados 256 frames de dos secuencias train y
256 de otras dos secuencias de selección, semilla 7 y 512 updates máximos con
batch 4. El backbone utiliza ImageNet1K V1; las cabezas de detección se
inicializan al azar. El límite de entrenamiento es 1.200 segundos.

Las poblaciones se seleccionaron por hashes y tiempos, sin elegir imágenes
según sus etiquetas. Tras los filtros explícitos de profundidad y visibilidad,
train conserva 829 objetos de siete clases y selección 938 de cinco clases.
`other_vehicle` no tiene positivos train admitidos y `bicycle` solo tiene uno.
Las clases ausentes no quedan ocultas: el reporte incluye macro sobre soporte
y macro de ocho clases con cero en ausentes.

Las cifras finales se incorporan desde los recibos del run. Los
[controles prefijados](IMAGE_SIGNAL_CONTROLS.md) comparan el detector con cajas
constantes ajustadas solo en train y con sus predicciones permutadas entre
imágenes de selección. Detectar una señal útil en este piloto no sustituye una
evaluación amplia por secuencias ni tres semillas.

La cabeza de velocidad obtiene error vectorial medio 0,594901 m/s, frente a
1,381150 para cero y 0,759128 para la media train, sobre los mismos 401 matches
de tres clases. El segundo componente empeora frente a ambos controles. Es
una medida condicional a las detecciones emparejadas, no una evaluación de todos
los objetos ni de velocidad relativa válida para TTC.

La [inferencia independiente](INFERENCE_VISUALS.md) procesó 279 frames sin abrir
etiquetas. Las predicciones de los 16 train y los 256 de selección coinciden
exactamente con las del runner, también en velocidad. En selección se midieron
47,89 ms de forward y 82,99 ms por frame completo, medianas sin inicialización
ni PHASE. Las cuatro visualizaciones elegidas por posición fija muestran
duplicados y errores de clase: constituyen evidencia visual de esos límites.

## Integración con PHASE

| Salida solicitada | Procedencia ejecutada | Alcance |
|---|---|---|
| Clase y confianza | Logits del detector y sigmoid | Ocho salidas; soporte positivo incompleto en este piloto |
| Centro 3D, dimensiones y yaw | Cabezas MonoDGP y conversión eAP explícita | Geometría admitida para el protocolo RGB, scorer oficial pendiente |
| Velocidad XY | Cabeza aprendida adicional | Componentes nativos publicados; semántica física completa pendiente |
| TTC firmado | Pesos finales existentes de PHASE | Inferencia sobre ROIs detectadas; sin nueva medida de error TTC |
| Fase y límites q10/q90 | Salidas originales del modelo PHASE | No se reconstruyen desde TTC ni implican cobertura calibrada |

El [puente ejecutado](DETECTOR_PHASE_BRIDGE.md) usa detección, asociación causal,
historia de ROIs, eventos y RGB reales. Mantiene separados score de detección,
ambigüedad de asociación y salidas de PHASE; puede abstenerse si la historia no
es suficiente. No consulta IDs GarlTTC ni etiquetas TTC para producir la salida.

La [salida integrada del checkpoint final](evidence/integrated_prediction_pilot_v1.json)
contiene un coche detectado con geometría 3D, velocidad aprendida y score, junto
a TTC firmado **−5,23490 s**, fase **−0,0189224** y límites de fase
**[−0,0224637, −0,0155820]**. PHASE tardó 0,71745 s en CPU con los tensores ya
preparados; no es latencia integral. La política seleccionó un objeto distinto
del ejemplo anterior: sus TTC no deben compararse como si fueran del mismo track.

Los límites `q10` y `q90` pertenecen al espacio de fase. Si cruzan cero, no se
presentan como un intervalo continuo de TTC. Este replay usa secuencias a las
que PHASE ya estuvo expuesto, sin GT TTC alineado: demuestra transporte e
inferencia, no una nueva medida de precisión TTC ni cobertura calibrada.

## Límites y trabajo siguiente

La siguiente inversión útil es ampliar el entrenamiento de detección con
soporte real de las ocho clases, validación por grupos de adquisición y un
protocolo fijado antes de nuevos resultados. Después se podrá medir la pérdida
de calidad TTC causada por ROIs e identidad imperfectas y comparar RGB frente
a RGB+eventos con controles equivalentes. Compartir backbone o entrenar ambas
tareas conjuntamente sigue siendo un experimento posterior.

Para una submission oficial faltan la confirmación del contrato y scorer de
detección, la convención física de velocidad y completar los medios test
ausentes. No se usaron etiquetas Test12 ni se envió una submission. El coste
cloud de estas pruebas es cero; las mediciones actuales no justifican consumir
el presupuesto RunPod para este piloto.

## Reproducción y conservación

Los pesos, imágenes, anotaciones y predicciones por objeto permanecen en
`output/` local. Git conserva código, configuración, hashes, tablas y recibos
agregados. Los runs anteriores al commit de implementación incluyen snapshots
de código exactos, además del HEAD, para no atribuir a Git contenido que entonces
todavía no estaba versionado. El piloto parte de
`3ebac039e56e3e5af98e0e3302084f00923235a1`.

Para regenerar las tablas desde los recibos versionados, sin dataset ni GPU:

```powershell
python scripts/summarize_detection_runs.py --docs docs/detection_local_20261029
```

La revisión final corrigió dos protecciones para futuras ejecuciones: admisión
del manifest y split antes de construir el detector, y hashes del manifest de
medios/calibración en todas las rutas de inferencia. El
[recibo retrospectivo de admisión](evidence/pilot_manifest_admission.json)
confirma que los inputs concretos del piloto ya eran válidos; no se atribuye al
run antiguo una protección que se incorporó después. Las comprobaciones de
métricas entre intérpretes permiten solo redondeo flotante de 1e-12, conservando
conteos, clases, tokens y estructuras exactos.

La [verificación final](evidence/final_rgb_verification.json) registra **232
tests correctos**, sin errores ni skips, Ruff y formato sobre 49 archivos Python
y Pyright sin errores en ocho módulos nuevos del detector. Una ejecución real
del lanzador rechazó un split con hash distinto antes de construir el modelo.
Estas comprobaciones cubren los cambios de detección, datos y puente; no son
la suite completa de PHASE ni una nueva evaluación del artículo.

La [revisión independiente final](INDEPENDENT_FINAL_REVIEW.md) no deja hallazgos
materiales abiertos en esta entrega. Recomputó los controles y verificó hashes,
paridad, tablas y preservación; no se presenta como otro entrenamiento GPU.

La evidencia del artículo y el código histórico de PHASE permanecen intactos.
El [informe de integración inicial](LOCAL_EXECUTION_REPORT.md) describe una etapa
anterior, cuando aún no se habían ejecutado estos entrenamientos.
