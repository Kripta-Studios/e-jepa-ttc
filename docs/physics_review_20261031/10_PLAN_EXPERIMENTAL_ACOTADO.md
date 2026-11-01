# Plan experimental propuesto, sin ejecución

Este plan responde a si merece la pena probar las ideas. No lanza entrenamiento ni modifica la entrega. Los presupuestos siguientes son límites propuestos, no duraciones medidas ni compromisos de alcanzar una mejora.

## Paso 0: fijar qué modelo se intenta mejorar

Conservar dos referencias separadas: F0/TRAIN40 del artículo y `PHASE-TTC-Adapt-20261031`. El primero tiene resultado oficial GarlTTC; el segundo añade adaptación local. Cada intervención identifica pesos, normalizadores, cámaras, historia, política de fallback y emisión. El test privado y las 434 consultas expuestas no seleccionan configuración.

El objetivo de sistema debe incluir precisión local comparable, signo, cobertura, colas y coste. El objetivo de representación exige además retirar los auxiliares. Si no se distinguen, una mejora de predictor extra puede atribuirse al encoder de forma incorrecta.

## Paso 1: terminar la prueba temporal existente

**Actualización de cierre:** el screen ya terminó en trabajo concurrente. F0 pasa y el router falla. Aplicar las reglas siguientes a la ampliación de F0 y a la conservación de historias eAP; no repetir el screen ni duplicar la extracción en curso. Véase [resultado y alcance](13_ACTUALIZACION_CONCURRENTE.md).

Usar las 149 identidades, orden y reglas del [screen registrado](../superpowers/plans/2026-10-31-phase-canonical-screen149.md). No crear un subset nuevo favorable. Reproducir el baseline original, aplicar la única intervención CanonicalPair y reconstruir dependencias. Reportar F0 y el router entregado, por carretera, Slider y grabación.

Mantener la regla existente: ninguna predicción nueva ausente, ninguna de las cuatro métricas MAE/RAE macro empeorada y al menos una mejora estricta para justificar expansión. Esta puerta es de screening exploratorio; no certifica significación ni promoción. Si falla, conservar el resultado sin ajustar otra corrección al mismo screen. Si pasa, comprobar toda la población y toda la historia eAP, no solo la entrada PAIR actual.

Separar luego warp de tensores preparados y re-encoding RAW. La primera operación verifica tratamiento de unidades; la segunda comprueba la política real de selección temporal. Registrar signos, saturación, rechazos y observaciones admitidas. El coste es de extracción y replay, con cero updates.

## Paso 2: auditoría de información dirigida a q

Sin entrenar, descomponer Stage68 por causa de rechazo y comparar estimabilidad de la traza afín, ajuste isotrópico, condición global y cantidad de planos. Validar incertidumbre en un conjunto controlado con aperturas, texturas y movimientos mezclados. No bajar umbrales originales para hacer pasar la campaña antigua.

La decisión no debe ser «hay más válidos». Debe ser si la señal adicional distingue q medible de estimaciones sesgadas y mejora una decisión de fusión sobre consultas completas, fuera de los grupos de ajuste. Si solo refleja event count o dataset ID, no abrir la nueva cabeza.

## Paso 3: banco de mecanismos

Construir una matriz pequeña y fijada de trayectorias con Z, velocidad y aceleración conocidas. Cruzar textura, umbral, iluminación, escala, oclusión y fondo. Reservar configuraciones completas, no frames adyacentes, para confirmación del mecanismo.

| Intervención | Variable que debería responder | Fallo que detecta |
|---|---|---|
| α temporal moderado y controles extremos | q/α, αT | Pérdida de segundos o cadencia memorizada |
| Escala espacial fija | T constante dentro de soporte | Atajo de tamaño |
| Iluminación variable, objeto inmóvil | q≈0 con soporte declarado | Confundir actividad con movimiento |
| Aproximación/recesión | Signo y magnitud de q | Positividad impuesta o inversión errónea |
| Fondo estático/móvil independiente | q del objeto, incertidumbre | Confianza espuria por mezcla |
| Oclusión sin cambio de trayectoria | Visibilidad cambia, q no por esa causa | Silueta usada como expansión |
| Aceleración/frenada | Innovación y cambio de régimen | Reloj constante sobreconfiado o retrasado |
| Paquetes perdidos frente a silencio | Cobertura distinta | Censura mal interpretada |
| Zoom conocido de ROI | Estado en sensor conservado | Expansión inducida por coordenadas |

Los contornos sintéticos actuales sirven para geometría. El brazo sensor requiere simulación fotométrica y referencia de umbral. No se necesita añadirlo al primer screen de tiempo.

## Paso 4: un solo brazo aprendido cada vez

| Brazo | Comparadores mínimos | Qué justificaría continuar |
|---|---|---|
| Reloj | Baseline, predictor libre, persistencia/filtro y futuro permutado | Mejora q/TTC con tasas reales, sin retraso extra en frenadas |
| Observabilidad | Calidad existente, control de capacidad, ZERO-event | Información incremental de eventos y menos errores extremos con cobertura completa |
| Sensor | Auxiliar genérico de igual coste y sensor con correspondencia permutada | Movimiento mejor recuperado bajo iluminación/umbral distintos |
| Material/visibilidad, posterior | Transporte actual y ciclo de dos frames | Menos error por oclusión/fondo, con tripletes y soporte comparable |

Para cada familia aprendida propongo un screen con una semilla, arquitectura fija y máximo 1.000 updates por brazo, con techo de cómputo registrado después de medir 100 updates. Si no cabe, parar como ejecución incompleta; no comparar un brazo corto con otro completo. No iniciar todas las familias. Una señal favorable habilita confirmación con tres entrenamientos completos de la representación cuando se reclame ese beneficio.

El número 1.000 es un presupuesto inicial para detectar señal, no prueba de convergencia. Un resultado negativo bajo ese límite debe conservar esa salvedad. La evaluación principal se fija antes; no escoger checkpoint o subset mirando H ni elegir entre muchos pesos auxiliares con el mismo conjunto final.

## Métricas y promoción

Usar MAE/RTE micro y macro por grabación, MiD donde su dominio esté completo, signo, crucial/negativo, mediana, p95, excedencias y errores máximos. Conservar todos los rechazos y el error del fallback. Reportar tamaños de cada banda y no interpretar un intervalo ancho que cruza cero como equivalencia.

Para incertidumbre: cobertura/anchura en q, score propio, respuesta a corrupción y falsos casos de alta confianza. Derivar TTC con cuidado al cruzar q=0. Para dinámica: error futuro por horizonte, innovación, error de tasa y retraso de alerta bajo aceleración. No sustituir estos resultados por smoothness.

La confirmación debe usar bootstrap pareado por adquisición/secuencia y variabilidad de entrenamiento. Si hay pocas secuencias, presentar el límite. Fijar márgenes prácticos de no inferioridad antes de abrir la confirmación; no inventarlos tras conocer los deltas. La cobertura y causalidad son requisitos. Una mejora estadística minúscula sin beneficio práctico tampoco justifica una cadena más cara.

Comparar coste con mismas entradas, precisión, hardware y frontera de medición; incluir preparación, historia, componentes y host-output. Los auxiliares retirados no cargan inferencia; un observador o solver retenido sí. Probar cold start, pausas y varios objetos.

## Orden y reglas de parada

Orden recomendado: screen temporal → autopsia de observabilidad → mecanismos → un brazo de reloj o confianza justificado → sensor → combinación de brazos que ya funcionen por separado. Si ningún brazo mejora su control, no combinarlos esperando compensación. Si mejora el mecanismo pero no TTC, conservarlo como diagnóstico. Si mejora TTC solo con información privilegiada no disponible, no presentarlo como mejora desplegable.

Resolver adquisición nueva, calibración y targets puede ser una inversión más informativa que más GPU. La nueva hipótesis debe identificar una discrepancia observada, la intervención mínima que la corrige y el resultado que la refutaría. Ese contrato evita otra campaña abierta de arquitectura.
