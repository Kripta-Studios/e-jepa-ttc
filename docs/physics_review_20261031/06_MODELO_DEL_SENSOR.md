# Modelo del sensor: buen auxiliar posible, con más requisitos

## Qué hipótesis es nueva respecto a la cadena actual

La cadena usa representaciones, supervisión de geometría, relaciones de un profesor y TTC. En las rutas finales revisadas no se identificó una loss que exija explicar los cruces de umbral del sensor. Pedir consecuencias sensoriales a un estado físico sería una hipótesis distinta de predecir el embedding del profesor o añadir una penalización de altura.

En el modelo ideal, un píxel produce un evento al cruzar un umbral de incremento de log-intensidad respecto a su referencia. La restricción relaciona movimiento, gradiente espacial y cambio temporal; sus desviaciones incluyen umbral, ruido y dinámica de respuesta. [Survey de visión de eventos](https://arxiv.org/abs/1904.08405).

El trabajo de Guo, Hamann y Gallego combina estimación de flujo e intensidad mediante formación de eventos, error fotométrico, contraste y consistencia temporal. Demuestra que un objetivo de este tipo puede entrenarse; no demuestra mejora en TTC ni compatibilidad directa con el adaptador de PHASE. [Trabajo y fuente primaria](https://arxiv.org/abs/2503.17262).

## Por qué no lo pondría como segundo entrenamiento inmediato

La ecuación L_t+u·grad(L)=b tiene ambigüedades. Si b es libre por píxel, absorbe movimiento. Si contraste y umbral son libres, se pueden reescalar juntos. El flujo solo queda restringido en dirección del gradiente local, y un fondo en movimiento puede dominar el ajuste. Un buen residuo de reconstrucción puede coexistir con q incorrecto.

En PHASE, la proyección intersensor omite traslación dependiente de profundidad; Slider mantiene calibración provisional. El tensor RGB no equivale directamente a log-radiancia del sensor de eventos: hay respuesta, exposición, ganancia, saturación y diferencias espectrales. El propio eAP registra exposición y balance de blancos automáticos. [eAP](https://arxiv.org/html/2603.16303v1). Un término fotométrico fuerte podría castigar la física correcta por un error de calibración.

Las cachés raw17, expert3, PAIR133 y phase340 no conservan toda la secuencia de cruces por píxel. El auxiliar exigiría reabrir datos crudos o construir una caché específica que preserve ese contrato. Su coste de preparación y almacenamiento debe medirse antes de lanzar entrenamiento.

## Primer modelo reducido razonable

Usar parches pequeños con un mapa de textura de referencia, movimiento local parametrizado por expansión y traslación, y un cambio de iluminación global o de muy baja dimensión. Fijar inicialmente umbrales o su escala; permitir asimetría positiva/negativa solo cuando exista un control de identificación. No dar a cada píxel parámetros suficientes para explicar cualquier secuencia.

Predecir incrementos integrados y polaridad con una loss robusta, conservando máscaras de saturación, visibilidad, cobertura y mala alineación. La igualdad exacta al umbral es una aproximación ideal; no convertir ruido y latencia de disparo en targets exactos. Un likelihood probabilístico requiere definir su ruido, no basta llamar NLL a un error cuadrático.

Separar el ensayo en tres condiciones: movimiento con textura fija; iluminación variable con objeto inmóvil; ambos combinados. Añadir dominio con umbrales y ruido diferentes. El control auxiliar debe tener parámetros y presupuesto comparables, para distinguir información física de regularización o trabajo extra.

La evaluación principal sigue siendo q/TTC, signo, cobertura y respuesta a intervenciones. Reconstrucción de imagen o flujo más nítido no justifica promoción por sí sola. Si el auxiliar se retira tras entrenamiento, su coste extra afecta al entrenamiento; si se conserva para medir confianza, también al runtime completo.

## Silencio como observación censurada

Entre cruces, el cambio respecto a la referencia debe permanecer entre −C− y C+ bajo el modelo ideal. E-NeRF ya usa una loss de ausencia de eventos; no es una idea nueva por sí misma. [E-NeRF](https://cvai.cit.tum.de/_media/spezial/bib/klenk2023nerf.pdf).

La inferencia solo es válida si se observó el píxel y se conoce o marginaliza su referencia. Al cortar una ventana, la referencia anterior al inicio no pasa a cero por conveniencia. Un paquete perdido, un píxel muerto, un periodo refractario, un stream filtrado o un subsampling no son silencio físico. Penalizar eventos ausentes tras dropout como si el sensor no los hubiera generado introduciría supervisión falsa.

Conviene empezar con intervalos de un simulador donde la referencia y cobertura sean conocidas. Después usar el stream real completo con estado de sensor y máscaras. Los fixtures actuales de [phase_event_expansion/synthetic.py](../../operational/phase_event_expansion/synthetic.py) declaran que son contornos ideales, no simulación fotométrica: no validan este auxiliar.

Robust e-NeRF aporta un antecedente para umbrales asimétricos y refractario; su reconstrucción 3D no es un requisito para adoptar esas precauciones. [Fuente](https://arxiv.org/abs/2309.08596).

## Criterio de abandono

No continuar si el modelo explica observaciones usando iluminación/umbral sin recuperar movimiento, si el beneficio desaparece al controlar capacidad o si solo funciona con alineación y parámetros de sensor GT no disponibles en inferencia. Si la calibración intersensor domina el error, se puede estudiar primero el auxiliar solo con eventos y medir por separado cuánto añade RGB.

El coste y las ambigüedades hacen de esta idea una segunda etapa razonable. No justifica abrir un NeRF ni cambiar la representación completa antes de terminar la auditoría temporal y de soporte.
