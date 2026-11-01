# C-SENSOR: mejorar un productor explicando los eventos

## Beneficio buscado en el release

El propósito es que el productor de eventos extraiga una señal de aproximación menos dependiente de textura, contraste e iluminación. La hipótesis no requiere que el release ejecute un reconstructor 3D ni un profesor en producción. El auxiliar se usa durante entrenamiento y se retira al inferir, salvo que otro ensayo justifique conservar sus diagnósticos.

El primer punto de inserción propuesto es **E_C2F**, manteniendo la rama RGB intacta. Así se puede estudiar un productor sin cambiar a la vez los tokens que consume PAIR_E de E_A5. Antes de implementar, comprobar que la ruta ejecutada expone mapas espaciales adecuados. Si no los tiene, registrar el coste de obtenerlos y decidir entre E_A5 o una lectura auxiliar pequeña; no inventar un acceso disponible desde las cachés finales.

## Requisitos de dato

PAIR133, PHASE17 y el estado340 no contienen el historial de cruces por píxel. Construir una caché específica de parches a partir de datos RAW, con coordenadas, tiempos, polaridades, cobertura y referencia anterior al inicio. Mantener separadas preparación de entrenamiento y futura entrada de inferencia.

La alineación por rotación no resuelve paralaje; RGB con respuesta ISP, exposición y balance automáticos no equivale a log-intensidad del sensor de eventos. Empezar con parches de calibración conocida o con textura latente de eventos anclada, y comparar luego si RGB añade información. No usar una pérdida píxel a píxel fuerte sobre Slider con calibración provisional.

## Modelo físico reducido propuesto

En el sensor ideal, el cambio de log-intensidad desde la referencia cruza `p*C_p`. Parametrizar movimiento local con expansión, traslación y, cuando el diagnóstico lo requiera, términos afines limitados. Vincular la expansión a q del estado; si el auxiliar tiene una cabeza de flujo independiente que ignora q, puede reconstruir eventos sin mejorar TTC.

Una parametrización inicial es `u(x)=b_xy + A(x−x0)`, con traza(A)/2 vinculada a q bajo las hipótesis locales admitidas. La textura de referencia se transporta con ese campo. Añadir una variación de iluminación global por parche o de muy baja dimensión. Fijar inicialmente C+, C− o su escala por sesión de calibración. Evitar textura, iluminación y umbral libres por píxel y tiempo.

El auxiliar predice cambios integrados y polaridades con residuos robustos. Puede empezar con intervalos anclados en eventos observados y una tolerancia explícita, antes de intentar un likelihood completo de timestamps. La suma firmada de umbrales aproxima el cambio entre referencias; el incremento de una ventana arbitraria incluye residuos de umbral en ambos extremos. No imponer igualdad exacta donde se desconoce ese estado.

La identidad del objeto, visibilidad y cobertura limitan qué intervalos participan. Las máscaras no se aprenden con libertad para apagar todas las observaciones: fijar una cobertura mínima y reportar el denominador, con control que use las mismas máscaras.

El modelo generativo y la estimación conjunta de intensidad/flujo tienen antecedentes operativos en [Guo et al., ICCV 2025](https://arxiv.org/abs/2503.17262). Su resultado respalda la viabilidad de la tarea auxiliar; no prueba mejora de este productor ni de TTC.

## Secuencia de experimentos

**S0: identificación en simulación fotométrica.** Con movimiento y parámetros de sensor conocidos, comprobar recuperación de q al variar luz y umbral. Incluir escenas que produzcan igual recuento con velocidades distintas. Los contornos ideales existentes son insuficientes para validar este brazo.

**S1: calibración local sin cambiar el release.** Fijar textura/sensor cuando se conocen y comprobar si la medición predicha desde estados R0 explica los eventos. Si domina el error de alineación o referencia, corregir el dato antes de entrenar. Separar fallo de sensor y fallo de movimiento.

**S2: ajuste de un productor.** Partir de la misma inicialización permitida en cada comparación y mantener la loss de tarea existente. Comparar productor reajustado sin auxiliar, auxiliar genérico de igual coste, auxiliar físico y auxiliar físico con correspondencias temporales permutadas. Aplicar la misma normalización, ventanas y presupuesto.

**S3: integración.** Reconstruir las features físicas y descendientes afectados con genealogía compatible. Comparar inserción congelada y reajuste de contexto/fusión, con un control reajustado equivalente. El resultado principal es la cadena completa frente a R0, no una imagen reconstruida más nítida.

Para la primera prueba, congelar la mayor parte del productor y adaptar una parte registrada. Si falta señal, un resultado negativo no descarta todos los entrenamientos posibles, pero tampoco justifica un escalado automático. El siguiente cambio necesitaría un diagnóstico concreto de capacidad o gradiente.

## Silencio como segundo auxiliar

Solo después de S0–S2, incorporar restricciones entre eventos: el incremento respecto a la referencia permanece dentro de los umbrales. Un píxel sin eventos puede ser uniforme y móvil; su silencio no etiqueta q=0. Una pérdida de paquetes, un píxel muerto y un periodo refractario requieren máscaras/modelos distintos.

Simular dropout conservando la etiqueta de pérdida; no usar los intervalos fabricados por esa augmentation como silencio físico. Contrastar pérdidas de evento, de silencio y ambas con el mismo presupuesto. [Robust e-NeRF](https://arxiv.org/abs/2309.08596) modela umbrales asimétricos y refractario; sirve como referencia de los efectos que hay que controlar.

## Medidas y criterios de descarte

Reportar TTC/fase, signo y errores extremos por condición de luz/sensor; error de q en simulación; residuo sensorial; recuperación de umbral si se estima; fracción de datos admitidos. Quitar el auxiliar al evaluar el productor y verificar que conserva la mejora. Si solo mejora el residuo del sensor, no promover.

Parar si iluminación absorbe el movimiento, si q no afecta a la predicción sensorial, si el beneficio desaparece con control de capacidad o si depende de parámetros GT que no existirían al inferir. El coste extra de extracción y entrenamiento se contabiliza aunque el auxiliar no permanezca en producción.
