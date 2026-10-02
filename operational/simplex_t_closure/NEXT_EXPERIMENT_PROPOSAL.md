# Una continuación propuesta: replicación acotada de H16

**Estado: propuesta, no ejecutada.** La autorización de esta sesión sigue siendo
cero updates. `TPR-D1-H8-C160` permanece como candidato registrado. Esta propuesta
requiere un protocolo nuevo; no modifica el freeze cerrado ni abre Stage76.

Proponemos únicamente completar las cabezas `TPR-D1-H16-C160` con seeds 13 y 23
en los tres folds, conservando D1, productores, normalización, contexto causal,
batch128, CPU FP32 y endpoint2500. Son seis fits y 15.000 updates **propuestos**.
Los H8 de esas seeds ya existen como controles; no se volverían a entrenar.
La cabeza H16 seed7 ya existe. La comparación resultante tendría tres seeds de
cabeza y las mismas nueve secuencias OLD_DEV, sin aumentar independencia de datos.

## Por qué esta prioridad concreta

La ganancia exploratoria de H16 seed7 frente a H8 es pequeña y su intervalo
jerárquico cruza cero. El experimento reduciría la incertidumbre de optimización
de una pregunta acotada: si extender la historia mejora de forma estable el error
de estimación. No demostraría un mecanismo cronológico, reacción AEB o tracking.
Las entradas necesarias son las tres cachés H16 ya verificadas. Antes de una
eventual ejecución se deben volver a medir recursos y confirmar su disponibilidad;
la disponibilidad durante el cierre no es una autorización futura.

Ampliar diversidad TRAIN con un par H1/H8 fijo tiene mayor alcance científico,
pero los 22 grupos de expansión inspeccionados ya pertenecen a D1. No se ha
establecido en el alcance autorizado un lote adicional independiente con etiquetas,
licencia y roles nuevos. No se presentarán datos no inspeccionados como disponibles.
La adquisición exigiría definir los roles y exclusiones antes de mirar resultados.

La agregación invariante merece atención después de EWMA: separaría invariancia
al orden de almacenamiento, conservando pares feature-edad, de eliminación real
de edades/orden. Requiere otra receta y no se combinaría con esta replicación.
Aquí se prioriza una intervención con coste exacto y entradas ya auditadas;
las otras dos líneas son alternativas aplazadas, no lanzamientos adicionales.

EWMA quedó completada en los tres folds. Su score global no mejora la mediana
actual; los valores y los intervalos post hoc se conservan en
[METRICS.csv](supplement/METRICS.csv) y
[ARITHMETIC_VERIFICATION.json](supplement/ARITHMETIC_VERIFICATION.json).
Esto motiva separar agregación simple y cabeza aprendida, pero no convierte una
nueva agregación en una reparación equivalente de la campaña cerrada.

## Registro previo que tendría que existir

Antes de entrenar, un nuevo protocolo debe fijar esos seis IDs, hashes de entradas,
endpoint, comparador H8 por seed, MiD y contraste jerárquico/por secuencia con la
receta vigente. Fijar también límites para errores de signo y bucket crucial,
criterios de parada operativa, presupuesto físico y tratamiento de fallos. No
elegir hiperparámetros o checkpoints con OLD_DEV ni cambiar umbrales de cambio
rápido. Guardar resultados negativos y comparar pérdidas emparejadas; la media de
pérdidas de seeds no es un ensemble de TTC. La decisión no puede basarse sólo en
un p-valor o en el mejor fold. Ningún outcome sustituye automáticamente al H8
registrado. La confirmación conjunta con Stage76 queda reservada para un contrato
separado, posterior, sin apertura en esta sesión.

## Literatura como hipótesis posterior

Se recuperaron HTML de [MotionJEPA](https://arxiv.org/html/2609.23881v1) y
[PixelUMM](https://arxiv.org/html/2609.38597v1), y el PDF de
[FAR](https://arxiv.org/pdf/2609.34677). Se inspeccionaron resúmenes y métodos;
no se declara lectura íntegra de todos sus apéndices.

MotionJEPA regulariza embeddings estáticos y predice el embedding de diferencias
visuales con un módulo auxiliar. Su modelo de mundo conserva acciones en el
predictor principal. Orienta un futuro encoder entrenable y no corrige una cabeza
supervisada sobre A5 congelado. Su demostración no es sobre cámaras de eventos/TTC.

FAR supervisa recuperación mediante utilidad predictiva del futuro durante TRAIN;
la recuperación en inferencia no necesita ese futuro. Orienta una futura selección
de observaciones útiles, después de controles simples, y no otro selector de tres
TTC. Su sistema de navegación/modelo de mundo no constituye evidencia AEB.

PixelUMM conecta patches/tubelets de píxeles con un backbone multimodal sin un
encoder visual/VAE independiente en la interfaz principal. Es una referencia sobre
la interfaz de representación, no una dependencia ni una propuesta de desplegar
un gran modelo multimodal en esta campaña. No se instalaron paquetes externos ni
se descargaron sus checkpoints.
