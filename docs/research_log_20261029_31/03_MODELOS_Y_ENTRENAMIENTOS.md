# 03. Qué modelos se crearon y entrenaron

## Grafo original y frontera de adaptación

El modelo recibido es PHASE-TTC TRAIN40. Su cabeza puntual `PhaseOnlyFinal`
selecciona veinte coordenadas físicas de un estado de 340 dimensiones y utiliza
una MLP `20 → 32 → 32 → 3`, con 1.955 parámetros. La salida puntual parte del
contexto RGB y recibe una corrección en fase. El namespace y los identificadores
históricos de checkpoints se mantuvieron para preservar compatibilidad y trazas.

Los trabajos nuevos no sustituyen la evidencia de entrenamiento de los productores
originales. Cada experimento declara si modifica esa cabeza, añade una fusión
externa o deja todos los componentes previos congelados. Un bundle con todos los
pesos necesarios para inferencia no implica que todos se hayan reentrenado.

## F1: reajuste de la cabeza original

F1 conserva capacidad y entradas de F0. El refit completo contiene 92.315 filas,
dos épocas y 1.444 actualizaciones por semilla. Se guardaron las semillas 7, 13 y
21. La semilla de entrega 7 estaba predefinida; no se eligió retrospectivamente
por el mejor resultado de test. Los estados se materializaron desde el pipeline
sensorial y se recuperaron shards válidos al reanudar una extracción interrumpida.

Las filas EvTTC del refit incluyen las 32 grabaciones nativas. Por eso los scores
del refit son ajuste, incluso cuando se muestran separados por dominio. Una
segunda campaña hizo 21 entrenamientos: siete familias por tres semillas,
excluyendo la familia evaluada y reiniciando cada vez la cabeza original. Esa
campaña conservó TRAIN40 en entrenamiento y retiró las ocho grabaciones solapadas.

Se probaron además una consistencia temporal fija T0, controles geométricos,
predicciones de expertos y probes. F2/Fcap tal como estaban propuestos en el
plan inicial no se abrieron en esta primera campaña: la anchura nativa no ofrecía
una asociación consistente dentro de grabación para justificar esa ampliación.
La fusión de calidad posterior plantea otra intervención con controles propios.

## Gate robusto de calidad: 401 parámetros por semilla

La siguiente arquitectura combina expertos neuronales y geometría causal. La
geometría usa cajas suministradas, timestamps reales y ajuste robusto de escala
inversa sobre hasta 650 ms. El gate aprende a ponderar estimadores según la
calidad observable. Ni el error contra target, ni la distancia real, ni el ID de
secuencia entran como features.

Se comparó el gate de calidad con un control de igual número de parámetros.
Ambos tienen disponible el estimador geométrico: el contraste investiga el valor
de la información de calidad, no solo el acceso a una nueva modalidad. La
selección exigía mejoras en macro-RAE y macro-MAE por grabación de carretera y
en RAE/MAE de Slider-500. Los grupos reservados no eligieron la receta de esa
iteración, aunque sí tenían exposición histórica anterior.

El bundle contiene dos identidades:

| Variante | Etiquetas usadas | Uso de sus resultados |
|---|---|---|
| `evaluated` | Desarrollo; CPNAO separado para calibración | Comparación con reservas de la iteración, expuestas históricamente |
| `all_data_refit` | Las 2.677 consultas elegibles | Modelo ajustado con todas esas etiquetas; scores de fit |

Los tres gates se promedian. Sin geometría/historia aplicable se conserva el
modelo nativo. La calibración de la variante evaluada no se transfiere como
garantía al refit que ya ha usado esas mismas etiquetas.

## Router de arranque: cuatro expertos y 688 parámetros

El arranque de Slider-750 motivó una intervención más localizada. El observador
previo requería tres cajas; se añadió un cálculo causal desde dos cajas y un
contrato de frescura. Los cuatro expertos son F0, contexto RGB, contexto de
eventos y geometría de cajas. La entrada de calidad describe soporte y evolución
física de la observación, no su error real.

Las veinte variables incluyen tasas de expansión, desacuerdos entre ejes,
residuos, duración de historia, número/fracción de observaciones, edad,
cuantización, diferencias de escala a corto plazo, aspecto y su cambio,
desplazamiento de centro y dispersión. La definición exacta y su orden están en
`operational/phase_startup/geometry.py`; un vector con las mismas dimensiones
pero distinta semántica no es una entrada compatible.

Se ensayaron dos coordenadas, segundos e inversa de TTC, y tres condiciones:

- `Q`: calidad observable sin la corrupción adicional de expertos de QA.
- `QA`: calidad más perturbaciones de los expertos durante el ajuste.
- `NAcap`: control de capacidad equivalente sin esas nuevas variables de calidad.

Los modelos mantienen 688 parámetros. La receta usa AdamW con learning rate
0,01, weight decay 0,01, 300 updates, clipping de gradiente a 1 y semillas
7/13/21. La normalización usa solo el conjunto de ajuste del fold. La pérdida
combina error relativo protegido cerca de cero y MAE escalado, con balance por
familia. En QA se mezclan filas limpias y corruptas; una fracción de las filas
altera un experto neuronal con factores como −3, −1, 0,25 o 4, sin cambiar targets.

El desarrollo de esta iteración incluye 1.747 consultas de carretera y las 146
Slider. CCRm y CPNAO quedan fuera del ajuste/selección corriente. Esto amplía la
exposición respecto al gate anterior: no se puede atribuir toda diferencia entre
ambos exclusivamente a la arquitectura.

Cada coordenada ejecutó seis folds por tres condiciones por tres semillas:
54 ajustes, seguidos de tres ajustes del candidato. Ambos ensayos suman 114;
los tres refits sobre todos los datos completan **117 ajustes**. Se conservan los
ensayos descartados y sus checkpoints, además de los seleccionados.

## Por qué se retuvo la coordenada inversa

La mezcla en segundos fue peor en carretera agrupada que el gate robusto previo.
La inversa permitió combinar evidencia de expansión y signo con menor influencia
de estimadores de TTC muy grandes. La implementación limita magnitudes internas
de expertos/tasas, con referencia de TTC en `[0,1; 60]` s en valor absoluto; la
cancelación exacta de tasas vuelve a F0. Esto forma parte del modelo, no es un
recorte posterior de las predicciones para acomodar el scorer.

El signo se conserva y la construcción controla la equivariancia ante cambio
conjunto de signos. La falta de historia o su antigüedad superior a 150 ms activa
la salida original. El fallo observado al perder cajas obligó a explicitar esa
frescura: una historia causal puede ser demasiado antigua para representar el
estado de la consulta.

La variante inversa se entregó como candidato adicional. El cambio de 12,8336 %
a 12,6513 % en carretera OOF no justificaba declararla ganadora universal: el
intervalo por familias incluía cero y otras métricas empeoraban.

## PHASE-TTC-Adapt-20261031: receta fija para el nuevo envío

La entrega de competición utiliza la receta inversa QA ya investigada, con tres
routers de 688 parámetros: 2.064 parámetros aprendidos en el ensemble. Conserva
el grafo original de nueve componentes. El fit contiene 2.608 filas de 25
grabaciones y excluye las grabaciones del benchmark del ajuste final.

`eligible_population()` rechaza una entrada que contenga las ocho grabaciones de
carretera excluidas y retira Slider-750/1000. El script exige 2.608 filas y 25 grupos.
Después ejecuta las tres semillas, guarda checkpoints/configuración/población,
exporta TorchScript y comprueba recarga. No elige una nueva receta al observar las
434 predicciones oficiales recién calculadas.

La identidad `MODEL.json` enlaza upstream, freeze del router, exportado e
implementación. Los loaders verifican inventarios y hashes. El empaquetado une
esa identidad con replay, evaluación, entradas de entrenamiento, targets y
commit verificado; una carpeta internamente válida de otro modelo no basta.

El entrenamiento repetido en otra instalación produjo los mismos tres archivos
`checkpoint.pt` byte por byte y las mismas 2.608 predicciones. La lógica del
script ejecutado antes de formatear y la versión publicada tienen el mismo AST;
también se conservaron los bytes originales y sus hashes.

## Lo que estos entrenamientos no cubren

No se entrenaron nuevos encoders de eventos/RGB, un foundation model, un detector
3D ni los contrastes JEPA J0/J1/J2 en esta campaña. No se ajustó un parámetro de
reloj para imitar el PDF. No se usó una reserva históricamente expuesta como si
fuera una adquisición nueva. El término «final» identifica un paquete congelado
y reproducible, no una demostración de que no existe un modelo mejor.
