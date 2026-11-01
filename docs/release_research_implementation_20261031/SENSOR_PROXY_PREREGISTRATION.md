# S1 exploratorio con los datos disponibles

El usuario confirma que no existen adquisiciones reservadas ni calibración
fotométrica adicional y pide continuar con los datos actuales. Este ensayo usa
esa autorización sin convertir desarrollo expuesto en confirmación independiente.

Se mantiene el banco S0 y sus ambigüedades. Aquí se pregunta si una referencia
RGB aproximada y el estado q de R0 explican actividad firmada de eventos mejor
que controles igualados. No se identifica C físico ni se trata RGB procesado
como log-intensidad calibrada del sensor. No se cambian pesos ni se leen TTC GT.

Población prefijada: las 149 consultas del screen. El análisis fotométrico usa
las 72 consultas de 24 adquisiciones EvTTC con calibración geométrica; Slider
se conserva como no admitido por alineación provisional. No aplicar una loss
píxel a píxel fuerte a Slider. Los fallos y los denominadores se publican.

Para cada consulta, usar las dos últimas ventanas de eventos del último
snapshot y la última imagen RGB anterior o igual al inicio de la ventana actual.
Todo el soporte queda dentro de 650 ms y de la consulta. La referencia es
anterior a los eventos reconstruidos; q procede de la fase cruda R0 al final
de la consulta y se transporta hacia atrás con la hipótesis axial constante.
Es explicación de eventos ya observados, no predicción independiente del futuro.

Proyectar puntos del sensor de eventos a RGB con intrínsecas, distorsión y
rotación inversa de la calibración existente. La traslación/paralaje queda sin
resolver y se registra. Dividir la ROI original en 16x16 celdas. Promediar el
gradiente espacial de log-luminancia RGB en una cuadrícula fija 5x5 por celda,
con diferencias centrales de un píxel del sensor de eventos. Exigir 80% de
muestras mapeables y luminancia de referencia entre 5 y 250. La máscara no
usa TTC ni se aprende. Para el ajuste usar celdas con algún evento observado;
las celdas silenciosas se cuentan, pero no etiquetan q=0 ni añaden loss de silencio.

Target auxiliar: (n_pos-n_neg)/(área de celda * duración), con polaridades y
ambas duraciones explícitas. La igualdad de umbrales y respuesta común son
aproximaciones del proxy. Comparar cuatro modelos con dos parámetros globales
por fold: pendiente no negativa e intercepto de iluminación, sin parámetros
libres por píxel, consulta o adquisición de evaluación.

1. Actividad: tasa firmada de la ventana anterior.
2. Textura: -grad(log RGB)·(x-centro), q constante igual a 1.
3. Física: el mismo término multiplicado por q de R0.
4. Permutación: q desplazado cíclicamente entre consultas de la adquisición;
   es un control diagnóstico contrafactual, no una entrada causal de despliegue.

Ajustar mínimos cuadrados ponderados con adquisición excluida; igual peso por
adquisición, consulta y celda dentro de su nivel. La pendiente negativa se
proyecta a cero y se reajusta el intercepto. Misma máscara y población para
todos los brazos. 24 folds x 4 brazos, sin búsqueda de hiperparámetros.
Publicar MSE/MAE macro, coeficientes, cobertura, errores por adquisición y CI95
pareado por adquisición (2000 remuestreos, semilla 7). Guardar extracción ciega
antes de ajustar. El estado R0 y los padres tienen exposición histórica.

Una ventaja consistente sobre los tres controles es señal para diagnosticar
alineación/escala; no demuestra calibración física ni mejora TTC. No abre S2
automáticamente. Si no hay señal, registrar el fallo del proxy, sin concluir
que un auxiliar con referencia y calibración correctas sea imposible.

Las unidades, proyección y regresión se verifican antes con fixtures.

## Enmienda de recursos anterior a la extracción

La comprobación local registra 32 CPU lógicas, 65,1% de carga total, 5,89 GiB
de RAM disponible y aproximadamente 6,4 MB/s de lectura agregada en tres SSD.
Se permite un perfil de una consulta en CPU, sin CUDA visible y con OMP/MKL/
OpenBLAS limitados a un hilo. Si su pico de memoria es menor de 1,5 GiB y quedan
al menos 4 GiB disponibles, continuar la extracción auxiliar de 72 consultas
RAW sin reservar GPU. Registrar consumo y progreso del worker principal.
Esta excepción de capacidad no permite duplicar ni modificar population2608.
Si no se cumplen esos límites, esperar a que termine el worker.
Evidencia previa: `evidence/SENSOR_PROXY_RESOURCES_BEFORE.json`.
