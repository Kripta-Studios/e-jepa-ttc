# C-OBS: corregir RGB cuando los eventos aportan evidencia

## Relación directa con el release

La fusión final conserva un ancla RGB y aprende una corrección. En transferencia, RGB tiene menor MAE medio que la fusión en la cohorte del artículo. Esto sugiere estudiar cuándo la corrección ayuda, pero no demuestra que el problema sea falta de observabilidad. Puede ser desajuste de calibración, exposición, normalización o sesgo aprendido.

El primer producto de C-OBS será una tabla por consulta con error de R0, error RGB, tamaño/signo de corrección y evidencia física disponible. No empezar entrenando una confianza antes de comprobar qué predice realmente los fallos.

## Diagnóstico O0

Registrar `gain = loss(RGB, y) − loss(R0, y)` en desarrollo. Un gain positivo indica que la corrección original ayuda. Es una variable de evaluación, nunca una entrada de inferencia. Calcularla en fase y segundos: no tienen por qué ordenar igual los ejemplos.

Cruzar gain con cuatro familias de variables: soporte ya presente; discordancia entre expertos; innovación temporal; estimabilidad de expansión. Separar por grabación, cámara, banda, tamaño y oclusión. Una correlación global puede reflejar identidad del dominio y desaparecer dentro de cada secuencia.

Como techo diagnóstico, calcular el selector oracle por consulta entre RGB y R0, con etiquetas. Declararlo irrealizable: su distancia al baseline cuantifica oportunidad máxima de ese selector, no rendimiento esperable de un gate. No usar su score como candidato.

## Medir información sobre q, no exigir todo el flujo

En el modelo `y=Hβ+ε`, usar β=[Axx,Axy,Ayx,Ayy,bx,by] y `q_local=(Axx+Ayy)/2`. La cantidad es estimable si `c=[0.5,0,0,0.5,0,0]` pertenece al espacio fila de H. Medir mediante SVD el componente de c en el núcleo y la sensibilidad de la estimación en las direcciones observadas. Normalizar coordenadas de forma documentada y transformar c con las mismas unidades.

Una pseudoinversa siempre puede devolver un número; eso no prueba estimabilidad. Un prior ridge estabiliza un número mediante una hipótesis, no añade observaciones. Tampoco basta una Fisher alta: normales correlacionadas, fondo mezclado y error en H pueden producir falsa confianza.

Stage68 ya implementaba expansión isotrópica+traslación como fallback, además de exigir 32 planos antes de ajustar. Revisar su cascada de rechazo evita repetir el mismo intento. La prueba nueva se ejecuta en consultas del release con idéntica ROI y frontera causal; los resultados históricos no se heredan como si fueran los de R0. [Código histórico](../../src/e_jepa_ttc/models/normal_flow_routing_v10.py).

## Variables propuestas y controles

Empezar con un máximo de seis indicadores nuevos: fuga al núcleo de c, sensibilidad de q, residuo robusto, consistencia temporal, mezcla/visibilidad y fracción de soporte independiente estimada. Definir unidades, valores ausentes y coste. No aceptar un descriptor por llamarse «físico» si usa targets o información futura.

Comparadores: soporte actual; condición global; event count; mismo gate con indicadores permutados dentro de grupos de soporte; misma capacidad con campos constantes; indicadores físicos. Los ensayos F2/Fcap ya probaron más tiempos/recuentos sin pasar sus criterios: el nuevo brazo debe aportar información distinta. [Antecedentes](../physics_review_20261031/04_LECCIONES_EXPERIMENTALES.md).

La permutación no debe crear una señal trivial de inválido. Conservar distribuciones, máscaras y estratos definidos sin targets. Presentar casos de bajo rango con q identificable, y casos bien condicionados pero sesgados por fondo.

## Integración mínima O1

Conservar la corrección cruda original δφ y probar:

```text
φ_candidate = φ_RGB_raw + λ(x) * δφ_original
0 ≤ λ ≤ 1
```

El gate únicamente atenúa; no extrapola una corrección nueva. Un parámetro global constante es el primer control, junto con λ=1 (R0) y λ=0 (RGB). Si la mejor regla es constante, reportar calibración global, no observabilidad contextual.

En el primer diagnóstico, mostrar también el riesgo de apagar una corrección que arregla sesgo RGB: δφ depende de ambos sensores. No sumar precisiones como si RGB y eventos fueran independientes. El gate puede usar información de ambas ramas, manteniendo la hipótesis de evidencia incremental como objeto de evaluación.

Las anchuras requieren un contrato explícito. Para el diagnóstico de punto se conservan y se declaran no recalibradas; para un candidato probabilístico, ajustar las anchuras/intervalos en una partición de calibración separada. Multiplicar δφ por λ no valida automáticamente q10/q90.

## O2: refit de una fusión con calidad física

Solo si O0/O1 muestran información incremental. Ampliar la interfaz de 20 campos con los indicadores admitidos y ajustar una cabeza pequeña con control de parámetros y updates. Su referencia es la cabeza original reajustada bajo la misma genealogía, además de R0 congelado. La reconstrucción OOF debe abarcar todos los padres que producen sus entradas.

Medir utilidad de eventos mediante entradas reales, ZERO y permutación compatible. ZERO sobre pesos ya entrenados mide dependencia de inferencia; para atribuir beneficio al entrenamiento, comparar también controles reentrenados. La cabeza no debe identificar una secuencia y memorizar su política de fallback.

## Qué justificaría incorporarlo

Con cobertura completa, menos daño de correcciones y conservación de las útiles, mejora agregada frente a R0 y frente al gate constante, sin degradar bandas crucial/negativa. Publicar también el error de los casos en que se ignoran eventos; no descartarlos del denominador.

Evaluar curvas error-confianza, frecuencia de errores graves aceptados, calibración y coste. Un intervalo de q que cruza cero induce TTC no acotado o separado en dos ramas; no producir un intervalo simétrico de segundos como si fuese gaussiano. Una abstención interna puede guiar la fusión, pero el formato del benchmark conserva la salida definida del sistema.

Parar si el descriptor solo reproduce recuento, si su confianza aumenta en estimaciones sesgadas o si el gate pierde el beneficio nativo de eventos. En ese caso estudiar medición/sensor antes de hacer la cabeza más grande.
