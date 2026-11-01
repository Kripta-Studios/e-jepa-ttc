# Recuperación de q con escala fotométrica estimada y movimiento accesorio

El cambio RGB observado explica mejor los eventos que el modelo radial ligado
a R0. Antes de entrenar un productor, comprobar si la medición permite recuperar
q cuando traslación e iluminación dejan de confundirse con expansión.

Reutilizar los recuentos RAW, la máscara común y los ajustes `rgb_observed` de
S1-reference, sin abrir targets TTC. Cada adquisición usa exclusivamente el
ajuste de escala de las otras 23. Una pendiente no positiva invalida la medición;
no inventar calibración. Leer solo la referencia RGB ya verificada para recuperar
los gradientes que la caché radial no conserva. No cambiar ROI, celdas ni muestras.

Modelo local de tasa logarítmica: -grad(L)·(A(x-centro)+b)+l. Usar una unidad
espacial isotrópica para posición y velocidad. q=traza(A)/2. Comparar cuatro
lecturas fijadas: radial+iluminación (2 parámetros), radial+traslación+iluminación
(4), afín+iluminación (7), y afín con correspondencia espacial permutada (7).
La permutación es determinista por identidad de consulta, semilla 7.

Ajustar sobre las mismas celdas de cada consulta con cinco iteraciones IRLS,
Huber 1,345*MAD normalizada (factor 1,4826, suelo 1e-6). Exigir al menos 32
celdas y comprobar por SVD que el funcional q pertenece al espacio observable,
rtol=1e-8. No exigir identificación de parámetros que q no necesita; registrar
rango, grados residuales, sensibilidad y residuo. No presentar el residuo como
incertidumbre calibrada ni las celdas como observaciones independientes.

La tasa media del intervalo se interpreta en su punto medio; transportarla al
instante de consulta con velocidad axial constante. Mantener signo y rechazar
singularidades antes de la emisión oficial. El desfase de la referencia RGB,
ISP, umbrales asimétricos, paralaje y aproximación de gradiente siguen explícitos.

Sellar mediciones y ausencias antes de unir los targets de desarrollo existentes.
Evaluar q, signos y TTC de cada lectura, siempre sobre la misma población.
Informar la lectura sola y el sistema con fallback exacto a R0, preservando su
ausencia original. Comparar también con el contexto de eventos. CI pareado por
adquisición, no por celda. No seleccionar el mejor brazo para reescribir esta
prueba ni afirmar confirmación independiente. No abre JEPA/S2 automáticamente:
primero hay que demostrar medición útil, atribución y valor para la cadena.
