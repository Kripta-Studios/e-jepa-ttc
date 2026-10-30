# Ficha para consulta de protocolo

Borrador preparado; no enviado.

- Precisar reloj, origen, offset, interpolación e instante físico del TTC de cada
  consulta: inicio/final del par, último frame o timestamp de query.
- Precisar distancia y eje geométricos: centro óptico, frente del vehículo,
  superficie del objeto u otra referencia.
- Aclarar targets y tratamiento de las filas `CCRs1-medium:32` a
  `CCRs1-medium:42`, que quedan fuera del soporte del CSV local al aplicar
  el offset demostrado por igualdad RGB.
- Documentar la transformación de las dos predicciones negativas, incluidos
  vecinos usados, bordes, huecos consecutivos y reglas de agregación.
- Confirmar la población de 434 filas y la frontera de tiempo del runtime.
- Confirmar el uso permitido de EvTTC público para adaptación y cómo declararlo.
- Aclarar si Slider-500 de Event-Aided TTC comparte sesión/adquisición con
  Slider-750/1000, y si existe otra colección o evaluación externa independiente.
- Confirmar permiso de reutilización y semántica de cajas/calibración en esa fuente.

Adjuntar únicamente el resumen y los IDs necesarios cuando el autor autorice el
envío. Mantener los recibos históricos intactos.
