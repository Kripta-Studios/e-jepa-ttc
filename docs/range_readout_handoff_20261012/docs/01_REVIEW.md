# Revisión independiente del cierre V14

## Lo comprobado
Los seis informes de GitHub coinciden con el ZIP después de normalizar sólo
CRLF a LF, conforme a sus identidades Git. La estructura y los 771 hashes internos
del ZIP se verificaron sin modificar sus originales. Las 6 × 16.432 predicciones recuperadas en
float32 permiten recalcular las cuatro métricas macro por grupo y MAE con
máxima discrepancia 2,85e-14. No se ejecutaron sus modelos: faltan inputs H/V.
El recuento 772 corresponde a miembros; 771 a archivos cubiertos por el manifiesto.
No es una nueva ejecución de los 41 tests y 17 subtests históricos.

## Tabla principal
| Receta | MiDc | MiDs | MiDl | MiDn | Overall | MAE s |
|---|---:|---:|---:|---:|---:|---:|
| M0/L0 |156.832|54.776|52.217|61.382|106.208|2.512|
| M0/L1 |149.592|51.168|56.973|61.009|101.945|2.029|
| M1/L0 |134.470|63.112|76.397|84.208|102.229|2.979|
| M1/L1 |136.187|65.151|83.999|80.041|104.043|2.341|
| M2/L0 |134.871|67.812|77.768|87.619|104.318|3.369|
| M2/L1 |134.591|70.530|83.826|79.928|104.830|2.379|

NO_CANDIDATE es correcto según los gates originales. M0/L1 es una mejora de algunas
métricas, no el ganador registrado. M1/L0 reduce crucial 14,26%, pero empeora largo
46,31% y negativo 37,19%. Las masas M1/M2 trasladan crucial de ~10% a 50%, negativo
~40% a 10% y largo ~21% a 10%. El cambio de compromiso es consistente con la intervención,
no una demostración de que mejorar todos los rangos sea imposible.

M0/L1 reduce MAE ~19,23% y overall ~4,01%, pero sólo mejora MiDc 4,62%, empeora MiDl
9,11% y pasa de 4/76 a 0/76 avisos <=1s. M2/L1 mejora a 14/76, todavía muy insuficiente.
Las tasas de falsa alerta pequeñas no convierten ninguno en un AEB validado.
No hay evaluaciones FCWD nuevas de estos seis endpoints. V14 V no es FCWD ni Dev32.

## Hallazgo añadido 1: optimización efectiva
Desde CURVE.csv: fracción de updates con norma pre-clipping >1:
M0/L0 16,3%; M1/L0 37,2%; M2/L0 58,0%. Últimos 250: 11,2%, 42,4%, 62,4%.
Con 52/18.422 filas ≤1 s y draws uniformes, la probabilidad de un batch128 sin
ninguna de esas filas es aproximadamente 69,64 %. Los brazos de mayor peso registran
más clipping global; eso no identifica qué fila provoca cada recorte.
Esto es un mecanismo plausible que merece separar del objetivo esperado. NO prueba
que exista conflicto de gradientes entre bandas; faltan sus productos escalares.
No comparar las magnitudes de loss entre M0/M1/M2 como si fueran el mismo objetivo.

## Hallazgo añadido 2: rango condicional del productor
El archivo configs/rgb_port/producers.json fija C=0.05 y mezcla temporal 0.75.
En A5/C2F el residual es C*tanh(s), añadido al log-ratio analítico. En cuatro
controles H <=1s el residual actual ocupa en media 90,9% / 93,3% de ese rango.
Manteniendo fijas sus alturas y permitiendo de manera optimista residual +C
independiente en las dos parejas, el TTC mínimo positivo sigue por encima del GT
para los ocho pares consulta/productor. En H 1–2s sucede en 2/4 A5 y 3/4 C2F.

Esta cota utiliza el transporte nativo de inverse TTC de la pareja anterior al
anchor actual. No supone expansión constante. Es condicional a alturas/tiempos
congelados y C=0.05; NO limita PAIR ni H8, que pueden producir otras respuestas;
tampoco limita un encoder reentrenado que cambie las alturas. La muestra es de
32 controles H elegidos en V14, no un estimador de prevalencia en todo P/H/V/FCWD.

Ejemplo físico, no dato experimental: para delta=0.1 y TTC actual=0.8,
r=log1p(0.1/0.8)=0.1173. Si la geometría da r=0.006, sumar como máximo0.05
no puede alcanzar ese valor en la pareja actual. El blend requiere considerar
ambas parejas, por eso el script calcula su relajación optimista por consulta.

La explicación que investigo ahora no es «corregir con una constante»:
es comprobar si el readout satura parte de la señal antes de que la cabeza la reciba.
Cambiar C en un checkpoint sin entrenar alteraría todas sus salidas: no sería una
reparación equivalente ni una demostración de solución.

## Lo que queda sin demostrar
No hay prueba de que PHASE17 haya agotado su información, ni que el cap sea la causa
única, ni que RGB resuelva la transferencia. La geometría casi constante podría
proceder de entrenamiento, crop, foreground o falta de señal. Los tests estructurales
no distinguen esas causas. V13 intacto no implica que su RGB haya finalizado; leer
recibos vivos. Evitar otra ronda de losses sin hipótesis y también evitar afirmar
que una CNN/Transformer mayor ganará sólo por su tamaño.

## Fuentes
- Los seis informes en docs/range_transfer_results_20261012, commit1c25178.
- CURVE.csv y PREDICTIONS.csv incluidos en el ZIP de V14.
- matched_H_raw_adapted y H_DIAGNOSTIC_CONTROLS.csv.
- configs/rgb_port/producers.json y models/causal_scale_ttc.py en el commit citado.
Las tablas independientes y el script están en evidence/ y scripts/.

## Referencias directas del snapshot
- Índice: https://github.com/Kripta-Studios/e-jepa-ttc/blob/1c251781ac845a5d661523e607e0f7c4a8d3af8a/docs/range_transfer_results_20261012/README.md
- Método: mismo directorio, 01_EJECUCION_Y_METODO.md.
- Diagnóstico: mismo directorio, 02_DIAGNOSTICOS.md.
- Resultados y gates: mismo directorio, 03_RESULTADOS_Y_DECISION.md.
- QA/coste: mismo directorio, 04_PRUEBAS_COSTE_Y_CONTABILIDAD.md.
- Reproducción: mismo directorio, 05_REPRODUCCION_Y_ENTREGA.md.
- Configuración del residual: https://github.com/Kripta-Studios/e-jepa-ttc/blob/1c251781ac845a5d661523e607e0f7c4a8d3af8a/configs/rgb_port/producers.json
Las tablas del paquete son recálculos propios, no nuevas predicciones de sensores.
