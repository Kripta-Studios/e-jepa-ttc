# Cabeza de componentes nativas de velocidad

La extensión del piloto añade una MLP de dos salidas a cada capa del decoder
3D. Se supervisa con las dos primeras componentes publicadas por eAP, mediante
los mismos emparejamientos de cajas que usa el detector. Es una cabeza aprendida;
el exportador no rellena velocidades constantes cuando esa cabeza está ausente.

Este contrato no resuelve si las componentes publicadas incorporan compensación
de ego-motion ni convierte velocidad en TTC. La rama PHASE sigue estimando TTC
a partir de observaciones RGB/eventos. Tampoco se interpreta la velocidad de un
objeto como pose o velocidad del vehículo propio.

## Supervisión y salida

- `prepare_sample(..., include_velocity=True)` produce `velocity[N,2]` y una
  máscara booleana `velocity_valid[N]`. Las anotaciones de cajas siguen el mismo
  filtro geométrico que el perfil RGB.
- Una etiqueta ausente o inválida no se convierte en velocidad cero válida. Se
  conserva la caja, se enmascara esa pérdida y se cuentan los casos en el recibo.
- El modelo produce `pred_velocity[B,Q,2]`; el decoder emite `velocity_mps` solo
  si ese tensor existe y es finito. La entrada de inferencia no contiene etiquetas.
- La pérdida suma los errores absolutos de ambas componentes y divide por el
  número de objetos emparejados con velocidad válida. Incluye pérdidas auxiliares
  de las capas 3D; el coeficiente se fija en la configuración de cada run.

Esta pérdida mantiene gradientes en las predicciones válidas. Cuando no hay
ninguna, devuelve cero diferenciable, sin leer los valores inválidos como targets.
No se añade velocidad al coste del matcher, para que el emparejamiento conserve
el contrato de detección que se está evaluando.

## Diagnóstico separado

[`evaluate_velocity_labels`](../../src/e_jepa_ttc/detection/velocity_diagnostics.py)
empareja por confianza, clase y distancia XY estrictamente menor de dos metros.
Los emparejamientos no se escogen por su error de velocidad. Si una caja GT
carece de velocidad, sigue consumiendo su emparejamiento de caja y se excluye
únicamente de este diagnóstico.

Se registran MAE por componente, suma de MAE de componentes y error vectorial L2,
con número de pares válidos y desglose por clase. Son errores contra las etiquetas
nativas, expresadas en m/s por el dataset; **no son un AVE oficial ni una medida
de error TTC**. La ausencia de emparejamientos produce valores `null`, no error cero.

## Alcance del experimento

Los sobreajustes v1/v2 anteriores conservan sus checkpoints y fuentes originales,
sin esta cabeza. El piloto que la active utiliza una configuración y un snapshot
propios, y registra explícitamente su inicialización ImageNet y el coeficiente de
velocidad. No se atribuye causalmente una diferencia de métricas a esta cabeza:
también cambian población, presupuesto e inicialización respecto al sobreajuste.

Una imagen aislada no determina físicamente toda velocidad de objeto. La cabeza
puede aprender regularidades del entrenamiento; demostrar velocidad robusta
requiere evaluación adicional, contexto temporal y aclarar la semántica de las
etiquetas. Los resultados medidos del piloto se deben interpretar con ese límite.
