# C-VIS: coordenadas, superficies y visibilidad

## Qué puede mejorar del sistema actual

R0 ya utiliza un recorte común entre endpoints. La idea útil no es volver a introducirlo: es comprobar coherencia espacial de **todo el historial** y distinguir cambio de visibilidad de cambio de escala. Una caja que pierde una parte del coche por oclusión puede inducir contracción sin que cambie su aproximación.

La intervención puede actuar en preparación, en calidad de la medición o en transporte. Solo si esos diagnósticos muestran un límite de correspondencias se justifica aprender otro bloque.

## V0: prueba de coordenadas con pesos congelados

Para `x_crop=s(t)*(x_sensor−o(t))`, la derivada de escala incluye `sdot/s`. En particular, si `h_crop=s(t)*h_sensor`, entonces `d log h_crop/dt = d log h_sensor/dt + sdot/s`. El movimiento inducido por recorte debe retirarse o representarse explícitamente al formar la medición física.

Registrar por snapshot escala, origen y cámara. Comprobar que normalización de coordenadas, resize, intrínsecos y cualquier flujo se transforman juntos. Un flujo expresado en píxeles del crop y un centro expresado en sensor no forman una ecuación de expansión válida.

Ensayos: recorte fijo; traslación conocida del recorte; zoom suave conocido; salto de localización; cambio de resolución sin truncamiento; mismo objeto con margen de fondo diferente. Mantener el target físico y distinguir cambio de coordenadas de pérdida de información por recorte. Reportar el resultado de la cadena completa y de cada productor.

## V1: medir expansión de superficies persistentes

Tomar parches que correspondan a la misma superficie visible entre a, b y c. Para un transporte ψ y Jacobiano J que preserve orientación:

```text
r_ab(x) = 0.5 * log(det J_ab(x))
r_ac(x) = r_ab(x) + r_bc(ψ_ab(x))
```

El cambio de área local tiene antecedentes en [Yang y Ramanan](https://gengshan-y.github.io/expansion/expansion.pdf). La composición sirve como contraste únicamente si la ruta directa a→c se estima de forma independiente; construirla multiplicando las otras dos y penalizar la igualdad da una identidad sin nueva evidencia.

La relación con q exige hipótesis de proyección, profundidad y movimiento. Un giro 3D puede cambiar área aunque no cambie Z como supone el modelo. Etiquetar esos casos como violación del modelo local y estudiar incertidumbre, no entrenar una equivalencia universal entre logdet y TTC del objeto.

## Comparación sobre el release

| Brazo | Uso de la nueva evidencia | Coste inicial |
|---|---|---|
| V0 | Auditoría de transformaciones, sin cambios aprendidos | Re-encoding/replay |
| V1 | Calidad/visibilidad alimenta el diagnóstico C-OBS | Correspondencias locales y estadísticas |
| V2 | Medida de escala por parches como experto adicional | Extracción y reentrenamiento superior controlado |
| V3 | Auxiliar de correspondencia, área y visibilidad en un productor | Entrenamiento de productor y descendientes |

Empezar por V0/V1. No introducir simultáneamente nuevo flow, máscara y fusión: se perdería la atribución. Para V2, comparar medida sola, medida+visibilidad y control de igual coste/capacidad. Conservar el fallback del release y todas las consultas.

## Oclusión, fondo y degeneraciones

El solver local reciente aceptó q incorrecto con fondo estático al 50% en fixtures. Por tanto, buena condición numérica no garantiza que el movimiento pertenezca al objeto. Usar separación objeto/fondo, acuerdo entre parches y visibilidad con denominadores registrados; no elegir las regiones cuyo error TTC es menor usando targets.

La máscara de visibilidad debe tener un control de cobertura y supervisión/criterio independiente. Rechazar todo lo difícil reduce error condicionado y puede empeorar el sistema completo. Reportar cuánto cae en fallback y su error, no solo las regiones aceptadas.

La maximización de contraste admite colapso de eventos. No imponer conservación de área para evitarlo, porque eliminaría expansión real. Controlar deformaciones degeneradas dentro del modelo admitido y comprobar fixtures de aproximación/recesión. [Antecedente de colapso](https://arxiv.org/abs/2212.07350).

## Éxito y siguiente decisión

V0 se admite si arregla un contrato espacial comprobado y conserva o mejora precisión. V1 se mantiene si su calidad explica errores mejor que soporte existente. V2/V3 requieren mejora de TTC bajo oclusión/fondo y conservación del dominio nativo, no solo menor error de correspondencia.

Si el error dominante sigue siendo cambio del punto más cercano del objeto frente a la superficie seguida, separar ambas magnitudes. Mantener el target oficial y usar la superficie como medición auxiliar; no redefinir TTC para favorecer el nuevo módulo.
