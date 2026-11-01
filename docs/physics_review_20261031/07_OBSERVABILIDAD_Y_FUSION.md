# Observabilidad de q y fusión: diagnóstico antes que otra confianza aprendida

## Corrección importante sobre Stage68

El argumento de estimar solo lo que importa es válido, pero Stage68 ya tenía una protección contra exigir todo el movimiento. En el commit de entrenamiento `3820f76e`, `fit_affine_normal_flow` ajusta seis coeficientes y cae a `_fit_isotropic`, que ajusta traslación más expansión. Su docstring señala que un contorno circular puede identificar expansión sin identificar rotación.

También existe una puerta anterior: con menos de 32 planos normales aceptados, retorna inválido sin intentar el ajuste afín ni el isotrópico. El informe registra mediana de cinco planos, 368 consultas con alguna ventana válida, 583 ventanas afines válidas y ninguna isotrópica. No se sabe, solo con esos agregados, qué fracción exacta falló por conteo, rango, condicionamiento o estimación de los planos. [Código](../../src/e_jepa_ttc/models/normal_flow_routing_v10.py), [resultado](../stage66_69_results.md).

Por tanto, «se rechazó porque se exigía una transformación completa» es una hipótesis incompleta. La primera prueba debe contar las causas a lo largo de la cascada. Bajar el umbral de 32 después de ver las métricas sería otra receta, no una corrección demostrada del experimento anterior.

## Cantidad estimable bajo un modelo lineal

Para y=Hβ+ε y q=cᵀβ, q es identificable en el modelo lineal sin ruido si c es ortogonal al núcleo de H, equivalente a c pertenecer al espacio fila. En el orden del código:

```text
β = [Axx, Axy, Ayx, Ayy, bx, by]
q_local = (Axx + Ayy)/2
c = [1/2, 0, 0, 1/2, 0, 0]
```

Una SVD permite medir la proyección de c sobre direcciones no observadas. Tras escalar columnas hay que transformar c de forma consistente y recuperar unidades. Usar un número de condición global de H no es lo mismo que medir sensibilidad de cᵀβ.

Ejemplo: si se observan β1 y β2 pero β3 es libre, el objetivo β1+β2 se identifica aunque el sistema no tenga rango completo. Si solo se observa β1+β2 y el objetivo es β1−β2, no se identifica. La pseudoinversa devuelve un número en ambos casos; ese número no certifica observabilidad.

En ruido, además de la proyección al núcleo, interesa la varianza sobre las direcciones observadas. Bajo un modelo lineal gaussiano con covarianza Σ válida, puede estudiarse cᵀ(HᵀΣ⁻¹H)^+c, **solo después de comprobar estimabilidad**. El complemento de Schur elimina nuisance bajo las condiciones del bloque correspondiente. Una regularización ridge produce un número estable incluso sin evidencia: se debe separar información aportada por el prior e información de las observaciones.

## La ecuación no corrige un modelo equivocado

En normal flow, H también se estima a partir de eventos ruidosos; no es necesariamente un diseño fijo sin error. Planos solapados comparten píxeles y timestamps. Multiplicar el número de planos no multiplica por igual la información independiente. IRLS y pesos dependientes de residuos tampoco producen automáticamente una covarianza gaussiana calibrada.

La traza afín representa expansión local bajo un modelo de proyección y movimiento. No siempre equivale a −Zdot/Z del objeto. Fondos, oclusión, giro y superficies con distinta profundidad pueden producir ajuste bien condicionado con sesgo. El experto de expansión del 31 ofrece un ejemplo concreto: con fondo estático al 50% aceptó q incorrecto en ambos fixtures. Un residuo pequeño o confianza alta no descarta error de modelo.

El antecedente CodedEvents usa información/cotas para diseñar óptica y tracking 3D. Apoya el uso de la herramienta matemática; no valida esta aplicación TTC ni una confianza lista para fusionar. [CodedEvents](https://arxiv.org/abs/2406.09409).

## Auditoría propuesta, sin entrenar

1. Recuperar el diagnóstico por ventana y los planos retenidos. Registrar rechazo por cantidad, rango espacial, calidad temporal, rango afín, rango reducido y condicionamiento. Conservar todos los denominadores.
2. Calcular estimabilidad y sensibilidad dirigidas a la traza para el modelo afín completo. Comparar con el reducido, que introduce isotropía como hipótesis adicional.
3. Añadir fixtures de expansión, traslación, apertura, orientaciones pobres, ruido correlacionado y mezcla objeto/fondo. Mostrar casos donde todo β no se identifica y q sí, y casos donde tampoco q se identifica.
4. Contrastar confianza con error y signo en desarrollo, por grabación y soporte. Comparar con recuento de eventos, condición global, residuo y confianza actual. Evitar una correlación pooled dominada por el dataset.
5. Mantener todas las consultas del sistema. Si eventos no aportan una medida utilizable, registrar el fallback y su error; no eliminarlas para mejorar la media.

La caché actual conserva planos y snapshots para diagnóstico en [event_geometry_cache_v10.py](../../src/e_jepa_ttc/data/event_geometry_cache_v10.py). Antes de prometer una auditoría sin lectura RAW hay que confirmar que el archivo original necesario está disponible y que conserva esos campos.

## Cómo podría influir en la fusión

Una variable útil describiría información incremental respecto a RGB, no solo calidad absoluta de eventos. Ambas ramas comparten ROI, escena y parte de la supervisión; sus errores pueden estar correlacionados. Sumar precisiones como si fueran sensores independientes puede aumentar confianza sin aumentar evidencia.

Una corrección anclada en RGB puede usar observabilidad para limitar o modular la contribución de eventos, con entrenamiento agrupado y ZERO/control de igual capacidad. Debe compararse contra los campos actuales de soporte: F2/Fcap y F2n/Fcapn ya fallaron su criterio completo. El objetivo nuevo sería ambigüedad de expansión o desajuste entre observaciones y predicción, no contar más eventos.

Un filtro bayesiano o gate no debe poder inflar libremente varianza para apagar las muestras difíciles. Hace falta score probabilístico propio, cobertura, anchura, error condicionado y resultado agregado con cobertura completa. Si el intervalo de q cruza cero, el correspondiente conjunto de TTC puede ser no acotado o separado en dos ramas; no es un intervalo gaussiano simétrico en segundos. Emitir ±60 s no resuelve esa incertidumbre física.

## Dictamen

Prioridad alta para diagnóstico, coste menor que entrenar otra representación. Prioridad de integración solo si la nueva señal supera a las variables de calidad existentes y predice fallos bajo cambios de dominio. El objetivo no es afirmar «el sistema sabe cuándo no puede medir» hasta verificar ese comportamiento.
