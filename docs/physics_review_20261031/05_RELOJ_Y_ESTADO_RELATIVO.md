# Reloj físico y estado relativo: correcto bajo hipótesis, útil por demostrar

## Derivación y dominio

Sea Z(t)>0 una profundidad respecto al sistema de cámara y q=−Zdot/Z. Al derivar:

```text
qdot = q² − Zddot/Z
T = 1/q, si q ≠ 0
Tdot = −1 + Z·Zddot/Zdot², si Zdot ≠ 0
```

Si Zddot=0, q(t+Δ)=q(t)/(1−Δq(t)) y T(t+Δ)=T(t)−Δ. La fórmula admite q negativo: recesión da TTC firmado negativo. No se debe imponer positividad al estado del benchmark. Para aproximación hay que mantener 1−Δq>0; atravesar contacto o acercarse a la singularidad invalida ese rollout.

La igualdad vale para velocidad axial constante **en la coordenada de profundidad elegida**. Velocidad constante en el mundo no basta si gira la cámara o cambia el punto de superficie representado. También puede variar el punto que determina la profundidad mínima del target. Frenada, rotación y cambios de visibilidad deben aparecer en pruebas de violación de hipótesis.

La composición FΔ₂(FΔ₁(q))=FΔ₁₊Δ₂(q) se cumple por álgebra en ese dominio. Penalizarla dentro de una capa que implementa exactamente F añade cero información. Lo informativo es comparar el estado propagado con una medición o embedding futuro independiente, y comparar contra persistencia y dinámica libre de igual capacidad.

El vínculo clásico entre TTC, escala proyectada y cocientes sin escala métrica está documentado por [Horn, Fang y Masaki](https://people.csail.mit.edu/bkph/articles/Time_To_Contact.pdf). La derivación anterior es cinemática, no evidencia experimental nueva de PHASE.

## Tres coordenadas que no se deben confundir

Para h=k/Z, con k fijo, y r=log(h_b/h_a):

```text
q_b = expm1(r)/Δ
T_b = Δ/expm1(r)
T_a = Δ/(1−exp(−r)) = T_b + Δ
φ(T) = −log(1−d0/T), d0=0,1 s
q = −expm1(−φ)/d0
```

r describe el cambio entre endpoints; q es tasa en el endpoint actual; φ es la coordenada canónica del scorer. La cadencia real Δ y d0 no son la misma variable. Con h_b/h_a=1,1 y Δ=0,1 s, T_b=1 s y T_a=1,1 s. No se puede corregir una comparación restando Δ sin verificar el target y la semántica de las alturas aprendidas.

La emisión desplegada recorta fase y fuerza magnitud finita ≤60 s; cero fase se emite como +60 s. Esa salida no sirve como estado físico para aprender q=0. Conviene conservar q latente sin el suelo de 1/60 y aplicar la política finita solo al emitir el resultado. Los logs deben distinguir estado, salida, soporte y saturación.

## Qué aporta rho y qué no

Con rho=1/h=Z/k, la dinámica ideal es rho(t+Δ)=rho(t)+Δ·v_rho, con v_rho constante, y T=−rho/v_rho. Una transformación rho→c·rho, v_rho→c·v_rho con c>0 constante no altera TTC. Si c cambia por instante, q cambia en −d(log c)/dt. Por eso una normalización independiente por frame puede inventar o eliminar aproximación.

El latente rho no es identificable por llamarlo «distancia relativa». Se debe anclar a ratios/correspondencias observados y fijar su convención de escala por trayectoria. El estado [rho,v_rho] es suficiente para este modelo axial ideal, pero no para explicar todos los eventos: también intervienen traslación lateral, textura, visibilidad e iluminación. «Estado mínimo para TTC» y «estado suficiente para generar el sensor» son dos requisitos distintos.

No propongo sustituir el backbone por un vector de dos valores. Un piloto puede mantener z visual y reservar un subestado físico [rho,v_rho] o [q] cuya dinámica y uso sean comprobables. Si ese subestado no modifica las predicciones observables al intervenirlo, su buena sonda no demuestra que la red lo utilice.

## Aceleración y cierre del estado

Introducir a_rel=Zddot/Z permite escribir qdot=q²−a_rel. Si se aprende a_rel sin restricciones, puede absorber cualquier error de q y hacer vacía la loss. Si se fuerza a_rel constante durante mucho tiempo, se introduce otra hipótesis falsa. Incluso con aceleración axial métrica constante, a_rel cambia con Z.

En el caso especial de aceleración axial constante, un estado ampliado [q,a_rel] satisface a_rel_dot=q·a_rel. También puede usarse [rho,v_rho,a_rho] con evolución polinómica. Son modelos de régimen y no leyes universales del tráfico. Un corrector de proceso acotado, acompañado de innovación sensorial y señal de desajuste, es más defendible que pedir al modelo elegir retrospectivamente un régimen con acceso al futuro.

Las aceleraciones futuras no observadas no son identificables desde un prefijo idéntico. El sistema debe expresar esa incertidumbre, no prometer anticipar una frenada arbitraria. Un objetivo de TTC instantáneo se conserva aunque el predictor de trayectoria falle por una acción futura imprevisible.

## Experimento recomendado

Primero evaluar la ley sobre trayectorias sintéticas y datos de desarrollo, con deltas reales, objeto persistente y target definido en el mismo punto. Separar el error de medición del error de modelo dinámico. No escoger las trayectorias «casi constantes» con las etiquetas del conjunto de evaluación.

Después comparar, con mismo encoder, historia, observaciones, cabezas y updates: supervisión actual; predictor temporal libre; estado de velocidad constante anclado; estado con corrección de proceso restringida. Un control con observaciones futuras permutadas debe conservar la marginal y no la correspondencia física. Añadir la comparación contra persistencia y un filtro causal sencillo, midiendo retraso en frenadas.

Retirar los auxiliares al evaluar el encoder congelado separa beneficio de representación de beneficio del predictor extra. Conservar el predictor puede ser útil como sistema, pero exige otro coste y otra afirmación. No se necesita ejecutar ambas líneas a gran escala para detectar primero si q es recuperable y usado.

## Relación con el código existente

[normal_flow_routing_v10.py](../../src/e_jepa_ttc/models/normal_flow_routing_v10.py) ya tiene `inverse_ttc_at_anchor`, con advertencia sobre aceleración y singularidad, desde `3820f76e`. [controls.py](../../src/e_jepa_ttc/simplex_t/controls.py) implementa EWMA de fase. Ninguno demuestra que un encoder aprendió un estado dinámico útil. La novedad que merece probarse sería **anclar y contrastar temporalmente un estado aprendido**, después de corregir el tratamiento de unidades y con un baseline dinámico explícito.
