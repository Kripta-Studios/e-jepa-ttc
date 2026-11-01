# Simetrías, coordenadas y superficies materiales

**Actualización posterior al corte base:** el screen149 terminó durante esta revisión; F0 pasa y el router no. La ampliación de F0 está registrada en curso y la conservación de historias eAP sigue pendiente. Los pasajes que describen el screen como pendiente corresponden al corte inicial. Véase [actualización de cierre](13_ACTUALIZACION_CONCURRENTE.md).

## Reescalado temporal: la prueba más inmediata

Si t'=t0+α(t−t0), entonces Δ'=αΔ, T'=αT y q'=q/α para la trayectoria reparametrizada. El piloto local usa un factor de **velocidad** a=1/α y evalúa a·T'/T≈1. No hay contradicción de fórmulas: cambia la convención del factor.

Hay dos protocolos distintos. En el warp de observaciones preparadas se conservan imágenes y conjuntos de eventos y se cambian relojes/canales dependientes de duración. En el replay de un stream retemporizado con ventanas de duración fija se vuelven a seleccionar eventos y frames; puede requerir historia original más larga. Se deben medir ambos sin llamar al primero simulación completa del segundo.

La fase de salida permanece definida con d0=100 ms. No debe cambiarse al delta de cada pareja, ni dividir el TTC final sin regenerar PAIR, fases, diferencias, tiempos, contextos y fusión. Los canales de tasa también cambian; los de tiempo normalizado pueden permanecer iguales si representan posición relativa dentro de la misma ventana transformada.

El [piloto existente](../research_log_20261031_producer_timewarp/README.md) ya muestra el desvío de PAIR. La [intervención canónica](../research_log_20261031_canonical_pair/README.md) mejora los cuatro contrastes de F0, pero empeora el contexto de eventos de carretera a=0,5: cociente 0,673790→1,891432. Eso impide declarar que corregir un bloque garantiza corregir el sistema.

El [screen149](../superpowers/plans/2026-10-31-phase-canonical-screen149.md) está fijado y exige no perder cobertura ni empeorar MAE/RAE macro en carretera o Slider para cada ruta. Es la primera prueba pendiente razonable; no procede diseñar otra selección a partir de las dos consultas favorables. La auditoría eAP de 88.744 entradas actuales no reconstruyó todas las historias, por lo que no certifica conservación del modelo completo.

## Escala espacial y recorte

Un escalado fijo y coherente de coordenadas preserva el TTC ideal. En una cámara discreta, downsampling puede perder bordes e información; por tanto se espera invariancia dentro del régimen observable, no igualdad mágica a cualquier resolución. Hay que transformar intrínsecas, ROI, offsets y distorsión de forma consistente, y registrar cuándo el aliasing impide la comparación.

La prueba factorial propuesta es útil: mismo TTC con distintos tamaños y texturas; mismo tamaño con distintas tasas de aproximación. Añadir mismo event rate con q diferente y q igual con event rate diferente impide que densidad haga de sustituto de física.

Para h_crop=s(t)h_sensor, la derivada de log-altura contiene q más sdot/s bajo las hipótesis de proyección. Compartir ROI entre endpoints elimina parte de la distorsión, pero no garantiza una escala coherente entre todas las ventanas históricas. Un estado persistente necesita transformar coordenadas entre consultas o conservar el mapa de crop a sensor. No basta comparar tensores redimensionados que parecen representar el mismo objeto.

Una prueba con zoom conocido del crop debe recuperar el mismo estado en coordenadas de sensor. Este control no añade conocimiento físico nuevo al sensor: verifica que el software no lo destruye. La inspiración dimensional tiene antecedentes, pero su valor aquí está en el contrato de unidades y transformaciones. [Buckingham Pi y aprendizaje dimensional](https://www.nature.com/articles/s43588-022-00355-5).

## Superficies materiales y composición

Para una correspondencia diferenciable ψ_ab con Jacobiano de determinante positivo, r_ab=(1/2)log det J_ab mide cambio logarítmico de escala de área. Por regla de la cadena:

```text
J_ac(x) = J_bc(ψ_ab(x)) J_ab(x)
r_ac(x) = r_ab(x) + r_bc(ψ_ab(x))
```

Esta identidad necesita composición en las posiciones transportadas, visibilidad compatible y mapas válidos. Si se define ψ_ac como la composición de los otros dos, comprobar la identidad es tautológico. El contraste informativo compara una correspondencia a→c estimada de manera independiente con la composición a→b→c.

Bajo expansión isotrópica y el mismo modelo de proyección, ese r puede entrar en q_b=expm1(r)/Δ. Giro 3D, deformación, inclinación o profundidad desigual invalidan la equivalencia general. Una silueta visible puede cambiar sin que la superficie se contraiga. La relación entre expansión óptica y cambio de profundidad ya aparece en [Yang y Ramanan](https://gengshan-y.github.io/expansion/expansion.pdf).

PHASE ya usa correlaciones locales, divergencia, entropía y ciclos forward–reverse. La nueva hipótesis debe añadir una trayectoria de tres instantes y tratamiento explícito de visibilidad, con control frente al transporte existente. Predicciones de visibilidad libres pueden ocultar todos los puntos difíciles; hacen falta cobertura mínima y controles sobre sus máscaras.

El trabajo [From Contrast to Consistency](https://arxiv.org/abs/2605.25570) propone continuidad/estructura y arquitectura bidireccional. Sirve de antecedente, no de componente causal listo para usar. Debe verificarse la frontera temporal de todos los inputs; el profesor de entrenamiento puede mirar objetivos futuros, el estimador online no.

## Restricciones que descartaría

Conservar área borraría la expansión que interesa medir. Conservar masa de eventos trataría cruces de umbral como partículas materiales. Maximizar contraste sin protección puede inducir colapso del campo, un problema ya estudiado y también observado en el solver local. [Regularización contra event collapse](https://arxiv.org/abs/2212.07350).

Tampoco usaría I∝1/Z² como ley general de brillo para una superficie extendida resuelta. La radiancia se conserva a lo largo de un rayo en espacio vacío bajo las condiciones correspondientes; irradiancia y flujo recogido tienen otras dependencias geométricas. [Fundamentos radiométricos](https://people.csail.mit.edu/fredo/comp-photo-book/02-fundamentals-01-light-and-physics.html).

La extensión material merece prioridad después del diagnóstico de tiempo y observabilidad, si la inspección muestra que oclusión, fondo o asociación explican errores. No requiere empezar por una reconstrucción 3D completa.
