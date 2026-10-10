# 3. Arquitectura propuesta y qué se entrena

Nombre de trabajo: DUAL-TTC. No es un claim de novedad de cada operador o de SOTA.

## B. Puente RGB sobre el sistema actual

Conservar A5/C2F/PAIR y una copia histórica H8 seed7 sin gradientes. Extraer el estado
final de GRU160 mediante helper que reproduzca exactamente el forward original; no
cambiar la clase congelada ni el esquema PHASE17. La pareja RGB pasa por un mismo
ResNet50 genérico ImageNet, inicialmente congelado, y un readout de diferencias/pareja.

`phi_fused = phi_H8 + 0.03 * gate(zE,zR,quality) * residual(zE,zR,dt)`.

La última capa residual comienza en cero. La compuerta comienza abierta parcialmente
(sigmoid(0)=0,5), **no cero simultáneamente a residual cero**: así el residual recibe
un gradiente desde la primera actualización. RGB inválido omite el encoder y devuelve
exactamente el TTC H8 almacenado, no una reconversión innecesaria.

B_TRUE: pareja RGB real. B_REPEAT: ambas posiciones contienen el frame actual con su
identidad/tiempo real; control de evidencia pasada RGB. B_ZERO: bloque RGB anulado tras
codificar, con el mismo readout y pesos de capacidad; prueba recalibración del lado E.
Estas cabezas sólo se entrenan en TRAIN40; son exploratorias. El backbone RGB frozen
es una primera prueba, no el modelo RGB definitivo. El core D sí entrena encoders.

## D. Predictivo event-only nuevo, no otro selector

Backbone compartido entre endpoints: ResNet50 con inicialización ImageNet1K_V2.
La capa RGB3→20 de eventos se inicializa repitiendo la media de los tres kernels y
multiplicando por 3/20. Esto conserva la respuesta a canales idénticos, no pretende
que eventos e intensidades tengan la misma distribución. No descargar pesos eAP Garl
para inicializar un modelo que se evalúe en el split D.

Se conserva BatchNorm en modo evaluación con estadísticas genéricas fijas y affine
entrenable; convs y FPN entrenables. Es una elección explícita por microbatches locales,
no equivalencia al entrenamiento BatchNorm batch128 de Garl. Cambiarla sería otro brazo.

De C2 (32×32), C3 (16×16) y C4 (8×8) se construye FPN de 128 canales a 32×32. C5 aporta descriptor
global de 128. Para cada par consecutivo se mantienen los dos mapas y su diferencia;
la rama CORR añade similitudes coseno locales para radio4 (81 candidatos), máscaras de
borde y las dos direcciones. No reducir primero a un desplazamiento medio global.

Procesador compartido por par:
`[Fprev, Fcurr, Fcurr-Fprev, cost_forward, mask_forward, cost_reverse, mask_reverse]`
→ proyección 1×1 → dos bloques residuales espaciales → pooling media/máximo
→ token 160, unido a descriptores globales y cuatro tiempos observados.

Secuencia de tokens → dos GRUCell160 → cabezas de fase y cuantiles. P2 tiene un único
par; CONTEXT4 tiene tres. El estado se reinicia por consulta; no se declara tracking.
No usar todos los pares de todas las posiciones ni materializar un volumen 4D full-field.

EV_DIRECT_P2 procesa la misma familia de mapas y reserva iguales bloques, pero sustituye
las similitudes por ceros preservando masks/shapes; es el control de información de
correspondencia. EV_CORR_P2 las conserva. EV_CORR_T4 añade alcance temporal.
La referencia calcula correlación también en DIRECT antes de anularla para igualar
forma/cómputo en la ablación. Su despliegue puede omitirla si se selecciona DIRECT.

Los outputs no dependen de A5/C2F/PAIR. Se permite cualquier fase dentro del contrato;
no existe suma geométrica acotada ±0,05 ni obligación de pasar por altura.
No se necesita reconstruir flow exacto o superar el gate de 32 planos de Stage68.

## R. RGB real para el candidato seleccionado

Un segundo encoder ResNet50/FPN recibe las dos RGB. No comparte pesos de BatchNorm,
normalizadores o stem con eventos. Aprende un descriptor de pareja con el mismo módulo
de correspondencia si el encoder RGB tiene evidencia; la unión E/R es tardía, después
de la comparación temporal de cada sensor. No interpreta coordenadas iguales de
cámaras distintas como correspondencia métrica conocida.

Se inicializa desde el evento seleccionado, se congela ese E en la fase R y se entrenan
el encoder RGB, su procesador de par y la cabeza de fusión. El E congelado conserva su
rendimiento y constituye un fallback exacto. R_TRUE y R_REPEAT tienen arquitectura,
optimización y capacidad idénticas; repiten o no el frame actual. La pérdida auxiliar
RGB permite aprender una salida RGB desde el descriptor, sin depender sólo del gate.

El candidato fused es `phi_E + gate * residual` con residual inicial cero. Este gate
no es una probabilidad calibrada de que haya colisión. El candidato E es un artefacto
separado; una red fusionada a la que faltan imágenes no sustituye su evaluación.

Durante entrenamiento, dropout de la modalidad RGB del 25% de consultas, independiente
del target. Las filas sin RGB siguen en la pérdida principal y fallback; contribución
a la pérdida auxiliar RGB sólo donde la pareja existe. No entrenar event-only leyendo
RGB para esconder decisiones de crop o quality. Máscaras, disponibilidad y fallos
salen en el informe.

## Fase, inicialización y pérdidas

Conservar el conversor de `simplex_t/phase.py` y registrar ambas cantidades:
`raw_location` y fase del TTC realmente emitido (cap ±60). Métricas sobre emisión real.

Para un predictor directo, NO inicializar todo a fase cero: cero emite +60 y el clamp
tiene una región plana. Inicializar el bias/prior con la mediana ponderada de fases
**del TRAIN del fit**, que debe estar fuera de esa región. Las muestras de preparación
no eligen el prior por targets de desarrollo. Rechazar un prior no válido en admisión.

La pérdida D común es la L1 en fase emitida, escala0,03, más 0,1 de pinball q10/q90.
Sin cabeza auxiliar de costes de expertos para D. La receta conserva masas globales
por secuencia/bucket y no renormaliza en cada minibatch. Registrar capping/saturación
y gradientes; un collapse es resultado o fallo diagnosticado, no permiso para barrer
otras pérdidas. La selección de esta receta está hecha antes de correr D.

En R: `L_fused + 0.25*L_rgb_aux`; E permanece congelado. Cuantiles de fase no equivalen
a intervalos calibrados de segundos ni probabilidades AEB. La calibración posterior,
si se autoriza, exige su partición propia y no se ajusta en EvTTC.

## KD opcional: mejorar E usando RGB sólo durante TRAIN

Si R_TRUE aporta ≥3% frente al E emparejado en D con guardrails, se abre un experimento
TRAIN de distilación de **predicción de fase**. Teacher: R_TRUE de ese mismo split/seed,
congelado. Student: copia del E correspondiente, todas sus capas entrenables.

`L_student = L_supervised + 0.2*abs(phi_student - stopgrad(phi_teacher))/0.03`.
Comparador E_EXTRA: misma copia E, mismos 10.000 updates, sin término KD. Cachear sólo
predicciones del teacher sobre TRAIN; no usar outputs de teacher que haya visto DEV.
No obligar a coordenadas latentes de encoders independientes a coincidir.

Una mejora KD se describe como **event-only en inferencia, entrenado con RGB
privilegiado**, no como entrenamiento E puro. E_EXTRA es imprescindible para separar
el beneficio de KD del entrenamiento adicional. Ninguno hereda superioridad de R.

## Pesos de inicialización

El código de referencia usa weights=None sólo para tests sintéticos sin red. Para
D/R de producción se exige cargar el checkpoint genérico ImageNet1K_V2 de ResNet50,
registrar su SHA completo y verificar state_dict. Se admite descargar sólo ese recurso
público si no está local, no cientos de GB o servicios de pago sin necesidad. Un fallo
de descarga no autoriza fingir pretraining: pausar esa dependencia o entregar el bloqueo.
B usa el descriptor avgpool de2048D del mismo ResNet50 genérico congelado, sin FPN;
D/R usan la proyección128 y FPN entrenables. No mezclar esos formatos por coincidencia
de nombres. Los 20canales event nativos se conservan según native_feature; no se
presupone que sean diez bins por cada polaridad.

## Augmentación fijada

D: flip horizontal con probabilidad0,5, común a todos los endpoints del ejemplo;
actualizar metadatos de crop si se guardan. Sin escalado independiente entre tiempos: 
eso fabricaría expansión. Normalización y orden canónicos fuera de augmentación.
R: no flip adicional de eventos; RGB gain común a las dos imágenes U(0,8,1,2), aplicado
al floatRGB antes de ImageNet; no alteración independiente de escala/rotación.
B y calibradores: sin augmentación de features cacheadas. KD/EXTRA: sin augmentación,
mismos inputs exactos que sus targets de teacher cacheados; no mezclar versiones.
Todas las elecciones son prospectivas. No añadir augmentaciones después de ver DEV.

En la referencia, HeadOnlyRGBBridge acepta quality=None para tests sintéticos; el
adaptador real debe aportar los cuatro observables definidos en el contrato. No usar
cuatro edades bajo el nombre de calidad si el checkpoint aprendió otros valores.
