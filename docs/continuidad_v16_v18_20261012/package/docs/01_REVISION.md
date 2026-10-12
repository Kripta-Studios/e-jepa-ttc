# Revisión de resultados V16

## Alcance comprobado

ZIP exterior: 503.683.384 bytes, SHA-256
`07ff9ea7491099c74e357df8efab4ae4ff57dca383f9e6e6d8b126d03e5a4afb`.
1.988 miembros y 1.987 entradas verificadas del manifiesto.
ZIP interior: 461.398.299 bytes, SHA-256
`a59bac60a1a97c4c5e0b279247c98b96b1b4ce1ff3dc8db7781b4cf01372bab9`.
47.843 miembros y 47.842 entradas verificadas. Todos los CRC y hashes coinciden.
No sumar estos archivos como observaciones independientes: el exterior vuelve
a publicar algunos documentos del interior.

El blob Git del informe extraído es
`a1e08a3ec47603ac34b9cae5b30badc64b3cb136`, igual al consultado en GitHub.
La rutina publicada de métricas se volvió a ejecutar, redirigiendo su salida:
46.778 comprobaciones escalares, 13 tablas V, 34 externas y 32 checkpoints
verificados por sus recibos y hashes. Eso no equivale a reentrenar esos fits.

Se ejecutaron además las seis cabezas PHASE_ONLY EVENT/ZERO seeds 7/13/23
sobre 16.432 inputs V guardados cada una: 98.592 salidas. La diferencia máxima
respecto a los TTC FP32 publicados fue 3,052e-5 segundos; la del agregado V fue
3,474e-7 MiD. No hay igualdad bit a bit entre este runtime y el original.
No se reconstruyeron eventos/RGB ni se ejecutaron los productores originales.

## Estado

V16 es una campaña cerrada de 32 cabezas y 80.000 updates, no otra copia de V17.
PHASE_ONLY pasa los criterios internos originales del 2% en tres seeds de
cabeza. El 4% posterior es una sensibilidad que no mejora una sola predicción;
no hace falta recurrir a ella para justificar PHASE_ONLY.

Los padres y normalizadores V13 siguen congelados. Las nuevas cabezas aprenden
sobre H, cuyas cabezas previas también se supervisaron en H. Es una nueva etapa
supervisada in-sample, no OOF. GROUP_CROSSFIT prueba otra receta y queda negativo;
no invalida automáticamente el entrenamiento completo, pero tampoco certifica
que el stacking pueda reproducirse con cualquier genealogía.

## Desarrollo interno

R_CTX: agregado 93,505354 y MAE 1,883995 s.
F_TRUE: 90,649718 y 2,003460 s.
PHASE_ONLY EVENT: 89,222960 / 89,198875 / 89,220028 por seed; media de métricas
89,213954. La media no es un ensemble de TTC.
ZERO seed7: 92,781912 y MAE 1,785782 s. El EVENT mejora el agregado sobre ese
control, pero no toda métrica: incluso ZERO tiene mejor MAE en V.

El predictor nuevo tiene 1.955 parámetros, 20 coordenadas de fases y relojes,
MLP 32/32, anchor R_CTX y residual libre en fase. Elimina estados ocultos de la
nueva lectura, NO los encoders y GRU necesarios para producir sus fases.
V16 respalda una combinación compacta de mediciones aprendidas. No demuestra
que todos los mapas espaciales sean irrelevantes ni contradice entrenar un
encoder mejor en V18.

## FCWD: comparación multimodal ya terminada

En 472 etiquetas RGB comunes, F_TRUE: MAE 0,700261 s / mean_MiD 55,969252;
Garl full registrado: 1,252557 s / 76,388385. Con las 597 etiquetas completas:
F_TRUE+fallback E_CTX: 1,429754 s / 53,929182;
Garl full+fallback event: 2,067345 s / 72,723138.
No repetir esta evaluación como si siguiera bloqueada. Registro estimado y
calibración extrínseca oficial siguen siendo conceptos diferentes. La historia
propia y la de Garl no son necesariamente iguales; RGB propio FCWD usa eventos
para registrar el ROI.

## PHASE_ONLY: compromiso de precisión, no dominancia

| FCWD, 597 etiquetas | MAE | mean_MiD | MiDc | MiDs | MiDl |
|---|---:|---:|---:|---:|---:|
| PHASE_ONLY S7 |1,582103|59,504120|93,304249|52,486365|21,738252|
| F_TRUE, misma ruta |1,429754|53,929183|102,323162|31,682935|16,445268|
| Garl event |2,146173|61,816523|95,056826|36,698339|34,211782|

PHASE_ONLY supera puntualmente a Garl event en MAE, mean_MiD y MiDc, pero los
IC de mean_MiD y MiDc cruzan cero. Es multimodal, no SOTA event-only.
La contribución parcial 0,5*c+0,3*s+0,1*l es 64,571859, peor que F_TRUE 62,310988
y Garl event 61,959093. NO es overall oficial: falta MiDn y no se renormaliza.

De la diferencia PHASE_ONLY-F_TRUE en mean_MiD (+5,57494): c aporta -3,03652;
s aporta +6,58601; l +0,95752; >10 +1,06792. El tramo medio explica más que todo
el déficit neto, compensado en parte por la mejora crucial.

Recalculado: PHASE_ONLY en 0–3s tiene MAE 0,300083 y sesgo +0,009124 s; en
3–6s MAE 0,737361 y sesgo -0,621473, con 90,48% de infraestimaciones. No es
correcto seguir describiendo todos los fallos actuales como TTC demasiado grande.
La mediana de AE crucial Garl (0,167955) sigue siendo menor que la de PHASE_ONLY
(0,243707), aunque su MAE crucial sea mayor. Deben mantenerse ambas lecturas.

Algebra descriptiva: sin alterar c/l, MiDs tendría que bajar de 52,486 a 43,777
para igualar sólo la contribución positiva de Garl: -16,59%. Esto NO es un
resultado obtenido ni debe convertirse en un umbral ajustado sobre FCWD.

## Signo, fallback y dominio

PHASE_ONLY mantiene las 13 salidas negativas de R_CTX sobre GT positivo Dev32.
Esas 13 filas representan 20,09% del error absoluto, pero 8,80% del error MiD.
Eliminar signos erróneos podría ser útil; no equivale a resolver todo el error.
No se propone tomar abs(TTC), imponer TTC positivo o elegir el experto más
pequeño: P/H/V contiene alejamientos válidos y un TTC negativo no es inválido.

En FCWD las cinco salidas negativas de PHASE_ONLY están en el fallback event.
Las 125 filas sin RGB tienen MAE 4,184322 y aportan aproximadamente 55,4% del
error absoluto actual; no serían arregladas cambiando sólo la cabeza RGB.
Las 472 filas con registro aceptado no son una muestra aleatoria de las 597.
Bajar NCC para recuperar consultas no demuestra mejor geometría.

## Ingeniería

La reutilización exacta event ya fue medida: no volver a presentarla como
implementación ausente. El perfil actual completo incluye registro FCWD y
padres GPU: p50 1.213,1175 ms y p95 3.060,5612 ms sobre 64 llamadas. La cabeza
cuesta p50 0,523 ms. No sumar medianas de etapas ni compararlo directamente con
un piloto streaming histórico que tiene otro scope.

La preparación test12 ya terminó: 6.762 tokens / 5.805 observaciones físicas.
Los 957 aliases se deduplican al construir historia, no se eliminan del output.
Todas las predicciones tienen RGB admitido. No se dispone de score oficial.

## Fuente principal

https://github.com/Kripta-Studios/e-jepa-ttc/blob/71ec9d900189a120b06c3956029496e6fd704f6f/docs/multimodal_results_20261012/INFORME_DETALLADO.md

El informe y los CSV son evidencia de ejecución. Las recomendaciones siguientes
son propuestas propias de continuación, no resultados ni obligaciones del autor.
