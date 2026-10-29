# Protocolo prospectivo: separar muestreo y rango del readout

## 0. Identidad y datos
Nueva rama V15, base `1c251781ac845a5d661523e607e0f7c4a8d3af8a`.
V14 NO_CANDIDATE, sus pesos y gates no se modifican. Roles P/H/V existentes se
conservan: P=53.890/24grupos, H=18.422/8, V=16.432/8; contrastar recibos vivos.
Productores matched P-only, nunca sustituirlos por TRAIN40. P admite entrenamiento
del nuevo readout; H entrena los refinadores; V sólo evalúa recetas congeladas.
V ahora es desarrollo expuesto y NO una confirmación independiente.

La inspección preliminar reutiliza los recálculos del paquete. Completar sólo las
magnitudes no presentes en P/H: cap por pareja, r requerido, residuos, known y edad.
Contar episodios y distribución de crop; no inferir nada a partir de metadata ausente.
Si se encuentra un bug real de entrada, tratarlo como fix separado con referencia
intacta y paridad/regresión; no ajustar en paralelo hiperparámetros para compensarlo.

## S: mismo objetivo esperado, otra distribución del muestreo
Dos fits nuevos S-M0-L0 y S-M1-L0, seed7, 2.500 updates. Usan exactamente la
TemporalRefiner17/GRU160 y receta CPU FP32/batch128 nativas V14, sin L1 nuevo,
normalizador H y padres originales. Inicialización seed7 idéntica.

Para masa w_i normalizada y probabilidad de draw p_i, minimizar
 mean_batch[(w_i/p_i)*loss_i].
El control histórico muestrea uniformemente: p_i=1/N, por lo que usa N*w_i.
Los nuevos brazos fijan p_i=w_i en H y por tanto la pérdida muestreada NO se vuelve
a multiplicar por N*w_i: se usa loss_i, conservando el mismo riesgo esperado.
No normalizar pesos dentro del batch. La igualdad es del objetivo y del gradiente
antes de clipping; AdamW y clipping no vuelven equivalentes las trayectorias.
No se garantiza menor varianza para todo vector de gradiente: medirla con batches H
fijos de diagnóstico, junto a exposición y clipping. H-only; p nunca depende de la loss en V.
Guardar el generador y las probabilidades; torch.multinomial reemplazo=True es
aceptable con dtype/versión congelados. M0 y M1 comparten stream aleatorio base,
pero NO se afirma igualdad de IDs cuando p cambia. Medir exposición/clipping.

Contrastes: S-M0 vs V14 M0/L0; S-M1 vs V14 M1/L0; y S-M1 vs S-M0.
Esto no agrega datos independientes ni arregla por definición el tradeoff de M1.
Su ejecución no cancela el estudio R por un resultado negativo.

## R: ampliar el rango de un productor sin cambiar la información de entrada
Primero A5, no A5+C2F a la vez. Mismo E_A5_MATCHED padre final P-only. Congelar:
encoder, foreground/geometry/projectors, transport, uncertainty, pair projector,
auxiliary_inverse_ttc y toda otra pieza distinta del MLP residual.
No se usa el target al preprocesar inputs ni en el forward de inferencia.

Dos readouts prospectivos: R-C05 y R-C20. Definir s=0.5*(score(fwd)-score(reverse)).
 delta_r(C) = C*tanh((0.05/C)*s), C en {0.05,0.20}.
Para C=.05 es la función original. Ambas tienen derivada0.05 en s=0, igual número
de parámetros y misma antisimetría. Las dos copian el mismo hidden scorer del padre,
pero reinician a cero su Linear final y su bias antes del nuevo fit. Con ello tienen
exactamente la misma salida inicial analítica, no una recalibración inmediata del
checkpoint original. Los encoders y C2F/PAIR históricos permanecen intactos. Durante el fit, mantener
las piezas congeladas en eval y sólo el scorer residual en su modo de entrenamiento
registrado, incluido su dropout y RNG. La extracción de su caché no activa dropout
en los bloques congelados.

El nuevo checkpoint debe registrar `operator=equal_slope_tanh_v1`, C y slope=0.05,
además del padre y hashes. Entrenamiento, extracción e inferencia deben cargar el
mismo operador. NO guardar C=.20 como si el loader nativo C*tanh(s) lo implementase:
ese loader produciría otra función. C05 y C20 usan el wrapper nuevo y una prueba
específica debe rechazar un checkpoint con operador o escala diferentes.

Entrenar sólo el scorer residual, 6.000 updates por brazo, batch efectivo128,
CPU FP32 si se dispone de caché, AdamW lr3e-4→3e-5 cosine con warmup100,
weight_decay1e-4 y grad_clip1.0. Ambos consumen idénticos IDs P por seed,
muestreo uniforme de queries P y los mismos pesos/objetivo nativo del productor.
La receta es nueva (readout-only), no continuación equivalente de las18épocas.
Todas las máscaras, log-ratio targets y top-k global se calculan exactamente según
el objetivo nativo admitido; T2/T3 se reducen con sus denominadores reales.
No usar L1 de V14 ni reweight M1 en esta comparación: cambia sólo el rango.

Usar caché P de entradas anteriores al residual (geometry tokens y transport
forward/reverse), alturas, deltas y salidas congeladas necesarias para el loss.
Los términos constantes se pueden cachear; todo término que dependa del nuevo
residual, TTC, variance propagation, known o risk se recomputa. Debe verificarse
forward Y gradiente del scorer contra la ruta raw de referencia en P antes de
usar la caché. No basta paridad de predicción. Si no se implementa caché exacta,
usar forward del modelo congelado completo; declarar el coste y respetar recursos.
No cambiar el decoder ni omitir negativos donde una auxiliar geométrica no exista.
La función nativa define qué targets admiten cada auxiliar; la cohorte completa sigue
siendo evaluada. Detectar NaN/Inf y registrar fallos, sin sustituirlos por ceros.

El C=.20 NO garantiza cubrir todo TTC corto. Es una prueba fija de ampliación4x,
no un C elegido para mejorar V. Conservar score del A5 histórico como tercera
referencia, ya que ambos nuevos readouts fueron reinicializados.

## R-H: readout nuevo -> features y refinador coherentes
Con cada A5-R congelado, regenerar entradas H/V de la familia. C2F sigue original.
PAIR puede reutilizarse sólo si se demuestra igualdad de pair_token, soporte y
las133entradas nativas: el pair_projector no depende del residual en el código
inspeccionado, pero comprobar la versión efectiva. No asumir igualdad de features17:
recalcular fase A5, margen, incertidumbre propagada y todos sus desacuerdos.
Normalizador nuevo por familia ajustado exclusivamente con H; no reciclar el antiguo.
V usa ese normalizador sin ajuste. Nunca mezclar coordenadas/padres entre familias.

Entrenar R-C05-CTX y R-C20-CTX sobre H: dos fits seed7 de2.500updates, misma receta
M0/L0 V14, muestreo uniforme, misma arquitectura/timestamps/cap. No copiar una cabeza
histórica y llamar al cambio de input paridad. Conservar baseline V14M0/L0.
El contraste C20CTX−C05CTX prueba utilidad del rango; contra V14M0 prueba si aporta
al sistema original, no sólo si rescata el reset del control C05.

## Evaluación y selección
Congelar los cuatro endpoints de cabeza nuevos S0/S1/R05CTX/R20CTX y sus padres
antes de la primera evaluación V de esta cola. No usar resultadosS para rediseñarR.
Reutilizar exactamente cohortes y scorer V14. Separar macro grupos/bandas de mean/MAE.
Aplicar los gates V14 intactos como salvaguardas de promoción; candidatos S-M1 y
R-C20CTX deben pasar frente a su control pareado y frente al V14 M0/L0 histórico.
No promover el C05 o S-M0 como candidato preregistrado a posteriori: reportar mejoras
exploratorias sin alterar esta regla. Conservar resultados de todos.

Como máximo una familia para réplica. Si ambas pasan, menor MiDc; desempatar con overall
luego coste con mismo scope, nunca FCWD. Bootstrap de grupos pareado, 1.000 draws,
semilla12345, descriptivo por comparaciones múltiples. No exigir CI favorable para
replicar. No relajar gates sobre bandas o avisos porque la media sea buena.

Réplica S: S-M0 y S-M1 seeds13/23, cuatro cabezas=10.000updates.
Réplica R: repetir readoutC05/C20 y CTX de cada uno para seeds13/23:
cuatro readouts6.000 + cuatro cabezas2.500 =34.000updates.
Encoders permanecen compartidos: no son réplicas completas de representación.
Reportar cada seed y la media de pérdidas emparejadas por consulta, no un ensemble
de TTC construido a posteriori. El candidato debe conservar los gates frente a
su control pareado en el agregado de seeds y se publican todas las diferencias.
La baseline histórica seed7 conserva esa identidad: no fabricar sus seeds ausentes.
V ya fue observada durante selección; estas réplicas cuantifican variación de
optimización condicionada a esos grupos, no generalización a escenas nuevas.
Total científico máximo:5.000+12.000+5.000+34.000=56.000.

## Transferencia y cierre
Si no pasa ninguna, cerrar y entregar. Si pasa la familia replicada y conserva
sus condiciones, congelar candidato y controles antes de una evaluación completa
FCWD/Dev32. Son desarrollo externo ya expuesto, no confirmación nueva. No usar
sus errores para cambiar esta receta. FCWDfull sólo con calibración admitida.
Comparar Garl nativo y sensibilidad común separadamente; no puntuar faltantes como
ceros. No inventar overall si faltan negativos. Preparar propuesta de refit y
confirmación después; no hacer uploads/submissions ni usar test12 como tuning.

## No autorizado
Otros cap valores, losses/samplers, nuevas épocas de V14, joint training de encoder,
RGB adicional, nuevos datasets, otro router, mezcla posthoc según GT, selección de
casos FCWD, seeds extra, modificación deV13 o de sus admisiones. El saldo no habilita
esas ramas. Un bloqueo de cachéR permite terminarS y los análisis, no inventar datos.
