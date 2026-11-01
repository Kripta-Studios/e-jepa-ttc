# PAIR con conversión explícita del intervalo físico

Protocolo fijado después del diagnóstico de dos consultas `pilot_v2`, antes de
ejecutar la intervención. La prueba anterior está publicada en el commit
`4926b765`. No se usan etiquetas ni se actualizan pesos en este piloto.

## Hipótesis y población

En las dos consultas, A5/C2F RGB responden al cambio de intervalo según su
conversión analítica, mientras PAIR mantiene un TTC cercano al original. Eso
motiva separar su predicción aprendida de la conversión temporal. No demuestra
que esta separación mejore precisión ni que explique el error de todo Slider.

Reutilizar exclusivamente los tensores capturados y sellados del piloto
anterior: una consulta CCRm y una Slider-500. No volver a preparar sensores.
Ejecutar la intervención en las condiciones a=1, a=0,5 y a=2, conservando
selección causal, lotes y controles limpios emparejados del piloto anterior.

## Única intervención

Para cada PAIR, reconstruir sus tres variables temporales con el constructor
`pair133`: dt, log(dt), 1/dt, usando d0=0,1 s. Tokens y soportes se conservan.
Evaluar el mismo PAIR congelado para obtener su fase canónica sin emitir phi_c.
Transformar la fase antes de la política de emisión existente:

    q_c = -expm1(-phi_c) / d0
    q_nuevo = q_c * d0 / dt_real
    phi_nueva = -log1p(-d0 * q_nuevo)

No obtener q_c mediante `phase_to_ttc`, porque esa ruta introduce clipping y
un suelo de magnitud 1/60 antes de escalar. La política original `emitted_phase`
se aplica una sola vez por el consumidor, después de la transformación.

Para dt_real exactamente igual a FP32(0,1), devolver directamente la fase de
la llamada original con sus features originales, sin reconstrucción ni viaje
de ida y vuelta. Esta rama debe conservar sus bytes, incluso cerca de cero o
de la saturación. El resto del lote conserva tamaño y orden originales.

Rechazar explícitamente dt no positivo, valores no finitos o
1-d0*q_nuevo <= 0. No introducir un epsilon para fabricar fases. Inversos
negativos son matemáticamente válidos; registrar la saturación que aplique
la política de emisión original. Conservar rechazos completos de lote.

## Verificación y decisión

Tests: identidad FP32(0,1), cero, signo negativo, igualdad temporal analítica
con token fijo, dominio positivo inválido, tres columnas temporales coherentes
y lote mixto sin cambiar su agrupación. Registrar fase original canónica,
inverso corregido, fase corregida y fase emitida.

El piloto recalcula raw17, contextos y cabeza completa. Compara la respuesta
temporal de PAIR y de la salida final con el piloto original, manteniendo
explícitos los cambios de identidades admitidas. También mide cuánto cambia la
predicción limpia a=1: salvo la rama exacta de 100 ms, no se presume identidad.

Esta prueba no selecciona por accuracy. Si corrige la sensibilidad de PAIR,
el siguiente paso es evaluar precisión y cobertura agrupadas de desarrollo,
recalculando las historias afectadas. Para eAP se auditará qué deltas activan
la rama de identidad y qué contextos requieren reconstrucción. Una caché de
features actuales no sustituye los inputs de todas las observaciones históricas.
No se promueve ningún checkpoint por este piloto.
