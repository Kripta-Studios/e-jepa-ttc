# Diagnóstico de gradientes de A1-FULL

## Resultado

El diagnóstico terminó como `COMPLETE_ZERO_UPDATE_DIAGNOSTIC`. Se ejecutó sobre el
checkpoint histórico A1-FULL de 12.000 updates, con SHA-256
`72e0860e20abff01242219f4692bf88aa4ed3bfafeb1ccfac976d6398530742f`.
No realizó updates, no abrió H y no produjo un candidato. El hash del modelo antes y
después fue idéntico:
`573129ff571182bd5f094a286789083e3b312f80839f70825672b43d45f78f80`.

La evidencia completa está en
`artifacts/decision_recovery_20261017/diagnostics_gradients/RESULT.json`, con SHA-256
`3129b769eb073a510a016e14f8aeb17ce4663f5a24a82e2a9cb4206d9d500cec`. Su recibo
`COMPLETE.json` registra cero updates.

## Qué se midió

El panel contiene 384 consultas P: cuatro consultas seleccionadas por hash en cada una
de las cuatro bandas y cada uno de los 24 grupos. El estimador Horvitz-Thompson recuperó
las masas L0 registradas: 0,50 crucial, 0,30 pequeña, 0,10 larga y 0,10 negativa. Se usó
la pérdida original `emitted + raw`; el término `raw` ya contiene su factor 0,1.

El modelo permaneció en modo de entrenamiento, con dos módulos de dropout activos. El
estado RNG de CPU y CUDA se restauró antes de cada banda, por lo que las máscaras quedan
emparejadas por posición entre bandas. Las cuatro restauraciones fueron verificadas.

Las normas siguientes usan el gradiente combinado normalizado por la masa de su banda:

| Banda | Pérdida media | Encoder | Torre espacial | Cabeza |
|---|---:|---:|---:|---:|
| Crucial | 0,2524 | 4,7563 | 1,5420 | 0,8346 |
| 3–6 s | 0,1792 | 1,7211 | 1,1216 | 0,2858 |
| 6–10 s | 0,2468 | 3,3129 | 1,2587 | 0,3597 |
| Negativa | 0,3526 | 1,8073 | 1,9337 | 0,7237 |

Todos los bloques tuvieron señal finita. La banda negativa no presenta un gradiente
débil: su mayor norma está en la torre espacial y su pérdida media es la mayor de las
cuatro bandas. Esto descarta, en este panel, la explicación simple de que los negativos
no aportan señal de optimización.

## Direcciones entre bandas

Los cosenos del gradiente combinado L0-HT fueron:

| Par | Encoder | Torre espacial | Cabeza |
|---|---:|---:|---:|
| Crucial / 3–6 s | −0,620 | −0,606 | −0,377 |
| Crucial / 6–10 s | −0,908 | −0,594 | −0,478 |
| Crucial / negativa | **−0,684** | **−0,551** | **+0,118** |
| 3–6 s / 6–10 s | +0,531 | +0,644 | +0,588 |
| 3–6 s / negativa | +0,733 | +0,422 | +0,249 |
| 6–10 s / negativa | +0,593 | +0,562 | +0,680 |

La evidencia principal es una oposición entre la banda crucial y las demás en el encoder
y la torre espacial. Para crucial frente a negativa, la oposición es clara en encoder y
torre, mientras que la cabeza es ligeramente concordante. Los negativos, en cambio,
están alineados con las bandas 3–6 s y 6–10 s en los tres bloques.

Por tanto, el patrón no es un conflicto uniforme de «positivos contra negativos» ni un
problema localizado únicamente en la cabeza. La banda crucial constituye la dirección
distinta en esta realización, especialmente dentro del encoder. Esta observación es
compatible con interferencia durante la adaptación compartida, pero no prueba que esa
interferencia cause la regresión negativa de H.

## Actualización AdamW local

Se reconstruyó de forma analítica el siguiente paso AdamW a partir de los momentos y las
tasas guardadas en el checkpoint. El gradiente global estimado tenía norma 1,8388 y se
aplicó el clip registrado a 1,0, con factor 0,54385. No se llamó al scheduler ni a
`optimizer.step`.

El producto `g_banda · delta` aproxima el cambio local de pérdida: un valor negativo es
favorable bajo la linealización y uno positivo es adverso.

| Banda | Total | Encoder | Torre espacial | Cabeza |
|---|---:|---:|---:|---:|
| Crucial | −6,195e−4 | −3,563e−4 | −2,262e−4 | −3,691e−5 |
| 3–6 s | −7,846e−6 | −8,479e−6 | +1,013e−6 | −3,793e−7 |
| 6–10 s | +2,381e−5 | +1,954e−5 | +4,920e−6 | −6,582e−7 |
| Negativa | **−2,213e−5** | **+5,380e−6** | **−2,008e−5** | **−7,424e−6** |

La actualización hipotética es favorable en total para la banda negativa, pero su
componente del encoder es adversa; la torre y la cabeza compensan ese efecto local. La
misma actualización perjudicaría ligeramente la banda 6–10 s. Los momentos acumulados y
el precondicionamiento de AdamW importan: un coseno negativo entre gradientes no determina
por sí solo el efecto de la actualización real.

## Decisión

Este diagnóstico aporta evidencia para investigar cómo comparte capacidad el encoder,
pero **no establece un conflicto persistente**. Es un panel P de una sola selección, un
checkpoint, una semilla y una realización emparejada de dropout. No mide variación entre
paneles, checkpoints, semillas o etapas del entrenamiento, y no contiene evidencia de H
ni de generalización independiente. La aproximación `g · delta` es de primer orden; no
es una medida de la pérdida después de aplicar el paso.

En consecuencia, el resultado **no justifica automáticamente PCGrad, GradNorm ni otro
método de modificación de gradientes**. Antes de esa intervención haría falta demostrar
que la oposición se repite a lo largo del entrenamiento y que predice daño fuera del
panel, comparando después cualquier método con L0/LW bajo el mismo padre, presupuesto y
salvaguardas. La evidencia actual sí favorece mantener como hipótesis una intervención
dirigida al encoder o a su adaptación compartida. El [contraste L0/LW ya completado](LOSS_SCREEN_RESULTS.md)
mejora negativos, MAE y signo al aumentar su peso, pero empeora agregado y crítico:
ese cambio de masas no resuelve conjuntamente los criterios registrados.
