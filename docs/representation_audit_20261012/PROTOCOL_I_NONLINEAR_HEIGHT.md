# Cola I condicionada: lectura no lineal de la geometría congelada

Registro anterior al cribado H de G y H. Esta cola sólo puede comenzar si ambas quedan cerradas como NO_CANDIDATE y no se ha reclamado una familia de réplica. No cambia ni reabre sus gates. Es una investigación nueva dentro de la autorización V17 de18h GPU/1M updates, no una reasignación del techo local de G o H.

## Hipótesis y control

El ajuste P de G reduce el error de altura al conservar posición vertical, pero apenas cambia su error temporal respecto al pooling global. La regresión lineal no prueba que el mapa carezca de información útil para una función no lineal. Se conserva exactamente el pooling8×1, el encoder original, sus512 entradas, P/H/V y el objetivo geométrico de G. No se entrenan foreground, encoder, transporte, incertidumbre ni PAIR originales.

Control: SP1 de G, con su calibración lineal y scorer C05 seed7 ya terminado. Candidato: esa misma calibración, más MLP512→64→1 con SiLU y última capa cero. La entrada es la normalización P ya fijada por SP1. El resultado se suma a la corrección lineal antes del mismo clip[-2,2]. El candidato empieza exactamente en el control geométrico. No otra resolución, coeficiente, anchura, pérdida o ensayo de aprendizaje.

## Dos etapas fijas

1. Ajustar sólo el MLP nuevo durante6000updates sobre P uniforme, batch128, FP32/TF32off, AdamW1e-4→1e-5, warmup100/cosine6000, wd1e-4, clipping1. Objetivo: error cuadrático de corrección de log-altura y de su diferencia temporal, divididos por las varianzas P congeladas en SP1 y con peso unitario. Sin targets TTC en esta etapa, sin evaluación H para elegir epoch. Mantener fijo el scorer y todos los módulos originales.
2. Congelar el MLP. Reiniciar la última capa del scorer C05 y entrenar sólo ese scorer6000updates sobre P con la receta nativa de G: batch128, FP32, LR3e-4→3e-5, wd1e-4, clipping1, pérdida y sampler sin cambios. Comprobar igualdad de inicialización, sampler y optimizador frente al control G antes de entrenar. Los tokens originales de PAIR133, transporte e incertidumbre se conservan por identidad.

Congelar MLP y scorer completos antes de evaluar P/H. El candidato debe pasar todos los gates originales frente a SP1-S7 y frente al A5 histórico. Un intervalo favorable no sustituye un gate; ningún fallo autoriza otro MLP o LR.

## Réplica e integración condicional

Si pasa, reclamar la única familia condicional global y repetir las dos etapas del candidato para seeds13/23. Sus controles SP1 requieren dos nuevos scorers de6000updates con la misma calibración lineal P; no se presentan como reapertura o promoción de G. Congelar los seis endpoints finales antes del cribado de réplicas. Sólo si todos los contrastes pasan, reconstruir PHASE17 y normalizadores H coherentes, registrar la integración y entrenar seis H8 M0/L0 de2500updates; congelar antes de V. V permanece desarrollo expuesto. FCWD/Dev32 no ajustan ni seleccionan.

Máximo científico nuevo I:12000 iniciales +36000 condicionales de réplicas/controles +15000 condicionales H8 =63000updates. Máximo técnico:60 sintéticos. Todos cuentan dentro de los techos globales, sin obligación de consumirlos. Si no hay recursos o el presupuesto no admite el trabajo restante, conservar estado y dependencia exacta; no disminuir poblaciones ni precisión.

## Verificación exigida

Identidad exacta del candidato inicial con SP1, ausencia de targets en inferencia, original PAIR133/uncertidumbre inalterados, congelación de parámetros y buffers por etapa, paridad de outputs y gradientes de la ruta preparada frente a la composición de mapas nativos, y continuidad10 frente a5+restore+5 por etapa. Registrar el operador, los pesos del MLP, fuentes, hashes, muestras, curvas, normas/clipping y coste con scope. El MLP agrega coste de readout; no agrega otro encoder, y no se oculta ese coste.
