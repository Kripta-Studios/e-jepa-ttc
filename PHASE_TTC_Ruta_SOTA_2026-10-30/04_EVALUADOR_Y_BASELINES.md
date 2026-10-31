# 04 — Evaluador, referencia del TTC y baselines

## Objetivo

Construir una comparación en la que una diferencia de score signifique una diferencia del sistema evaluado, sin cambiar silenciosamente timestamps, geometría, población o tratamiento de fallos. Empezar por las predicciones ya guardadas; no hace falta repetir inferencia para validar gran parte del evaluador.

El [plan revisado](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/docs/superpowers/plans/2026-10-30-evttc-transfer-diagnosis.md) distingue resultado oficial recibido, reproducción equivalente, diagnóstico parcial y desarrollo independiente. Mantener esas cuatro categorías en toda tabla.

## Paso 1. Escribir el contrato del target

Antes de puntuar, documentar:

| Campo | Decisión necesaria |
|---|---|
| Identidad | Grabación, objeto, índice original y timestamp textual original. |
| Reloj | Sensor de referencia, unidad, origen, offset y posible deriva. |
| Instante del TTC | Consulta, último frame, inicio/final del par o anchor; debe ser explícito. |
| Geometría | Distancia y velocidad relativas respecto a qué punto, superficie y eje. |
| Signo | Aproximación, alejamiento, no colisión, infinito y ausencia de medida. |
| Interpolación | Variable interpolada, método, bordes, gaps y precisión temporal. |
| Métrica | Fórmula, escala, denominador, valores cercanos a cero y elegibilidad. |
| Fallos | Entrada ausente, predicción no finita, GT ausente, salida fuera de soporte. |
| Población | Consultas obligatorias y exclusiones oficialmente admitidas. |
| Coste | Etapas incluidas, sincronización GPU, carga inicial y procesamiento por consulta. |

No ajustar un offset o factor para hacer coincidir errores. Primero contrastar relojes y geometría con metadatos, calibración y fixtures conocidos. Obtener del organizador los targets o instrucciones equivalentes cuando sea posible, sin dar por realizada esa comunicación.

## Paso 2. Distinguir expansión, fase y referencia temporal

Este fixture es una **prueba propuesta del contrato**, no un defecto demostrado del código.

Supongamos aproximación frontal a velocidad relativa constante, tamaño físico y focal constantes. Si `h` es tamaño aparente, `h ∝ 1/Z`. Para dos instantes separados por `Δ = 0,1 s`, elegimos TTC al final `T₁ = 1 s`. El TTC al inicio es entonces `T₀ = 1,1 s`, y:

\[
g=\frac{h_1}{h_0}=\frac{Z_0}{Z_1}=1{,}1,
\qquad
T_1=\frac{\Delta}{g-1}=1,
\qquad
T_0=\frac{\Delta g}{g-1}=1{,}1.
\]

El logaritmo de crecimiento es `r = log(g)`. Se usa aquí `g` para no confundir este cociente de alturas con la variable `η` de la métrica MiD en el documento 02. Por tanto, `q = expm1(r)/Δ` representa el inverso del TTC al final bajo esas hipótesis. El repositorio también emplea una **coordenada de fase canónica distinta**, `φ(T) = −log(1 − 0,1/T)`, con su inversa y soporte numérico. Esa definición aparece en [phase.py](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/src/e_jepa_ttc/simplex_t/phase.py).

En este ejemplo, `r ≈ 0,09531`, mientras que `φ(T₁) ≈ 0,10536`. **No introducir `r` donde la API espera `φ`**. Alimentar el decodificador canónico con `r` daría 1,1 s, pero eso sería una confusión semántica de la prueba, no evidencia de que el modelo actual decodifique mal. Los productores y la cabeza pueden estar utilizando correctamente sus conversiones y supervisión.

Implementar la prueba con salidas esperadas en ambos instantes y trazar qué timestamp representa cada valor real. Repetir con varios intervalos, alejamiento y aceleración, indicando cuándo deja de cumplirse la hipótesis de velocidad constante. No sustituir la constante canónica de 100 ms por la cadencia del sensor sin derivar todo el contrato aprendido. El [adaptador existente](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/docs/decisions/ADR-0043-evttc-competition-sensor-adapter.md) declara deltas reales en productores y una conversión final fija: deben auditarse ambas funciones por separado.

## Paso 3. Alinear etiquetas con cobertura explícita

El [anexo](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/docs/superpowers/plans/evidence/2026-10-30-evttc-audit.json) registra 354 de 365 consultas de carretera dentro del rango temporal del CSV. Las últimas 11 de CCRs1-medium quedan fuera. Esta cobertura no certifica equivalencia del target oficial.

Procedimiento:

1. Aplicar los offsets enteros demostrados, conservando timestamps originales y transformados.
2. Verificar orden, duplicados, unidades, gaps interiores y calidad de referencia.
3. Interpolar únicamente dentro de un intervalo admisible según el contrato fijado.
4. Mantener las 11 consultas fuera de cobertura como `missing_gt`; no extrapolarlas.
5. Separar Slider mientras no exista una referencia equivalente comprobada.
6. Puntuar el subconjunto elegible con el rótulo **diagnóstico local**.

La máscara de elegibilidad depende de datos y target, nunca de que la predicción sea favorable. Una salida fallida conserva su fila y su razón. El ledger propuesto incluye `query_id`, `query_us`, `target_us`, `target_status`, `prediction_status`, TTC bruto y versión del scorer.

**Cierre:** dos modelos reciben la misma lista de IDs, los mismos targets y la misma política. Para llamar a un resultado reproducción oficial, también debe verificarse el procesamiento exacto y coincidir con el informe dentro de su redondeo. Si faltan referencias, avanzar con desarrollo independiente y conservar el estado «equivalencia oficial pendiente».

## Paso 4. Conservar los negativos y separar el postprocesamiento

Según el plan, los organizadores sustituyeron dos negativos mediante interpolación entre predicciones válidas circundantes. La fórmula completa, los vecinos y el tratamiento de bordes deben confirmarse antes de reproducir ese procedimiento. Las salidas originales permanecen firmadas. [Fuente](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/docs/superpowers/plans/2026-10-30-evttc-transfer-diagnosis.md).

Publicar columnas separadas para salida bruta y política del organizador. La interpolación que usa un vecino posterior no es un filtro causal. No incorporarla a inferencia ni aplicar valor absoluto, clipping o exclusión de arranque para reproducir una cifra favorable.

Con targets compatibles y distintos de cero, un diagnóstico de error relativo puede usar `100·abs(pred−GT)/abs(GT)`, pero la implementación oficial debe confirmarse. Cualquier epsilon, censura o exclusión cerca de cero forma parte del contrato y se fija previamente. MiD de GarlTTC, error relativo de EvTTC y pérdida en fase son cantidades diferentes; no se convierten unas en otras por su nombre ni se mezclan en una media.

## Paso 5. Ejecutar baselines informativos

| Baseline | Pregunta que responde | Condición de comparación |
|---|---|---|
| PHASE original sellado | ¿Qué aporta la intervención frente a lo que ya existe? | Mismos inputs, población y scorer; adaptador identificado. |
| Rama RGB congelada | ¿Los eventos mejoran la predicción fuera del dominio fuente? | Exportación de la rama real, sin reentrenar ni usar GT en entradas. |
| Rama de eventos congelada | ¿El error aparece antes de fusionar? | Mismos tiempos y consulta; soporte registrado. |
| Garl completo, visual y eventos | ¿Cómo se compara con un método aprendido externo? | Pesos, modalidad, preprocessing, historia y cobertura declarados. |
| Geometría de expansión / FoE / CMax | ¿Una regla física ya explica la señal necesaria? | Misma referencia TTC y recursos de entrada explícitos. |
| Control temporal sencillo | ¿Una propuesta compleja supera una dinámica causal simple? | Estado reiniciado; usa predicciones previas, nunca TTC real previo. |

EvTTC publica referencias Image’s FoE y CMax en su [página de competición](https://nail-hnu.github.io/EvTTC/competition/); Garl ofrece [código y contratos públicos](https://github.com/NAIL-HNU/Garl-TTC). Las cifras publicadas son contexto hasta reproducir o confirmar el mismo protocolo. Si una referencia usa otra cámara, cajas, historia o modalidades, mantener esa diferencia visible.

La exportación de ramas y los controles nuevos son **trabajo propuesto**, no baselines ya puntuados en esta sesión. Empezar con el baseline congelado y los artefactos existentes; añadir las referencias necesarias para refutar la hipótesis concreta, sin abrir una búsqueda indiscriminada.

## Paso 6. Comparar Garl sin esconder sus huecos

El run documentado aporta 424/434 salidas finitas en Garl completo, 426/434 en visual y 434/434 en eventos. Su informe existente mide desacuerdo con PHASE, no precisión frente a GT. [Documentación del run](https://github.com/Kripta-Studios/phase-ttc/blob/30b28b5c14eb9ed60067ed2f1dc262d7b60f2654/docs/garl_evttc_competition.md).

Producir dos vistas:

1. **Pareada sobre consultas compartidas:** lista fija de IDs con targets compatibles y salidas disponibles en ambos. Advertir que puede representar casos más fáciles.
2. **Población completa requerida:** cobertura, causas de fallo y resultados bajo la política oficial si está confirmada.

No rellenar huecos de Garl con PHASE ni con interpolación. No afirmar superioridad global desde una media que omite sus fallos. La tabla VI de eAP y las tres filas correspondientes de PHASE tampoco son una comparación controlada mientras no coincidan consultas y procesamiento.

## Pruebas y entrega

Implementar pruebas de traslación de reloj, puntos exactos, interpolación interior, fuera de rango, gaps, targets cero/negativos, fallos de salida y ejemplos donde macro y micro difieren. Verificar que modificar predicciones no cambia la máscara de GT elegible.

Entregar un contrato legible, ledger completo, tablas generadas por código y lista de diferencias respecto al scorer oficial. Usar [la plantilla de reporte](plantillas/REPORTE_RESULTADOS.md). Ningún CLI nuevo descrito en este paquete debe presentarse como ejecutable hasta implementarlo y verificarlo.

Siguiente documento: [05 — Diagnóstico de sensores y modelo](05_DIAGNOSTICO_DE_SENSORES_Y_MODELO.md).
