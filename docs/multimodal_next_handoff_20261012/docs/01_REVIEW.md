# Revisión de V13 completado y relación con V14/V15

## 1. Lo nuevo respaldado por las fuentes

Los doce fits V13 finalizaron: cuatro productores de 30.330 updates, dos PAIR de 6.840 y seis cabezas de 2.500. La suma de esos endpoints es 150.000 updates científicos; no reconstruye por sí sola la contabilidad física de recuperaciones. La cola conserva un bloqueo del contrato FCWD oficial, no de los entrenamientos. El registro FCWD estimado se completó en otra ruta explícitamente exploratoria.

En V (16.432 consultas, ocho grupos), E_CTX=106,082; E_H1=113,450; F_TRUE=90,650; F_ZERO=107,128; R_CTX=93,505 y R_H1=96,821 en group_macro_bucket_MiD. La fusión mejora al evento y al control ZERO. R_CTX mejora R_H1, pero no supera el mínimo prospectivo de 5%; se conserva ese gate negativo, sin traducirlo a ausencia total de señal temporal RGB. No hay contraste F_TRUE–R_CTX en los gates publicados.

V12 TRAIN40, V13 P/H/V, Dev32 y FCWD son poblaciones/recetas distintas. En particular E_CTX V13 no utiliza todos los productores TRAIN40 V12. No atribuir su menor generalización respecto a H8 a una única variable: cambian datos de productores, cadencia, pesos de muestras y receta.

## 2. Reconciliación de Dev32

La evaluación multimodal final usa 1.024 consultas/946 labels. Se verificó que las 1.024 predicciones heredadas H8/Garl y sus targets coinciden con la evaluación histórica de 1.024 consultas. Todas sus fechas también aparecen en el CSV ampliado de 3.947 consultas, pero sólo 39 conservan simultáneamente el mismo query_id: se renumeraron los índices.

La unión por (sequence_id, anchor_us), comprobando unicidad y contenido, encuentra las 1.024; las diferencias numéricas máximas están al nivel de serialización. En otro dataset con varios objetos por tiempo se necesita además identidad de objeto/ROI. Nunca unir por posición o por el query_id aislado entre manifiestos distintos.

Eso explica la comparación Garl full con MAE 1,868 frente a la histórica ampliada con 94,510: el segundo conjunto contiene consultas adicionales, incluido el extremo de ~335.544 s; no es evidencia de que el mismo checkpoint mejorara. La evaluación final no es un test nuevo.

## 3. Recálculos adicionales

`DESCRIPTIVE_METRICS.csv`/los outputs del script contienen métricas derivadas, no el scorer oficial.

Dev32, 946 labels:
- E_CTX: mean_MiD 192,388; MAE 1,8554 s.
- R_CTX: mean_MiD 131,381; MAE 1,2579 s; mediana AE 0,5121 s.
- F_TRUE: mean_MiD 130,157; MAE 1,3685 s; mediana AE 0,4936 s.
- Garl full: mean_MiD 168,632; MAE 1,8683 s; mediana AE 0,6915 s.

F_TRUE frente a R_CTX tiene diferencia mean_MiD −1,223 y MAE +0,1106 s. Bootstrap post hoc por 32 secuencias, 4.000 draws, estimador micro de suma de errores/suma de consultas: intervalos aproximadamente [−10,762,+8,571] y [−0,146,+0,429] respectivamente. No prueban equivalencia ni inutilidad de los eventos. La fusión gana MiD medio por secuencia en 16/32 y MAE en 14/32 frente a R_CTX. En FCWD disponible sí mejora al RGB propio en las tres secuencias. No descartar eventos por el resultado agregado Dev32.

FCWD, las mismas 597 labels:
- E_CTX: mean_MiD 71,028; MAE 1,9672 s.
- F_TRUE: mean_MiD 53,929; MAE 1,4298 s; RTE 18,049%.
- Garl event-only: mean_MiD 61,817; MAE 2,1462 s; RTE 26,887%.

F_TRUE ya tiene mejor mean_MiD (~12,76%), además de MAE/RTE, en esa comparación de sistemas. Sigue perdiendo en MiDc: 102,323 frente a 95,057. No hay overall oficial porque no existen labels negativos; no construirlo renormalizando bandas. F_TRUE tiene modalidad adicional y registro event-assisted. No es victoria event-only ni comparación contra Garl full.

## 4. El problema de registro tiene dos efectos distintos

495/630 consultas tienen RGB aceptado; entre labels son 472/597. En 125 labels, F_TRUE y F_ZERO coinciden exactamente con E_CTX. Esas filas explican el 61,28% del error absoluto F_TRUE.

La cobertura no es uniforme:
- crucial: 195/201 (97,0%);
- 3–6 s: 170/189 (~90,0%);
- 6–10 s: 62/108 (~57,4%);
- >10 s: 45/99 (~45,5%).

Sólo seis fallbacks están en la banda crucial. El déficit MiDc F_TRUE–Garl se descompone en aproximadamente 3,9038 puntos de las 195 filas aceptadas y 3,3634 de esas seis rechazadas. La cobertura es importante, pero no explica todo el déficit crucial. Mejorar la cobertura global de RGB no garantiza mejorar los casos cortos aceptados.

El MAE 0,7003 de la fusión sobre las 472 filas aceptadas no es un pronóstico sobre las 125 rechazadas. Se seleccionan otras condiciones. No calcular un supuesto MAE final sustituyendo arbitrariamente el error de fallback por 0,7003.

El estimador R_CTX consume RGB, pero su ruta FCWD necesita eventos para registrar la ROI. Por tanto esa ruta de sistema no es RGB-only, aunque el tensor del productor sí lo sea. NCC consistente no es una calibración extrínseca métrica ni una medición independiente de error de reproyección.

## 5. Fusión y extremos

`DualClockFusion` crea dos GRUs y suma una corrección a la mediana de las fases de expertos event. No combina explícitamente los TTC refinados finales de E_CTX/R_CTX, ni hereda automáticamente sus GRUs. En la receta publicada la fusión se inicializa como modelo propio.

Ejemplo descriptivo: en CCRs-1-medium-100-overlap-100:01, GT=6,520 s, E_CTX=7,888, R_CTX=−8,321 y F_TRUE≈−60. Hay conflicto entre modalidades y una salida extrema. El CSV final no permite saber cuánto viene de normalización, signo de productores o residual final. Otro caso tiene R≈1,727 para GT≈0,944 y F≈22,649. No convertirlos en reglas manuales de elegir RGB.

En Dev32 F_TRUE tiene 12 errores de signo; R_CTX 13 y E_CTX 21. En esas 12 filas al menos uno de los unimodales ya se equivocaba de signo; no se halló F_wrong con ambos refinadores positivos. Esto es relación entre salidas, no atribución causal: F_TRUE no consume esos dos refinadores como inputs.

Conservar fase antes/después de clamp, base, residual y expert outputs es necesario para localizar el mecanismo. No usar una media de TTC firmados ni cambiar a mano predicciones negativas a positivas. Las distribuciones/intervalos no se convierten automáticamente en probabilidades calibradas de colisión.

## 6. Relación con V14/V15

La propuesta de masas del apartado D del nuevo informe ya fue ensayada en V14: seis fits, NO_CANDIDATE. No repetirla. La variación de gradientes y saturación del residual que motivaron V15 siguen siendo hipótesis. Este paquete no contiene resultados V15 ni autoriza rehacer sus brazos. Consultar su estado real, no asumir ejecución por existir el handoff.

## 7. Ingeniería y afirmaciones

Las reparaciones de 9–12 de octubre terminaron; no son doce entrenamientos pendientes. No repetir la reparación de known ni de la mediana determinista. Verificar que los entrypoints nuevos usen los adaptadores ya admitidos.

El coste observado V13 F_TRUE desde HDF5+RGB tar es ~2.581 ms y E_CTX ~1.656 ms, bajo carga concurrente. Son otras rutas/condiciones que el piloto streaming V12 (~82 ms H8 y ~37 ms Garl optimizado). No derivar un slowdown del modelo ni una ventaja de velocidad mezclando esos perímetros.

La evaluación test12 no está puntuada. Debe mantenerse sin selección iterativa con scores privados. Una comparación local favorable frente a un checkpoint no prueba dominio de toda la literatura.

## Fuentes

Base: https://github.com/Kripta-Studios/e-jepa-ttc/tree/62cf4dbe914ea077c68e42983f869cd5422ab71c/docs/agent_handoff_20261012

Código revisado: `src/e_jepa_ttc/rgb_port/fusion.py`, `src/e_jepa_ttc/simplex_t/model.py`, `operational/rgb_port/train_heads.py`, `operational/fcwd_rgb_registration/{geometry,prepare,transfer}.py`. Los resultados externos proceden de sus dos `SCORED_PREDICTIONS.csv`; los diagnósticos adicionales y sus scripts están en el paquete.

Referencia pública de comparadores y test sin GT: https://github.com/NAIL-HNU/Garl-TTC (consultada el 12-10-2026). No se hizo búsqueda exhaustiva de toda la literatura para adjudicar SOTA.
