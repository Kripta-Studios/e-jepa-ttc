# Interpretación de V15

Decisión de cribado: **NO_CANDIDATE**.

S-M1-L0-S7 frente a S-M0-L0-S7: MiDc cambia -11.06 %, MiDl +35.93 %, MiDn +31.38 % y MAE +15.69 %. Los signos positivos indican empeoramiento. Gates pareados incumplidos: large, mae, negative, sign, small. La tabla de resultados conserva también el contraste obligatorio con V14 M0.

R-C20-CTX-S7 frente a R-C05-CTX-S7: MiDc cambia +0.04 %, MiDl -0.12 %, MiDn -0.38 % y MAE -1.58 %. Los signos positivos indican empeoramiento. Gates pareados incumplidos: aggregate, crucial_10pct, crucial_groups_60pct, late_warning. La tabla de resultados conserva también el contraste obligatorio con V14 M0.

En S-M1, el clipping pasa de 37.16% a 18.84%. La prueba de igualdad del riesgo esperado y de su gradiente impide atribuirlo a una segunda ponderación. La estimación de varianza usa 32 batches en la inicialización: describe ese punto y no prueba que el clipping sea la única causa de las diferencias finales. M0 no mostró la misma reducción de varianza inicial.

En P, para 208 consultas de TTC ≤1 s, el límite de geometría fija deja fuera del alcance a 208 targets con C05 y a 0 con C20. Sin embargo, el MAE aprendido del productor cambia de 1.864 a 1.569 s y su mediana TTC de 1.706 a 1.300 s. El alcance disponible y el resultado aprendido son magnitudes distintas.

En H, para 52 consultas de TTC ≤1 s, el límite de geometría fija deja fuera del alcance a 52 targets con C05 y a 0 con C20. Sin embargo, el MAE aprendido del productor cambia de 2.087 a 3.703 s y su mediana TTC de 1.904 a 1.945 s. El alcance disponible y el resultado aprendido son magnitudes distintas.

La comparación C20/C05 mantiene el reset y los 6.000 updates del scorer en ambos brazos. El contraste con A5 o V14 históricos incluye esa nueva receta de entrenamiento; no se atribuye todo su efecto al cap. Los encoders permanecen compartidos. H es diagnóstico del readout y entrenamiento de la cabeza, no confirmación externa. Los avisos a 1 s son diagnósticos TTC, no validación AEB.

Estos datos no acreditan SOTA universal. Una campaña posterior necesitaría un protocolo nuevo, presupuesto propio y comparaciones de la misma población, modalidad y coste. El saldo reservado a réplicas no autoriza nuevos coeficientes ni selección sobre FCWD/Dev32.
