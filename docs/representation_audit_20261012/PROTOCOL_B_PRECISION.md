# V17 cola B: precisión congelada del NLL

Registrada antes de abrir los resultados nuevos de V de la cola A. Presupuesto independiente V17 compartido: máximo total 18h GPU/1M updates físicos, no saldo V15.

## Evidencia P/H y única intervención

La precisión inversa de varianza de A5 original tiene media P=26551,97057752027. En P≤1s su mediana es4323,87, frente a29441,24 en toda P. En H≤1s es2669,92 frente a30915,24 global. El error cuadrático estandarizado medio del log-ratio es≈20,7 en las consultas≤1s de ambos roles. Es un diagnóstico de la distribución de pesos y calibración; **no equivale a masa de gradientes ni prueba causalidad**.

Hipótesis: mantener una precisión heteroscedástica aprendida por un productor limitado a C05 condiciona la adaptación del scorer al ampliar el alcance a C20. La prueba modifica sólo la precisión que multiplica el residual cuadrático del NLL; no modifica las incertidumbres emitidas, arquitectura, datos o sampler.

U0: receta scorer-only C20 de V15, operador equal-slope0,05, última capa reiniciada, todos los demás pesos congelados, NLL heteroscedástico original.

U1: mismo entrenamiento, sustituyendo exp(−logvar_i) del término cuadrático por la media calculada exclusivamente en P original:26551,97057752027. Se conserva el término constante0,5logvar, riesgo, cola top10%, regularización, demás componentes y normalización nativas. El sampler permanece uniforme P; es un cambio del objetivo y de sus pesos, no sobremuestreo ni una reparación equivalente del checkpoint.

Seis fits: U0/U1 × seeds7/13/23,6000updates cada uno, batch128, AdamW3e-4→3e-5, warmup100, cosine, wd1e-4, clip1, FP32 GPU, TF32off. Dropout nativo0,05 del scorer entrenable, RNG y orden pareados. Toda CNN, geometría, transporte, PAIR, foreground y incertidumbre permanecen congelados. Se ligan operador, precisiónP, fuentes, caché y RNG en cada checkpoint.

## Cribado fuera del ajuste del scorer

La cola B se evalúa en P descriptivo y H después de congelar sus seis readouts. H no entrena el scorer; no se usa para elegir checkpoint, coeficiente o seed. No se abre V para esta cola de productores. Se aplican los gates originales de V14, sin cambios de umbral, a cada comparación U1/U0 sobre H y al histórico A5 original sobre H. Todos deben pasar antes de reconstruir H y entrenar refinadores. Un fallo termina esta cola como negativo; no se rescata una semilla.

Si los gates de productor pasan, la integración posterior requerirá reconstruir H/V PHASE17 y normalizadores H, comprobar PAIR/clocks, entrenar los seis refinadores2500updates y congelarlos juntos antes de V. Esa extensión sólo puede usar el mismo presupuesto V17 y debe quedar registrada con fuentes exactas antes del primer fit de cabeza. No se reutiliza una caché downstream vieja.

Pruebas previas: gradiente del NLL homogéneo contra fórmula cerrada; U0 conserva objetivo nativo; módulos congelados idénticos; dropout/RNG y reanudación GPU10 vs5+5 exactos; los ejemplos técnicos son sintéticos. El cap C20 no cambia. No encoders nuevos, ni RGB fits, ni FCWD/Dev32 para entrenamiento o selección.
