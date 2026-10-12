Añade esta revisión al contexto de V18 SIN reiniciar ni cambiar los fits activos.

V16 publicado en 71ec9d900189a120b06c3956029496e6fd704f6f obtuvo PHASE_ONLY,
1.955 parámetros sobre fases/relojes de padres V13, con agregado V 89,223/89,199/
89,220 en tres heads. Es un candidato multimodal; no es un encoder nuevo ni
SOTA oficial. Conserva F_TRUE, R_CTX y los event-only como comparadores separados.

Resultados de transferencia que motivan diagnósticos, NO selección de tus fits:
FCWD PHASE_ONLY tiene MiDc 93,304 pero MiDs 52,486 frente a 31,683 F_TRUE.
En 3–6 s subestima 90,48% de las consultas. En Dev32 conserva los 13 signos
erróneos de R_CTX; esos casos aportan 20,09% del AE pero 8,80% de MiD total.
No apliques abs(TTC), reglas por dataset ni un offset aprendido con esos labels.

Continúa tu matriz admitida NATIVE-P2-FT/DIRECT/CORR/CORR-FROZEN y las ramas
adaptativas de tu protocolo, con su presupuesto existente. No canceles encoder
entrenable por el fracaso del congelado. No vuelvas a las calibraciones V17.

Incluye por P/H/V: raw/emitted phase, signo, cap, sesgo por banda/subrango,
exposición/episodios y coste. Si un fit ya empezó, estos diagnósticos se marcan
posteriores; no se cambian loss, endpoint, sampler o gates. Un readout que mejore
H/V no demuestra haber mejorado FCWD hasta la evaluación completa congelada.

Si gana DIRECT, consérvalo; no fuerces CORR por preferencia. Si la representación
actualizada gana, replica el pipeline que corresponda. Si falla, interpreta P/H
y escoge sólo una siguiente hipótesis dentro de la autorización, no un barrido
para rescatar el score. No es necesario que cada control ya sea SOTA para probar
la siguiente alternativa legítima.

Cuando cambien los productores, reconstruye features, normalizadores y cualquier
fusión dependiente. No conectes pesos nuevos al PHASE_ONLY congelado como si el
pipeline fuera equivalente. No refit TRAIN40 hasta congelar receta e identidades;
los grupos usados para refit dejan de ser validación de esos pesos.

Los scores oficiales del candidato V16 se mantienen separados de tus decisiones
de aprendizaje. No uses test12 ni transferencias expuestas para escoger receta.
La ausencia de score oficial no bloquea TRAIN/H. No reclames como tuyo el trabajo
V16 ni generes otro propietario GPU sobre el mismo host.
