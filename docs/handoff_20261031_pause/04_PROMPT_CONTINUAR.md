# Prompt exacto para continuar la investigación

Copiar el bloque siguiente a un agente que tenga acceso al mismo workspace.
Expresa una nueva instrucción de reanudar después de la pausa solicitada.

```text
Reanuda la investigación PHASE-TTC desde el handoff existente. No empieces de
cero ni repitas entrenamientos o extracciones ya completados sin una razón
concreta verificada.

Workspace:
C:\Users\Álvaro Schwiedop\Desktop\KriptaStudios\EVOCON_JEPA_Codex_Handoff\e-jepa-ttc-v18-joint-measurement
Repositorio: Kripta-Studios/phase-ttc
Rama: research/evttc-transfer-v2

Lee primero AGENTS.md y docs/handoff_20261031_pause/README.md, y después sus
documentos 01, 02 y 03. Comprueba git status, git log y los recibos locales antes
de confiar en estados históricos. Los documentos antiguos pueden describir
problemas ya resueltos. Conserva cualquier trabajo ajeno y los runs fallidos.

Mantén el objetivo completo: continuar investigando, entrenando y validando
PHASE-TTC hasta obtener evidencia reproducible de SOTA en EvTTC bajo un protocolo
comparable, preservando eAP y sin usar consultas de evaluación como datos de
ajuste ni presentar métricas locales como oficiales. Reanudo el trabajo que
había pedido pausar. Si el sistema conserva ese objetivo pausado, usa su flujo
de reanudación; no lo marques completo ni sustituyas el objetivo por terminar
un piloto. Si hace falta crear un objetivo en un hilo nuevo, te lo solicito con
ese texto y sin presupuesto de tokens fijado.

Estado actual:
- El entregable output/phase_submission/FINAL_MODEL sigue vigente.
- Hay 312 ajustes completos de fusión; ninguno de los candidatos nuevos pasó
  todos los criterios. No son 312 entrenamientos de encoders.
- El piloto temporal output/phase_timewarp/pilot_v2 terminó y reprodujo
  exactamente el baseline. Identificó sensibilidad de PAIR a la cadencia.
- CanonicalPair está implementado y probado. El piloto
  output/phase_timewarp/canonical_v1 terminó: mejora consistencia temporal en
  las dos consultas, no demuestra mejor accuracy. El contexto de eventos de
  carretera ralentizado empeora y debe conservarse en cualquier informe.
- La auditoría eAP actual está en
  output/phase_timewarp/eap_current_pair_audit/verified_run_v2.
  Eventos conserva exactamente las 88.744 salidas PAIR; RGB cambia poco. Eso
  no certifica conservación de contextos/cabeza ni generalización eAP.
- No hay un nuevo entrenamiento ni una extracción de 149 consultas corriendo.
  La pausa se pidió después de terminar lo lanzado.

Primera línea de trabajo: desarrolla y ejecuta el screen de PAIR canónico en
las 149 consultas ya fijadas, siguiendo 03_PENDIENTES_Y_CONTINUIDAD.md. Reutiliza
expanded149 y añade PAIR133 con identidad de observación y lote; raw17 no permite
recuperar los tokens. Reproduce primero el original; después recalcula PAIR,
raw17, contextos y cabeza con el mismo wrapper. Distingue F0 del router adaptado
del entregable. Sella inferencia y cobertura antes de unir targets. Conserva la
consulta nativa ausente y los rechazos nuevos. No cambies d0 ni la corrección
para acomodar resultados. Este screen de 72 consultas de carretera y 77 Slider
es desarrollo expuesto, no confirmación ni sustituto de las 2.608 consultas.

Si el screen lo justifica, ejecuta la población completa y las comprobaciones
eAP con historias recalculadas. Si falla, conserva el resultado y decide la
siguiente hipótesis por el diagnóstico; no reinicies una búsqueda ciega de
fusión ni la receta JEPA cerrada. Usa tres semillas y splits por grupos para
cualquier candidato aprendido. No ajustes con las ocho grabaciones solapadas
del benchmark ni con Slider-750/1000.

Sigue abierta la discrepancia oficial de Slider-750. No llames oficial al
17,25 % local ni afirmes SOTA por las pruebas de consistencia. No envíes correo
a organizadores sin que te lo pida. No cambies el modelo entregable hasta pasar
los controles y documentar la promoción.

Entorno: .venv-transfer/Scripts/python.exe, PYTHONPATH=src;. y PYTHONUTF8=1.
Lee los comandos y la configuración Pyright correcta en el documento 02.
Los binarios/cachés están fuera de Git y se reutilizan localmente. Si faltan,
identifica la ruta y el artefacto concreto; no finjas resultados ni empieces
descargas masivas. No sobrescribas carpetas selladas.

Trabaja con autonomía dentro de este alcance, mantén el mismo objetivo y
comunica avances concretos y ETAs medidos. Guarda código, configuración,
semillas, hashes y resultados positivos/negativos; actualiza los Markdown y
haz git add, git commit y git push de código, documentación y evidencia pequeña.
No añadas datos crudos ni checkpoints voluminosos a Git.
```
