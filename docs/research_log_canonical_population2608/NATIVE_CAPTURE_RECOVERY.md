# Recuperación de captura nativa interrumpida

El 1 de noviembre se comprobó que no existían ni coordinador 44880 ni worker
63932, ni otro proceso con los comandos de captura/replay de este experimento.
El run `native_platform_followup_v1` seguía indicando capture, sin COMPLETE ni
FAILED. El log terminaba en RGB 1474/5805. Se encontraron 7280 carpetas con ambos
archivos escritos: las 5805 observaciones de eventos y 1475 RGB. Los 63 hashes
de fuentes/entradas del FREEZE original siguen coincidiendo. No se infiere una
finalización por un estado obsoleto ni se atribuye una causa no comprobada.

Al reanudar se encontró ya viva la recuperación concurrente
`scripts/resume_release_native_research.py all`, coordinador 35860 y worker 3836,
con salida `output/release_research_20261031/native_platform_followup_v2`.
No se lanzó una segunda captura. Se retiró el borrador alternativo de recuperación
de esta conversación, que no llegó a ejecutarse; no se alteraron los fragmentos.

La recuperación activa usa un directorio nuevo y conserva todo el intento anterior.
Los fragmentos no tenían sello individual durable: no se presentan como si
hubieran estado sellados al interrumpirse. Se inventarían y sellan sus bytes al
iniciar esta recuperación, antes de usarlos. El protocolo concurrente exige
metadata e inventario de arrays, recaptura de centinelas por secuencia/modalidad
con igualdad exacta y replay posterior de PAIR/raw17 en todos los fragmentos.
Los faltantes se capturan con el mismo código, pesos, lote y política numérica.
Los controles posteriores de la cadena completa siguen siendo obligatorios.

No sobrescribir el run anterior, no reutilizar fragmentos de un run FAILED,
no cambiar tolerancias y no evaluar el candidato durante la recuperación.
El nuevo run conserva el formato de captura requerido por los replays posteriores
y añade explícitamente su procedencia de recuperación. Los tiempos no son
latencia de inferencia completa ni el tiempo de una extracción desde cero.

El usuario ha limitado la continuación a cerrar: validación eAP disponible,
decisión del candidato según controles fijados y comparabilidad/SOTA. Después
se detendrá el trabajo. No abrir nuevas líneas ni declarar completos requisitos
que necesiten datos/evaluador ausentes. El objetivo SOTA no se dará por logrado
por cerrar este informe.
