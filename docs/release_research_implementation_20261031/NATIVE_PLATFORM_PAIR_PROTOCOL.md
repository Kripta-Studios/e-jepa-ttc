# Sensibilidad nativa emparejada en la plataforma disponible

La ampliación 2608 admite F0. El perfil original CUDA pasa la tolerancia TTC,
pero falla la paridad de tensores históricos, y ninguna de las cuatro políticas
TF32 la recupera. El original local se reproduce exactamente. Estos resultados
permanecen como fallos de la admisión histórica; no se relaja ninguna tolerancia.

Se añade una pregunta distinta y limitada, antes de ejecutar el candidato nativo:
¿qué cambia la corrección de cadencia respecto a la ejecución original sobre
esta misma plataforma? El usuario solicita continuar con los datos existentes.
No hay nuevas adquisiciones ni calibración adicional. Esta prueba no demuestra
conservación de precisión oficial, no abre targets privados, no cambia R0 y no
autoriza promoción. No renombrar el original local como réplica histórica exacta.

Fijar CUDA local, TF32 matmul/cuDNN desactivado, benchmark desactivado,
algoritmos deterministas, lote uno y una sola captura completa de los productores.
Usar las 5805 consultas físicas y todas sus observaciones/historias declaradas,
sin seleccionar las que cumplan paridad. Los 6762 aliases no son muestras
independientes. Reutilizar los runners de captura original y comparación histórica
sin el flag de candidato; guardar todos los fallos históricos a población completa.

Esta ampliación diagnóstica se aparta explícitamente de la ruta condicional
original, que cerró antes de la captura completa por fallo de paridad. Su código
y su resultado no se reescriben. La nueva continuación se sella con este protocolo
y fuentes antes de capturar. La captura usa el mismo contrato original de datos,
pesos y lotes; la excepción es continuar para medir sensibilidad local, sin
afirmar la conservación histórica que aquella ruta exigía.

Antes de la comparación local, exigir el replay exacto de PAIR/PHASE17 original
frente a toda la nueva captura y de cada contexto/cabeza frente al replay original
local completo. Después aplicar solo el mismo CanonicalPair ya fijado en 2608,
reconstruir PHASE17 y todos los descendientes. Exigir identidad exacta de eventos
y de su contexto, dado el reloj nominal observado. Conservar fallbacks/rechazos.

Reportar consultas exactas, ausencias, cambios de signo, distribución y máximo
del cambio TTC, aliases y fallos históricos originales, separando los de TTC de
los intermedios. Un cambio pequeño no acredita score conservado. Sin targets
nativos autorizados no se calculan Overall/Crucial/Negative ni métricas inventadas.
Los tiempos de captura/replay instrumentados no son latencia de servicio completo.
