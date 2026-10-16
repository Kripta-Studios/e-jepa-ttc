# B1: revisión de coordenadas antes de entrenar

Este es el registro B1 vigente. Corrige la supervisión conocida para los centros
receptivos nativos: 4*j+0,5 en el mapa32 y 8*j+2,5 en el mapa16.
La diferencia respecto al origen nominal es1,5 píxeles de entrada.

Se hizo antes de todo update B1, incluidos los técnicos. La prueba CPU nativa
vuelve a pasar once comprobaciones sin CUDA ni pasos de optimizador.
Presupuesto, warps, padre y comparador se conservan. La QA CUDA sigue pendiente.

[El registro v1](../evidence_B1_registration_20261016/README.md) se preserva
como propuesta no ejecutada. Sus fuentes están en Git27cc978 y en el archivo
local admission_revisions/v1_before_coordinate_correction. El código corregido
se fijó en Git2502eaa antes de esta admisión.

[Receta y límites](../B1_EXECUTION.md). MANIFEST.json verifica los tres recibos.
