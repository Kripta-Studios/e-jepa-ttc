# Limitaciones

- El resultado BASE pertenece a un split histórico y no se mezcla con grouped
  CV.
- Dense Patch gana en una confirmación matched de un solo split, pero pierde
  grouped CV de cinco folds × tres seeds frente a A0.
- AttnRes y KDA no mejoran A1 en la confirmación larga actual.
- Garl local adapta la altura visible EvTTC y no reproduce los targets eAP
  originales.
- El screen Garl es mucho más corto que las 50 épocas y el pretraining por
  ramas del código oficial.
- El bbox assisted no demuestra localización bbox-free.
- La geometría bbox usa hasta 21 observaciones frente a tres frames del modelo
  neural; no es una comparación de contexto equivalente.
- El port STRTTC solo cubre 27/40 muestras del screen y sus métricas
  success-only son insuficientes para promoción.
- La compensación traslacional necesita profundidad; distancia GT implica
  oracle.
- El router puede memorizar familias con solo 32 secuencias.
- eAP train-40 está completo, pero no ofrece TTC oficial; su pseudo-TTC solo
  cubre el 24,24 % de las filas y no es ground truth.
- CARLA es sintético, está cuantizado a 10 ms, carece de bbox temporales y su
  TTC positivo llega solo a ~3,85 s. El smoke SSL aprende sin colapso, pero no
  demuestra mejora TTC hasta completar la transferencia grouped-CV a EvTTC.
- La preparación CARLA está limitada por lectura/voxelización CPU/SSD; aumentar
  batch o workers llenó más VRAM pero redujo el throughput medido.
- Family-OOD degrada el score un 85,5 %, el error relativo un 111,4 % y el MAE
  un 48,8 % frente a validation.
- El perfil throughput es 4,02× más rápido en entrenamiento, pero empeora el
  score medio un 10,95 %; no es el perfil final de precisión.
- Benchmark-10 no se ha ejecutado.
- No existe validación para uso de seguridad real.

Estas limitaciones impiden afirmar SOTA, tiempo real certificado o
generalización a conducción abierta.
