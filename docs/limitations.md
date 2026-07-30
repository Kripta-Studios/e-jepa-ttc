# Limitaciones

- El resultado BASE pertenece a un split histórico; grouped CV v6 está
  ejecutándose.
- Dense Patch gana en una confirmación matched de un solo split; aún necesita
  cinco folds y tres seeds.
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
- eAP local no ofrece TTC oficial.
- Benchmark-10 no se ha ejecutado.
- No existe validación para uso de seguridad real.

Estas limitaciones impiden afirmar SOTA, tiempo real certificado o
generalización a conducción abierta.
