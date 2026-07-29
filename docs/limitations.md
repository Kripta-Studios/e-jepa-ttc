# Limitaciones

- El resultado BASE pertenece a un split histórico; grouped CV v6 está
  pendiente.
- Dense Patch, AttnRes y KDA tienen únicamente smoke de integración.
- Garl local adapta la altura visible EvTTC y no reproduce los targets eAP
  originales.
- El bbox assisted no demuestra localización bbox-free.
- La compensación traslacional necesita profundidad; distancia GT implica
  oracle.
- El router puede memorizar familias con solo 32 secuencias.
- eAP local no ofrece TTC oficial.
- Benchmark-10 no se ha ejecutado.
- No existe validación para uso de seguridad real.

Estas limitaciones impiden afirmar SOTA, tiempo real certificado o
generalización a conducción abierta.
