# 7. SOTA: fuentes y límites

Investigación web a 8 de octubre de 2026; no se certifica una búsqueda exhaustiva ni
un ranking oculto. Fuentes primarias verificadas:

- eAP/Garl: https://arxiv.org/html/2603.16303v1
- Repositorio oficial: https://github.com/NAIL-HNU/Garl-TTC
- Referencia actual upstream fijada en SOURCE_PIN.json.
- EvTTC: https://arxiv.org/html/2412.05053v2
- REACT: https://arxiv.org/abs/2609.19204 (resumen público recuperado; no reproducción).
- RAFT: https://arxiv.org/abs/2003.12039 (base conceptual de mantener correlaciones,
  no método TTC ni prueba de que nuestra propuesta funcione).
- CodaBench: https://www.codabench.org/competitions/17289/
- CodexCLI: https://developers.openai.com/codex/cli/reference/

Garl/eAP es la comparación directa de objetos. Publica66,2MiD event-only LHR y45,0
full RGB+eventos; no se comparan con119MiD OLD_DEV ni1,3sMAEEvTTC. Su tabla de ablaciones
muestra que añadirRGBporfusión temprana no garantizó una mejora; la fusión tardía y
representación de foreground ayudaron. Nuestra fusión tardía no es por ello novedosa:
la contribución a demostrar sería la representación, entrenamiento y/o resultado nuevo.

REACT (septiembre2026) reporta9,59%RTE y4,6ms en un sistema sin ROI de objetivo. Su
protocolo y scope no son los946queries con cajasoracle aquí. No competir usando unidades
incomparables ni adoptar una SNN como cambio de nombre de H8. El trabajo EV-TTC de2025
sobre TTC denso tampoco es el mismo dataset EvTTC de32escenarios.

La consulta al sitio CodaBench sólo expuso la página general en este entorno: no se
verificó una tabla actual de posiciones. El agente puede leer reglas públicas, no hacer
submissions ni optimizar modelos contra scores oficiales antes de autorización final.

## Claims permitidos por etapa

B: fusión exploratoria entrenada en TRAIN40, transferencia en desarrollo expuesto.
D: comparación de modelos con entrenamiento que excluye grupos DEV, no confirmación
fresca del proyecto. E/R: modalidades de inferencia separadas y exacto scope de crops.
KD: eventos en inferencia, RGB privilegiado durante TRAIN.
F: modelo finalentrenadoTRAIN40, pesos y exportador congelados; no ganador hasta scoring.
Oficial: sólo una comparación equivalente y un score verificado permiten «SOTA en
<dataset/tarea/modalidad/inputcontract/fecha>»; no SOTAuniversal ni promesas AEB.

Ninguna baja de RMSE dominada por un outlier basta para probar mejor percepción general.
Informar mediana, signo, colas y soporte nativo/común; no ocultar que el contexto de H8
es más largo. El presupuesto/fuente de pretraining se declara para todos.

## Alcance de la revisión de este paquete

Se leyeron los informes adjuntos completos y los módulos indicados en SOURCE_PIN.json
mediante GitHub conectado. Falló la clonación por DNS; no se ejecutaron pesosTRAIN40,
CUDA, medios eAP/EvTTC o todos los testsdelrepo. Los cálculos nuevos son aritméticos sobre
per-sequenceMAEtranscrito y testsCPUdelcódigodereferencia. La lectura pública de eAP fue
HTML; el PDF grande no pudo renderizarse y el screenshot de EvTTC falló. No se afirma una
inspección visual completa de esos PDFs.
