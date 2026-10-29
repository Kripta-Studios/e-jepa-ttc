# V18: entrega final verificada

El programa autorizado y su entrega portable están completos con **NO_CANDIDATE**.
No hay promoción del sistema ni confirmación SOTA. Las ramas de refit y nuevo
export test12 no se habilitan porque ninguna receta satisface sus gates completos.

- ZIP: `E_JEPA_TTC_V18_FINAL_RESEARCH_20261017_c3f1440.zip`.
- Tamaño: **91,583,756 bytes**, menor que 100.000.000.
- SHA-256: `65e762c560530ddacfc8523e5bd930d127ad4adc1b26403ebbd61545828aa181`.
- Código y evidencia empaquetados: commit `c3f14408babda11f9f9d50ab1e4158992e6f9c19`.
- Integridad: **12.424 archivos originales** recuperados sin pérdida.
- Replay independiente: **48 tablas, 1.354.938 filas y 22 exports**, todos aprobados;
  cero updates nuevos.
- [CI del código empaquetado](https://github.com/Kripta-Studios/e-jepa-ttc/actions/runs/38976381419): **4.260 pruebas aprobadas, 23 omisiones**;
  Ruff, formato, Pyright y sintaxis PowerShell aprobados.

La compresión conjunta aprovecha bytes repetidos sin eliminar consultas ni cuantizar
pesos. Tras extraer el ZIP exterior, `python -B unpack.py RUTA_ZIP CARPETA_NUEVA`
verifica y extrae su payload. El README interior y
[DELIVERY_REPRODUCTION.md](DELIVERY_REPRODUCTION.md) documentan el replay.

El paquete conserva el estado anterior a su creación: sus campos de entrega pendiente
no podían certificar bytes futuros. [El recibo externo](delivery_evidence_20261017/DELIVERY_COMPLETE.json)
une ese paquete exacto con la extracción y el replay ya completados. Esta documentación
de entrega es posterior al commit empaquetado; no modifica el ZIP ni sus resultados.

[El replay completo](delivery_evidence_20261017/INDEPENDENT_REPLAY.json) distingue las
pruebas ejecutadas de las no repetidas. Los checkpoints completos de los testigos
DIRECT/CORR/FROZEN y de resume CUDA no están en el ZIP compacto: sus verificaciones
históricas se conservan, pero no se presentan como repetidas desde esta extracción.
Tampoco se reproduce todo el sensor crudo ni se reentrena. Los pesos portables y sus
ejemplos sí se cargaron y ejecutaron.

La contabilidad final conserva **183.200 científicos, 858 técnicos y 273 de recuperación
superior**, 184.331 físicos superiores, cero pendientes. Los **196.739,722 s de scopes
GPU** incluyen siete reservas históricas conservadoras; no son kernels puros.

La [conclusión científica detallada](FINAL_SCIENTIFIC_REPORT.md), los
[resultados A1](A1_RESULTS.md) y las [tres semillas de fusión](FUSION_REPLICATION_RESULTS.md)
conservan las mejoras, regresiones y límites. H/V siguen siendo desarrollo, los padres
históricos son compartidos y no existe nueva puntuación oficial.
