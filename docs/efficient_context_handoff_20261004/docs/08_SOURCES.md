# 8. Fuentes y alcance de verificación

Las fuentes del proyecto se fijan en SOURCE_PIN.json; los informes originales están
en evidence/. La estadística procede de esos informes. La lectura de los índices
T6 y la revisión de redundancia de ventanas sí son cálculos independientes de esta
sesión. No se ejecutó la última ruta CUDA ni se reprodujeron aquí las 24 cabezas
nuevas. Los recibos publicados no se atribuyen a esta revisión.

Se verificó el ZIP T6 ya adjunto: SHA256, CRC y 1.995 entradas del manifiesto.
ACCOUNTING, EXTRACTED_VERIFICATION y el informe GPU adjuntos coinciden con los hashes
remotos recuperados. No se pudieron descargar los seis ZIP enlazados ni clonar el
repositorio por errores de acceso de red. El script de descarga/verificación local cubre esa limitación.

Fuentes primarias:
- eAP/Garl: https://arxiv.org/html/2603.16303v1
- Garl release: https://github.com/NAIL-HNU/Garl-TTC
- Configuración event-only: https://github.com/NAIL-HNU/Garl-TTC/blob/256661242b8a7f5e56aa3c1c02348b30f6e89de6/configs/ablation/event_lhr.yaml
- EV-TTC: https://ieeexplore.ieee.org/document/10979412/
- EV-TTC código: https://github.com/anthonytec2/EV-TTC
- REACT abstract: https://arxiv.org/abs/2609.19204
- Slow features: https://arxiv.org/pdf/2211.10831
- MotionJEPA: https://github.com/mkarmann/motion-jepa
- FAR: https://github.com/sony/far
- PixelUMM: https://arxiv.org/abs/2609.38597
- HDF5: https://docs.h5py.org/en/stable/high/dataset.html
- Codex CLI: https://developers.openai.com/codex/cli/reference/

El PDF de 2022 y su página de figura se inspeccionaron. Los PDF/HTML recientes de
MotionJEPA, FAR, PixelUMM y REACT no se recuperaron en los intentos actuales. Las
notas de los tres primeros utilizan fuentes oficiales recuperadas previamente en
esta conversación, no una supuesta lectura íntegra de apéndices. REACT se valora
como resultado reportado en su abstract. La búsqueda no es un inventario exhaustivo
de todo el SOTA. No se abrió CodaBench ni una evaluación privada.

El texto de contacto se conserva como idea propuesta, no como autoridad automática.
Su estado de perfilado 1.114/1.728 quedó obsoleto: la entrega completa acredita 1.728.
No volver a ejecutar las 614 medidas ni H16 basándose en el documento anterior.
