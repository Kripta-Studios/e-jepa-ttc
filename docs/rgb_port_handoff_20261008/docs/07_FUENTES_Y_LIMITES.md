# Fuentes y alcance

## Fuentes del usuario

Base: https://github.com/Kripta-Studios/e-jepa-ttc/tree/b44443ae331b5e84b94e940801468449b2858de4

Archivos recuperados mediante GitHub conectado:
- docs/sota_campaign_20261008/README.md
- docs/sota_campaign_20261008/NEXT_DECISION.json
- docs/sota_campaign_20261008/evidence/root_qa/full_outlier_audit/COMMON_BOUND_SENSITIVITY_POSTHOC.csv
- src/e_jepa_ttc/models/causal_scale_ttc.py (config, arquitectura, soporte y forward)
- src/e_jepa_ttc/training/causal_scale_eap.py (controles y contratos de entrenamiento)

ZIP de entrada conservado: evidence/SOTA_ESSENTIAL.zip. Verificación independiente de bytes/CRC/manifest y reproducción de tres scorers en evidence/review. No inferencia nueva de pesos ni entrenamiento. No se afirma revisar todos los archivos de todas las ramas.

## Referencias públicas consultadas

- eAP/Garl: https://arxiv.org/html/2603.16303v1
- Garl release: https://github.com/NAIL-HNU/Garl-TTC (revisión histórica 256661242b8a7f5e56aa3c1c02348b30f6e89de6)
- FCWD/Event-aided TTC: https://nail-hnu.github.io/EventAidedTTC/
- EvTTC: https://arxiv.org/abs/2412.05053
- REACT (resumen, no reproducción): https://arxiv.org/abs/2609.19204
- Codex CLI: https://developers.openai.com/codex/cli/reference/

La lectura pública no establece la genealogía completa del checkpoint ni una equivalencia entre los protocolos de eAP, EvTTC, FCWD y REACT. Los resultados sobre TTC relativo o dense/full-field no se ordenan como si todos fueran MiD object-level.

## Interpretación de SOTA

La física de expansión se traslada a RGB y Garl ya demuestra una familia de ese uso. Nuestra nueva hipótesis es si la combinación compacta de transporte, refinado temporal y dos modalidades mejora una comparación igualada. No se presenta esa hipótesis como una técnica jamás vista ni como un resultado obtenido.

No se garantiza que añadir RGB, más parámetros o preservar la arquitectura event supere al mejor comparador. La campaña está diseñada para producir un resultado útil positivo o negativo antes de escalar o distilar.
