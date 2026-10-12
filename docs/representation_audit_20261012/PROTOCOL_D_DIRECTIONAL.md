# Cola D: bypass de cambios temporales

Se registra después de la cola A y del diagnóstico C, antes de entrenar D o abrir sus nuevas predicciones V. V sigue siendo desarrollo expuesto, sin pretensión de confirmación independiente.

En los24 ajustes ridge de C, las diferencias temporales mejoran MAE phase fuera de grupo respecto a PHASE17 en8/8 grupos H y respecto a vectores completos en5/8. La ganancia media frente a PHASE17 es pequeña (≈4,1%) y la crucial apenas cambia. Esto justifica probar la señal temporal con H8; no demuestra que vaya a ganar ni que haya identificado causalmente un atajo de apariencia.

Se entrenan D-S7/D-S13/D-S23: cada uno2500updates. Misma arquitectura896→160 del bypass G, última capa cero, mismos parámetros iniciales y RNG, AdamW, batch128, M0/L0, FP32 GPU, scheduler y normalizador PHASE17 H nativos de A. Se reutilizan los controles Z/G congelados de A, cuyas fuentes y condiciones deben coincidir. No se vuelve a entrenarlos para elegir mejores semillas.

Único cambio de representación extra: concatenar dos diferencias de geometry_tokens (256 coordenadas) y dos diferencias forward_transport−reverse_transport (256); ajustar mean/std exclusivamente en H; rellenar con384 ceros para conservar la interfaz896 y la arquitectura. Ningún token ni target de V entra en el entrenamiento. Este cambio elimina información estática extra y reduce dimensión efectiva; no se atribuirán por separado ambos efectos.

Los tres D se congelan antes de su V. Gates de promoción: todos los gates originales V14 contra Z pareado en cada seed y contra V14M0 histórico en seed7; además MiDc D debe ser menor que G en cada seed. No se baja el objetivo del10% ni se rescata la mejor semilla. Coste E2E sigue siendo requisito antes de promoción del sistema, distinto del perfil de cabeza.

Presupuesto nuevo de esta cola:7500 updates científicos,20 sintéticos de continuidad, pertenecientes al techo global V17 de1M/18h GPU. No cambia B, sus pesos, su precisión o su selección H. No hay encoders/PAIR nuevos, RGB, FCWD/Dev32 fits ni submissions.
