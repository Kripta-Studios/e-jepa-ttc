# Controles de evaluación v2

Esta fase vuelve a comparar el checkpoint congelado con dos controles negativos
bajo exactamente la misma población y el mismo protocolo local v2:

1. Un prior constante ajustado exclusivamente con el GT admitido de train del
   piloto. Para cada clase usa medianas de centro y dimensiones, media circular
   de yaw y frecuencia de presencia por frame como score. Emite una caja por
   clase train dentro del límite fijo Q50.
2. La primera permutación sin puntos fijos producida por `random.Random(7)` sobre
   listas completas de predicciones. Conserva cajas, scores y número de propuestas;
   solo cambia el frame receptor.

El bootstrap es emparejado entre los tres brazos y remuestrea secuencias completas
con reemplazo. Nunca trata frames correlacionados como réplicas independientes.
Las clases sin soporte en una réplica permanecen `null`; no se convierten en AP
cero. El CLI expone 200 réplicas por defecto y permite fijar otro presupuesto antes
de leer resultados.

```powershell
python scripts/evaluate_detection_controls.py `
  --train-results <pilot>/final_train.json `
  --selection-predictions <pilot>/inference_label_free_selection256.json `
  --selection-ground-truth <s0>/ground_truth_v2.json `
  --protocol <s0>/evaluation_protocol_v2.json `
  --replicates 100 `
  --seed 7 `
  --output output/s0_controls/pilot_selection256_v2_r100
```

El directorio debe ser nuevo. Antes de puntuar se escribe `preregistration.json`
con hashes, semilla, unidad y presupuesto. Después se generan `controls.json` y un
resumen `CONTROLS.md`.

Estos controles miden señal específica del frame frente a referencias simples.
No prueban causalidad, paridad con el scorer oficial ni SOTA. El piloto tiene solo
dos secuencias de selección; por ello los intervalos por cluster son débiles y
deben publicarse con esa limitación.
