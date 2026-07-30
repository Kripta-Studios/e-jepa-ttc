# Progreso

## 2026-07-30

- reproducido BASE histórico con paridad exacta;
- separado `B0_HISTORICAL_BASE_EXACT` de `A0_MATCHED_GLOBAL`;
- corregida la comparación A0/A1/A2/K1;
- implementada recurrencia delta KDA;
- corregidas fórmulas height/area/affine/event contrast;
- auditado código público Garl-TTC;
- implementados G0–G7 y verificada topología ResNet-50;
- transformada navegación NEU al frame del evento;
- añadida compensación traslacional oracle con profundidad declarada;
- separados los outputs Core/Garl;
- actualizado el protocolo de almacenamiento;
- ejecutado screen Garl ResNet-50 G0–G7;
- ejecutada confirmación Core de hasta 40 épocas;
- promovido A1 Dense/Patch Policy: 15,210 % frente a 16,129 % de A0;
- rechazados A2 AttnRes y K1 Object-KDA en su forma actual;
- implementado y probado un port causal trazable al código STRTTC;
- evaluada geometría bbox causal con calibración train-only y fallback A1;
- completado grouped-CV 5 folds × 3 seeds: A0 seleccionado frente a A1;
- auditado CARLA DVS Looming con loader mmap, manifest y split bloqueado;
- implementado pretraining JEPA CARLA lazy, reanudable y compatible con BASE;
- implementadas evaluaciones validation/test CARLA y transferencia validada a
  grouped CV EvTTC mediante un orquestador único;
- medido el perfil de hardware: batch 24/acumulación 2/8 workers es el más
  rápido de los probes, aunque usa menos VRAM que perfiles más lentos;
- pilotos CARLA→EvTTC cerrados: CARLA-SSL empeora A0 en RTE un 1,72 % y el
  auxiliar TTC sintético un 17,3 % en fold 0/seed 7; no se promocionan;
- inventariado eAP train-40 y seleccionado un piloto firmado de 12 secuencias
  (9 train/3 validation) sin usar EvTTC;
- implementados eAP-SSL/eAP-Geo event-only, HDF5 bajo demanda, checkpointing,
  resume y orquestación completa hacia A0/A1;
- smoke real eAP-SSL completado con validation loss 0,06474, best/last y cero
  exposición a TTC/RGB/EvTTC/Benchmark-10;
- Ruff, 252 pruebas y los smokes eAP SSL/Geo pasan; el PDF fue recompilado y
  revisado visualmente antes del commit.

El screen corto había dado un orden engañoso: A1 necesitó 20 épocas para su
mejor checkpoint. La geometría bbox mejora algo el error relativo, pero no el
score compuesto; STRTTC tiene cobertura 27/40 y no se promueve.

CARLA queda como evidencia negativa: su full no se justifica con el mismo
objetivo mientras ambos pilotos empeoren la transferencia. eAP train-40 está
completo (40 secuencias, 536,64 GiB), pero el piloto usa solo 12 secuencias y
eventos. Compara SSL puro con geometría débil de cajas 3D proyectadas; no usa el
pseudo-TTC derivado como target. Se escala a 40 únicamente si mejora RTE y MAE
en al menos dos folds EvTTC.

El Benchmark-10 permanece sellado. No existe claim SOTA.
