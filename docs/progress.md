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
- iniciada la confirmación grouped-CV A0/A1 desde inicialización común sin
  contaminación SSL entre folds;
- `210 passed` y Ruff limpio.

El screen corto había dado un orden engañoso: A1 necesitó 20 épocas para su
mejor checkpoint. La geometría bbox mejora algo el error relativo, pero no el
score compuesto; STRTTC tiene cobertura 27/40 y no se promueve.

El Benchmark-10 permanece sellado.
