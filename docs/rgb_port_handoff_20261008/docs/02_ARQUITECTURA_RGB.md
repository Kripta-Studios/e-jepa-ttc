# Reutilizar el método, no confundir entradas ni pesos

## Hallazgo en el código exacto

`src/e_jepa_ttc/models/causal_scale_ttc.py` declara:

```python
modality: Literal["event", "rgb"] = "event"
in_channels: int = 12
```

El mismo módulo implementa foreground diferenciable, razón de alturas, matching, residual antisimétrico y TTC. Su `_sensor_support` ya diferencia densidad de eventos de una heurística de contraste/exposición RGB. La rama RGB no certifica calibración de confianza; incluye variación entre canales, por lo que una imagen plana de color puede obtener soporte distinto de cero.

A5 y C2F deben identificarse por la configuración de sus checkpoints. La ruta C2F usa la familia de transporte de pirámide adaptativa; copiar esa configuración exacta, no crear una clase ajena ni asumir que su nombre significa otro encoder completamente distinto.

## Primera adaptación

```python
from dataclasses import replace
rgb_cfg = replace(event_cfg, modality="rgb", in_channels=3)
rgb_model = CausalScaleTTC(rgb_cfg)
```

Este es el cambio arquitectónico nominal. NO equivale a cargar estrictamente los pesos event: la primera convolución y la distribución de todas las features son distintas. En el experimento inicial se entrena desde nueva inicialización por semilla; no promediar/repetir kernels event como inicialización oculta. Si una puerta de bins event está activa, no trasladarla ni desactivarla silenciosamente: el protocolo base no la admite.

Reutilizar el código existente en vez de duplicar la CNN. El máximo residual, geometría, radios y capas se mantienen de las recetas A5/C2F verificadas. Las diferencias necesarias de modalidad se guardan en RGB_PORT_DIFF.json.

## Trayecto de modelos

- R_A5 y R_C2F: reciben [B,T,3,H,W], T>=2, idealmente tres frames reales. Encoder y módulos geométricos se entrenan.
- R_PAIR: recibe token128 de R_A5 congelado, delta_t, log(delta_t), 1/delta_t y dos soportes RGB. Es una nueva cabeza de 133 entradas; no cargar la PAIR event.
- R_H1 y R_CTX: misma estructura de refiner GRU160 con nuevas features RGB y nuevo normalizador. H1 significa una observación experta (varios frames), NO una sola imagen.
- E_H1/E_CTX: controles con los productores event admisibles, sin RGB en inferencia.
- F_TRUE/F_ZERO: dos streams con GRU160, uno por modalidad y sus propios tiempos. Procesar por separado los historiales E y R, concatenar sus estados finales y emitir fase. F_ZERO conserva capacidad, máscara y metadatos disponibles, pero anula las features RGB antes de su proyección y las fases RGB del readout. No darle el TTC de un experto RGB por otra ruta.

La fusión usa como anchor la mediana de las tres fases EVENT actuales producidas por expertos entrenados sólo en P. No usar salidas in-sample de E_H8 entrenado en H para ajustar otra cabeza sobre ese mismo H. La fusión y los refinadores se entrenan directamente sobre las features expertas de H, por separado y desde inicialización nueva. La cabeza de fusión no recibe embeddings arbitrarios de productores distintos.

Cuando falta RGB, el wrapper invoca el E_CTX congelado correspondiente y devuelve su output exacto. Entrenar y probar el modo disponible; la cobertura RGB se reporta aparte y el fallback no convierte al modelo RGB puro en disponible. F_ZERO es un control, no una opción a promover retrospectivamente por el mejor score.

## Esquema R_PHASE17

2 estadísticas de RGB crudo: media de luminancia y log1p de gradiente espacial absoluto medio. Después, tres diagnósticos de R_A5 y tres de R_C2F (flow, margen, log varianza), tres fases expertas, tres desacuerdos firmados y sus valores absolutos.

Mantener nombres, hash de esquema y normalizador distintos de PHASE17 event. Igual anchura no significa semántica equivalente. El refiner de 17 entradas puede reutilizar su clase, NO sus pesos event ni sus medias/varianzas.

El soporte RGB del encoder necesita valores [0,1]. No introducir el tensor normalizado con ImageNet en esa heurística. La CNN compacta original usa sus normalizaciones internas. La representación del teacher DINO puede tener su propia normalización, pero es una ruta TRAIN separada. Verificar el dtype de su adapter: si espera uint8 y divide por255, no pasarle imágenes [0,1] para dividirlas por255 otra vez. Conservar el RGB original uint8 para esa ruta.

## Física que sí se traslada

Si h=fH/Z, el cociente de alturas cancela f y H bajo la aproximación correspondiente. El fenómeno proyectivo no depende de si observamos eventos o RGB. Sin embargo, visibilidad, sombras, blur, rotación, aparente foreground y cambios de exposición sí alteran lo que podemos estimar.

Para h0,h1 separados por dt:

T_previous = dt/(1-h0/h1)
T_current  = dt/(h1/h0-1)

No corregir el benchmark a partir de estas ecuaciones sin identificar el anchor contractual de la etiqueta. La fase canónica y TTC emitido deben seguir el contrato existente. Restar microsegundos como enteros antes de convertir los deltas.

## No hacer ahora

No replicar RGB cuatro veces para fabricar 12 canales de eventos. No dar conteos event ficticios. No poner en producción la antigua ObjectEventRGBFusion como si fuese compatible con A5/H8. No abrir una tercera gran arquitectura FPN/ResNet en esta misma cuota. La propuesta previa no se declara negativa: queda aplazada si no empezó; si está entrenando, se conserva su estado y se coordina, nunca se mata para empezar de cero.
