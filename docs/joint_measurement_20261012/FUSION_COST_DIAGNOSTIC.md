# Fusión: diagnóstico de la paridad sensorial

**El coste sensorial completo sigue sin admisión.** Se conserva el fallo de paridad frente a la cohorte almacenada y no se amplían las tolerancias. Los pesos, predicciones históricas y resultados de los tres seeds no se modifican.

El primer intento se detuvo en la admisión de fuentes históricas. La siguiente ruta cargó las definiciones nativas originales, pero encontró una mediana CUDA sin implementación determinista. Una política explícita y acotada permitió ejecutar esa operación conservando el warning. La prueba posterior falló en la primera consulta cronometrada: ZERO y F_TRUE no cumplen `rtol=1e-5, atol=1e-4 s`. Sus tiempos parciales no forman una medición completa.

## Localización sobre cuatro consultas fijadas

Se usan las primeras cuatro consultas del orden original de coste, sin selección por etiquetas ni cambio de la población. El diagnóstico no ejecuta optimizador.

Pixels P2 y separación temporal raw/preparados idénticos: **True**. C1 con el batch8 original reproduce features y TTC almacenados exactamente: **True**. Por tanto, en estos casos el cache sensorial P2 conserva los datos y la ruta de cohorte puede reproducirse.

| Índice V | Δ máx. features C1 batch1/cohorte | Δ máx. R_HISTORY | Δ máx. E_HISTORY | Δ anchor RGB, s | Δ F_TRUE raw/cache, s |
|---:|---:|---:|---:|---:|---:|
| 16169 | 8.77380371e-05 | 0.000770509243 | 0.00179815292 | 0.000158786774 | 0.00116825104 |
| 7651 | 3.05175781e-05 | 0.00121116638 | 0.00082886219 | 0.000741958618 | 0.00219297409 |
| 1762 | 3.52859497e-05 | 0.00118476152 | 0.00210994482 | 1.71661377e-05 | 0.000335216522 |
| 1515 | 3.81469727e-05 | 0.00113248825 | 0.00757494569 | 0.000367641449 | 0.00482702255 |

Los relojes y máscaras históricas coinciden; las features y fases de los productores raw presentan diferencias. C1 también cambia ligeramente al pasar de batch8 a batch1. Estas son diferencias numéricas observadas, no una demostración de fuga de datos ni una licencia para aceptar el coste.

El control CPU del anchor RGB reproduce la cohorte con batch128. Cambiar únicamente a batch1 introduce como máximo 9.53674316e-07 s en esas cuatro consultas; por sí solo no explica las diferencias de la ruta raw.

## Hipótesis de backend registrada

El extractor histórico y su constructor fijan determinismo, pero no fijan `cudnn.allow_tf32`; el setup V18 lo desactiva explícitamente. Se registra un diagnóstico de ambos valores sólo en las convoluciones de los productores históricos, manteniendo matmul TF32 desactivado y restaurando la política al salir. No cambia las convoluciones C1 ni los pesos. El flag efectivo histórico no quedó registrado: esta es una hipótesis verificable, no una causa ya demostrada por inspección del código.

Estado del contraste de backend: registrado, pendiente de GPU exclusiva.

Diagnóstico GPU: `66975efa18b0cb61cf2db50e81b8207a2886f6c7ca60502536267b7146ccea97`.
Diagnóstico CPU: `8ebc05adac761f087d3d3ab18c4e994d24a333b596552d644c179ee3b6d4c137`.

No se infiere precisión TTC de estos cuatro casos ni se utilizan para elegir un modelo. La inferencia preparada y los exports siguen teniendo un alcance distinto del coste sensorial completo.
