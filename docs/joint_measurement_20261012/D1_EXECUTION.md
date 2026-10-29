# D1: capacidad del predictor espacial

Estado al registrar: cero updates científicos D1; prototipo CPU aprobado;
QA CUDA pendiente. La configuración y la evaluación se han congelado antes de
entrenar. D1 es el segundo experimento adaptativo registrado, después de C1.

## Por qué está habilitado

El apartado D del handoff permite probar anchura ×2 cuando el ajuste P sigue
limitado después de verificar gradientes, loader y correspondencias. En el control
PAIR-CORR-S7, las 53.890 consultas P conservan un error de signo del 17,575 % y un
agregado 75,736 después de 12.000 updates. Su endpoint tiene 18 tensores del encoder
modificados, gradientes finitos no nulos y cursor de entrenamiento verificado.
El diagnóstico de correspondencia temporal ya aprobado muestra señal coherente.

Estos datos permiten **probar** capacidad; no demuestran que sea la causa. CORR
congelado obtiene incluso mejor objetivo P que CORR entrenable. Siguen abiertas
explicaciones de objetivo, optimización o información insuficiente. D1 no presupone
que aumentar tamaño deba mejorar TTC ni usa los resultados V de fusión para escoger
su receta. La evidencia de decisión procede exclusivamente del diagnóstico P/H.

## Contraste único

Se duplica la anchura de la torre espacial (96 → 192) y de la MLP (128 → 256).
La CNN nativa de 64 canales conserva exactamente la inicialización A5 P-only y
continúa completamente entrenable. No se duplican los canales de esa CNN ni se
cambian los inputs, las correspondencias, sus máscaras o los relojes.

El control tiene 578.337 parámetros; D1 tiene 1.693.185. En ambos, 176.192 pertenecen
a la arquitectura del encoder. Anchura ×2 no significa parámetros ×2. El número de
grupos de GroupNorm y las leyes de inicialización son los del predictor original;
los tensores nuevos tienen formas diferentes y no se afirma identidad de sus valores
aleatorios. La variante de anchura uno reproduce exactamente el constructor original.

El control emparejado es PAIR-CORR-S7, ya terminado en 12.000 updates. Se comprueban
su checkpoint, configuración, semilla, masas y genealogía. Se reutiliza sin repetir
entrenamiento ni contar de nuevo su coste. D1 añade un único fit seed7 de 12.000
updates P, batch efectivo 32, microbatch 4, AdamW y schedule originales. Mantiene
la pérdida de fase emitida más 0,1 de fase raw, el decoder firmado y las masas
globales P. No hay nuevos datos, clipping, augmentaciones ni coeficientes buscados.

## Criterios registrados

El endpoint es siempre 12.000. Se congela antes de evaluar P/H completos. Una
mejora P de al menos 3 % del agregado, sin empeorar el signo, se registra como señal
del contraste de capacidad. Se publican también pérdidas y todas las bandas.
Esta señal por sí sola no admite H, no promociona el sistema y no abre V.

La admisión de desarrollo usa el mismo A5-P2-FROZEN y los gates del protocolo:
ganancia agregada, crucial o coste con cobertura, bandas y signo protegidos.
El coste emparejado usa entradas preparadas y se etiqueta como tal. V sólo se
abre si pasa la selección H prospectiva; no se rescata el modelo mirando V.
La única familia seleccionada para réplica sigue siendo la fusión C1: este
experimento no autoriza una segunda familia ni una semilla favorable adicional.

## Implementación y validación

El motor de checkpoints, muestreo, Adam y acumulación es el existente. Un adaptador
de proceso cambia exclusivamente el modelo y los directorios D1; las pruebas
comprueban que restaura el namespace histórico incluso tras una excepción. Los
evaluadores reutilizan las cohortes, scorer, bootstrap y coste existentes con el
modelo D1 y rutas propias; no ejecutan el modelo geométrico C1 como si fuera D1.

La prueba CPU sin optimizador verifica inicialización, copia privada del encoder,
pesos del padre intactos, salidas y gradientes exactos entre el forward original y
su adaptación tensorial, RNG y guard de tiempo. Se conserva un primer fallo del
diagnóstico: comparaba RNG después de un forward adicional de tiempo inválido que
consume dropout. Se corrigió la comprobación, sin cambiar el cálculo del modelo.

Antes de ciencia, la QA CUDA registrada requiere 30 updates técnicos: eager 10,
captura 10 y captura 5 + restauración + 5. Comprueba paridad numérica eager/captura
y reanudación exacta de pesos, Adam, scheduler, RNG, sampler y curva. La nueva
captura se construye antes de asignar los buffers de gradiente propios y recoge
ciclos de grafos anteriores fuera de la captura. No se modifica el runtime de C1.

Se preservan FP32 sin TF32, CUDA Graphs de forward/backward, vistas pinned, caché
sensorial sin pérdida de 8 GiB y una sola precarga. Si la QA falla, se conserva y
se investiga antes de ejecutar D1 científico.

Los registros están en `adaptive/D1_predictor_width/REGISTRATION.json` y
`evaluation/REGISTRATION.json`, dentro de `artifacts/joint_measurement_20261012`.
La prueba CPU y su corrección están en `prototype`; la futura QA está en
`qa/capacity_D1`. Comandos, ejecutados secuencialmente cuando la GPU esté libre:

```powershell
python -B -m operational.joint_measurement.capacity_train
python -B -m operational.joint_measurement.capacity_evaluate
```

El Python debe ser el entorno compartido del proyecto y `PYTHONPATH=src;.` desde
el worktree V18. No iniciar estos comandos junto a otro escritor GPU. Las fuentes
registradas no deben editarse después de comenzar.
