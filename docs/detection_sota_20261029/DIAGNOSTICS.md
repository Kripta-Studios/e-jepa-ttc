# Diagnóstico S0 del checkpoint del piloto

El análisis se ejecutó sobre los 256 frames de selección y las predicciones label-free
congeladas del checkpoint
`dc0ce304ad1c8a791ef66395be85b09f22526bc11a834472dcb18ce16384e338`.
Las etiquetas raw se leyeron después, en otro proceso. El perfil v2 es local y no es el
scorer oficial de eAP.

La ejecución corregida está ligada al manifiesto público de archivos con SHA-256
`2e65cd7f2600eea55f37f59e9a22d55b5ba0aa50445756fd7aaf01332c2b3ff1` y revisión
`a814a946e60d37f7242358af5f344542177e50c5`. El metadata y los dos archivos de
etiquetas se verifican por tamaño y hash antes de leerlos; una diferencia aborta el
análisis.

El evaluador v2 contabiliza 1.936 anotaciones raw: 1.035 elegibles y 901 ignoradas por
estar fuera del rango provisional de profundidad. De 12.800 predicciones, 11.680 quedan
dentro del dominio simétrico. La AP101 v2 sobre las cinco clases con soporte es
`0.01982172`; el recall class-agnostic a 2 m es `447/1035 = 0.431884`.

En las 447 correspondencias uno-a-uno a 2 m, el error medio de centro es 1,1914 m en el
plano ego XY, 0,3613 m en la vertical ego Z-up y 1,0334 m en profundidad óptica de
cámara. El error en el plano lateral/vertical de cámara es 0,6258 m. Los errores
absolutos medios de dimensiones L/W/H son
1,9808/0,1765/0,3903 m y el error circular medio de yaw es 0,1482 rad. Estos valores son
diagnósticos sobre correspondencias, no métricas oficiales.

Al reproducir el matching por clase a 2 m, 403 propuestas son true positives, 611 son
duplicados, 108 quedan cerca de un objeto de otra clase, una se suprime contra un objeto
ignorado y 10.557 se clasifican como fondo o error de localización. Esta última categoría
no separa fondo real de una propuesta demasiado alejada; hacerlo requiere información
adicional que el output congelado no contiene.

Los oráculos conservan el pool y vuelven a ejecutar el matching completo:

| Sustitución oracle | AP101 v2, clases con soporte |
|---|---:|
| Clase | 0,08653597 |
| Centro | 0,02673897 |
| Dimensiones y yaw | 0,01982172 |
| Ranking | 0,07633992 |

El resultado prioriza una intervención de clasificación/ranking después de ampliar la
cobertura de datos. La sustitución de dimensiones/yaw no cambia AP porque la métrica
primaria local empareja por centro XY; los errores geométricos deben seguir reportándose
por separado. La secuencia `2Y46OkK3ff` alcanza recall class-agnostic 0,4784 y
`JaGC4Q9pgU`, 0,3825, por lo que el fallo tampoco es uniforme entre secuencias.

El output congelado no contiene logits completos, query ID, box2D, incertidumbre de
profundidad ni la descomposición entre prior y residuo. El análisis los marca como no
disponibles. El observador externo posterior sí expuso logits, índice de query implícito
por orden del decoder y la segunda componente de profundidad, entrenada como
log-varianza, manteniendo paridad exacta de predicciones. En 447 correspondencias, la
correlación de esa log-varianza con el error real de profundidad óptica es `-0,1170` y
los errores medios por cuartil aproximado son 1,0612/1,1332/1,0572/0,8807 m. No hay una
relación monótona ni evidencia de calibración de incertidumbre de profundidad. La
correlación `0,3953` observada antes corresponde al error vertical ego Z-up, otra
magnitud; queda conservada solo como diagnóstico geométrico y no como calibración.

El prior geométrico observable `fy·H_GT/h_proyectada` subestima sistemáticamente la
profundidad de cámara en estas correspondencias. El residuo medio
`z_cámara − prior` usando altura completa/recortada es 3,4359/3,1875 m para car y
7,1405/6,6210 m para truck. Esto no es el prior interno del modelo, que sigue sin estar
expuesto. El sesgo justifica estudiar geometría/proyección, pero no demuestra todavía
que una pérdida SPAN-like vaya a mejorar la AP.

La clase objetivo está en top-1/top-3/top-5 para 393/437/445 de las 447
correspondencias. Un control de igual presupuesto que selecciona las mejores 50
combinaciones de 50 queries × 8 clases obtiene AP101 `0,02005787`, frente a
`0,01982172` del decoder histórico. La mejora es pequeña: disponer de logits completos
no reproduce el gran oráculo de clase/ranking sin aprender una regla mejor.

Siguen sin estar disponibles box2D decodificada por query, prior/residuo geométrico y
foreground por query. Los resultados completos, hashes por frame, curvas PR, confusión y
procedencia fail-closed están en
`output/s0_diagnostics/pilot_selection256_v2_camera_axes_fail_closed_v3/`. Las carpetas
anteriores se conservan como historial y no son la evidencia canónica. Esta salida incluye además
`ground_truth_v2.json`, una serialización exacta del GT v2 para que los controles usen
la misma población y hashes sin volver a interpretar las etiquetas.
