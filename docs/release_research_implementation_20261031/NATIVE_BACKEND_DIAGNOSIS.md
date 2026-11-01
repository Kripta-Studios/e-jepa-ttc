# Diagnóstico condicionado de la política numérica del original

Solo ejecutar si la ampliación admite F0 y la continuación termina porque el
perfil CUDA original falla la paridad histórica. Esperar su COMPLETE. No ejecutar
en paralelo con population2608 ni con captura/replay nativo. Este diagnóstico no
relaja las tolerancias fijadas ni ejecuta el candidato.

Los archivos preservados identifican Linux/RTX 4090, mientras este host usa
Windows/RTX 5070 Ti Laptop. No se conoce con certeza la política TF32 activa en
el proceso original. Para separar esta variable del cambio PAIR, comparar las
cuatro combinaciones booleanas de allow_tf32 para matmul y cuDNN, en este orden:
false/false, false/true, true/false, true/true. Mantener benchmark desactivado,
algoritmos deterministas, lote uno, mismos pesos, decodificador, medios y dos
consultas físicas del perfil. No seleccionar consultas por el error observado.

El primer brazo debe reproducir exactamente el perfil CUDA local que ya existe,
incluyendo contextos y cabeza. Si no lo hace, detener y conservar el fallo de
repetibilidad antes de interpretar los otros brazos. Comparar cada brazo con
todos los campos históricos, con las mismas tolerancias de replay; guardar
también PHASE17 crudo y expertos por observación. Nunca leer targets Test12.

El ensayo solo localiza una discrepancia del original. Si alguna combinación
alcanza paridad, una captura completa posterior tendría que fijar explícitamente
esa política y repetir la admisión del original antes de tocar el candidato.
Si ninguna la alcanza, documentar que esta explicación concreta es insuficiente;
no declarar que todo el error procede del hardware ni ampliar tolerancias.
