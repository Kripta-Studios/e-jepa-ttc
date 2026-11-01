# V0/V1: coordenadas y superficies persistentes

El fallo O0 RAW localiza una limitación de cobertura de los planos temporales:
no habilita una cabeza mayor. Este diagnóstico usa correspondencias RGB para
comprobar si superficies persistentes ofrecen una medición complementaria; no
se presenta como nueva evidencia puramente de eventos.

Se usa la última observación RGB real del screen149: tres frames, mismo recorte
guardado, misma conversión uint8/bilinear128 del release. Se conserva T2 inicial
como falta de soporte para esta prueba, sin fabricar un tercer frame. V0 registra
cuadrado solicitado, crop efectivo entero/clipped, timestamps y el recorte de
cada snapshot histórico. No se introducen etiquetas de TTC ni frames futuros.

V1 fija 49 centros, parches 11x11, NCC>=0,5, diferencia>=0,05 frente al segundo
pico fuera de un vecindario 3x3 y retorno bidireccional<=1px. Búsqueda ±8px en
a→b/b→c y ±16px en a→c. Esta última ruta se estima directamente, no por composición.
Se admiten superficies con diferencia entre rutas<=2px. Se ajusta un mapa afín
con cinco iteraciones robustas y mínimo seis correspondencias no colineales.
Se informa traza espacial, determinante, medio logdet, residuo, cobertura y q
de endpoint. Es un modelo local: giro/paralaje no identifican TTC por esta ley.

La evaluación principal es descriptiva: errores y cobertura sobre todas las
consultas, con fallback R0 declarado para la medida no representable. No se
sustituye el release. No se sintonizan umbrales con los targets. Se contrasta la
relación entre consistencia/cobertura y error dentro de adquisición; una medida
aislada aceptada no acredita ganancia de fusión.

No se abre V2/V3 por menor residuo de correspondencia. Un nuevo gate exige un
contraste incremental prefijado y control de igual coste; este ensayo sirve para
localizar fallo espacial/visibilidad y verificar disponibilidad de medición.
