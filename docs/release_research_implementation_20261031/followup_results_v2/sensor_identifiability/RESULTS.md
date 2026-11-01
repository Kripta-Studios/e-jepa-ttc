# S0: identificación frente a recuento y textura desconocida

Casos: 24; streams coincidentes: 24.
q distinto, mismo stream: 12 casos.
Coinciden píxel, polaridad y timestamp (tolerancia 1 ps), no solo recuento.

En esta textura log-afín, el incremento contiene el producto q·gradiente.
Cambiar q y escalar inversamente el gradiente deja iguales esos incrementos;
las referencias estáticas iniciales son distintas y no están en el stream de eventos.
Al proporcionar la referencia/calibración conocida se pueden separar las condiciones.
Error máximo q con referencia conocida: 0.033773992 s^-1.

Escalar textura, luz y umbrales juntos también conserva los eventos.
Sin diversidad espacial persiste la ambigüedad entre luz y movimiento.
Este banco no demuestra imposibilidad general: muestra por qué S1 necesita anclaje
o identificación y por qué un auxiliar con textura y umbral libres puede ignorar q.
No habilita S2 ni JEPA y no afirma una mejora del release.
