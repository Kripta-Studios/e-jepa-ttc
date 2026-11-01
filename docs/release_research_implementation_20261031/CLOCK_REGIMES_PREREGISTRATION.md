# Control de reloj bajo cambios de régimen

Banco mecánico prefijado, sin datos reales ni selección de parámetros. Completa
el diagnóstico de respuesta tras frenada del dossier; no admite D4 por sí solo.
Se reutilizan observe y ClockConfig sin modificarlos: historia de 650 ms,
EWMA tau=150 ms, ruido de proceso 0,01/s, ganancia máxima 0,9.

Simular profundidad Z(0)=10 y velocidad -3, con aceleración 0, +6 o -4
desde t=1 s. Los tres futuros comparten exactamente el prefijo hasta t=1.
Añadir un caso de recesión con velocidad +3. No cruzar Z=0. Consultas cada
10 ms hasta 2 s; mediciones instantáneas de q=-v/Z, cadencia irregular fija
[40,140,70,90,160] ms, disponibilidad 25 ms después, hueco [0,7,0,95] s.
Ruido gaussiano de desviación 0 o 0,05 s^-1, semilla 7 y mismas realizaciones
entre futuros. Son medidas puntuales sintéticas, no ventanas RAW superpuestas.

Comparar persistencia, transporte, EWMA y observador; mismas observaciones
disponibles y reinicio por consulta. Registrar error de q, p95, signo, error
durante los primeros 250 ms tras el cambio, primer aviso q>=0,5 s^-1 y demora
para detectar la inversión de signo tras frenada. Los tiempos de cambio solo
sirven al análisis; no se pasan al estimador ni se usan para escoger régimen.

Exigir transporte exacto sin ruido y velocidad constante; predicciones idénticas
antes de divergir los futuros; no afirmar capacidad de anticipar maniobras.
Guardar trazas completas, denominadores y fallbacks. El banco no compara R0,
no valida incertidumbre y no reemplaza el contraste real D1–D3 ya ejecutado.
