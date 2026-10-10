# 1. Lectura de resultados y diagnóstico

## Qué está terminado

E0 desfavorable (155,442 MiD frente a 121,646 H8). WIDE: nueve fits, 22.500 updates;
media 118,954 frente a 121,784 H8 y 119,241 H16, pero ambos intervalos jerárquicos
incrementales incluyen cero. No aumentar lags ni repetir WIDE para buscar significación.
La rama E3 de Garl nativo conserva 17.948 updates y quedó pausada/supersedida por
orden del usuario. No hay cabezas Garl H1/H8 finales de aquella cola.

TRAIN40 sí entrenó A5 y C2F completos (49.932 cada uno), PAIR (6.840) y tres cabezas
H8 (2.500 cada una): 114.204 updates científicos nuevos. No confundirlo con la
campaña anterior de cabezas sobre encoders congelados. Una semilla de productores,
tres semillas de cabeza. Las 88.744 filas TRAIN40 sirven para entrenamiento/diagnóstico,
no para demostrar generalización.

R1 quedó en 486/768 pares en el snapshot, con 256 pruebas CPU exactas. Descubrir el
estado local actual; no inferir que siga en 486 ni duplicar su supervisor. Cerrar si
la ruta vigente está disponible; no bloquear RGB o entrenamiento D por esa medición.

## Transferencia EvTTC

946 targets válidos de 1.024 consultas, 32 secuencias. Soporte común para cinco modelos.

| Modelo | MAE s | Mediana AE s | RMSE s |
|---|---:|---:|---:|
| H8 seed7 | 1,320841 | 0,766047 | 3,027209 |
| H8 seed13 | 1,352809 | 0,731413 | 3,206100 |
| H8 seed23 | 1,408714 | 0,738464 | 3,392155 |
| Garl event-only | 8,048109 | 0,996929 | 159,246808 |
| Garl RGB+event | 1,868276 | 0,691512 | 7,875241 |

No significan «somos seis veces mejores». El peor Garl E aporta 64,21% del absoluto
y 99,61% del cuadrático; Garl full tiene mejor mediana. H8 está acotado a ±60 segundos,
Garl nativo no. Se necesita publicar nativo y el control de soporte común ±60, sin
sustituir la tabla antigua ni escoger el cap según el score. El cap60 procede del
contrato histórico, no de buscar un umbral favorable.

Los cálculos propios de `evidence/` parten de **MAE por secuencia transcritos** de los
CSV publicados, no de nueva inferencia ni de haber leído todos los targets locales.
Reproducen los cinco MAE agrupados al redondeo publicado. H8 gana a Garl full en
16, 18 y 18 de 32 secuencias. Las medias de pérdidas por familia son exploratorias;
no acreditan ocho adquisiciones independientes. La media de pérdidas de tres cabezas
NO es una predicción ensemble.

Un caso compartido grave: CCRs-1-low-100-overlap-100:28 tiene GT 0,9062 s, H8
11,10/31,02/60,00 s y Garl full 47,96 s. Ni cap60 ni menor RMSE garantizan resolver
los casos peligrosos. Registrar sobreestimación crucial, errores de signo y colas.
No eliminar esa fila ni adaptar su peso usando el resultado externo observado.

## Qué aprende e ingiere el sistema actual

A5/C2F TRAIN40 ya utilizaron `dinov3_local_relational`, peso 8.0, en TRAIN. Son
**event-only en inferencia con supervisión RGB privilegiada durante entrenamiento**.
No son pruebas de entrenamiento puramente sin RGB. La comparación RGB nueva corrió
Garl full, no un encoder RGB nuestro.

`models/multimodal.py` contiene una arquitectura anterior con geometría de bbox
explícita, ObjectCentricEventJEPA y objetivo inverse-TTC. No es la ruta H8 activa ni
prueba de que se entrenara esa variante. No sustituir PHASE17 por esa interfaz.

La hipótesis nueva se apoya en estos límites: información visual nueva puede mejorar
los casos habituales; correspondencias espaciales retenidas pueden permitir un
experto event-only mejor. Ninguna de las dos mejoras está demostrada de antemano.

## Qué no se repite

No reabrir T6, no volver a los 72 fits históricos, no reentrenar A5/C2F TRAIN40 para
reproducir su identidad, no reiniciar los 12 Garl nativos pausados, no insistir en LATENT
sin una hipótesis nueva. Se preservan todos esos checkpoints como referencias.
