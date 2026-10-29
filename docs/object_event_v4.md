# Object Event TTC v4 — contrato científico y operativo

## Motivación

Las auditorías v3 observaron que poner los eventos a cero mejoraba ligeramente
el resultado, barajarlos o invertirlos casi no afectaba la salida, y eliminar el
vector de movimiento destruía la predicción. El gradiente supervisado hacia el
motion encoder era decenas o cientos de veces mayor que hacia el encoder de
eventos. Además, ninguna muestra auditada disponía de precontexto válido.

V4 no intenta corregir este resultado ajustando pesos sobre el mismo contrato.
Cambia datos, arquitectura, entrenamiento y selección de checkpoints.

## Contrato de datos

Cada muestra contiene tres ventanas causales:

- `t0`: precontexto anterior;
- `t1`: primer endpoint de la pareja TTC;
- `t2`: endpoint donde se toma la etiqueta oficial GarlTTC.

Las tres ventanas se voxelizan en una única ROI cuadrada calculada como la unión
de las tres cajas más un margen configurable. Las coordenadas son idénticas en
los tres pasos. Por tanto, un objeto que crece mantiene ese crecimiento dentro
del tensor y no vuelve a ocupar artificialmente todo el crop.

El tensor es `[3,12,H,W]`. Los doce canales son los diez bins de polaridad más
conteo y tasa de eventos. Los nueve canales de compatibilidad que eran siempre
cero no se almacenan ni se introducen al encoder.

## Frontera de información

La rama event-only y el predictor JEPA reciben exclusivamente eventos. No reciben:

- `observable_motion`;
- cajas dentro de la ROI;
- alturas visibles;
- TTC;
- profundidad o geometría 3D;
- categoría.

La rama motion-only conserva las 18 variables observables únicamente para una
ablación y fusión tardía auditable. La predicción final nunca oculta las dos
predicciones independientes.

## JEPA local

El encoder online produce tokens espaciales para t0/t1/t2. Un predictor local
usa los tokens t0+t1 para predecir los tokens objetivo t2. El target encoder se
actualiza por EMA y no recibe gradientes. No existe el predictor global v3
condicionado por motion embedding.

## Expansión firmada y antisimetría

La variable primaria es `g = delta_t / TTC`. La cabeza event-only evalúa la
secuencia original y la invertida con pesos compartidos. La salida usa la parte
antisimétrica de ambos scores, por lo que invertir t0/t1/t2 cambia el signo por
construcción, no solo por una penalización blanda.

## Evitar el atajo geométrico

Durante las primeras épocas se elimina motion para todas las muestras. Después,
el dropout de modalidad elimina motion en el 50 % y eventos en el 10 %, nunca
ambos. La fusión tardía impone una contribución event mínima configurable.

## Gates de aceptación

Un checkpoint screen debe satisfacer simultáneamente:

- precontexto válido en al menos el 80 % de la caché;
- Pearson de expansión full >= 0,30;
- Pearson event-only >= 0,30;
- balanced sign event-only >= 0,60;
- recall negativo event-only >= 0,30;
- caída de Pearson al borrar eventos >= 0,05;
- caída al barajar eventos >= 0,03;
- error de antisimetría <= 1e-5;
- saturación TTC <= 1 %;
- gate event medio >= 0,40.

El perfil full endurece los gates de dependencia. Si no existe `best.pt`, el
screen ha falsado la arquitectura; no debe relajarse el gate retrospectivamente.

## Orden de ejecución

1. Aplicar el parche sobre el commit exacto declarado por preflight.
2. Ejecutar tests focalizados.
3. Construir la nueva caché screen; la caché v3 no es reutilizable.
4. Entrenar scratch seed 7.
5. Entrenar Level-transfer seed 7 con adaptación 21->12 solo en patch embedding.
6. Comparar ramas full/event-only/motion-only y perturbaciones.
7. Solo tras superar gates, repetir seeds 13/23 y plantear full.
8. EvTTC continúa sellado hasta congelar candidato.
