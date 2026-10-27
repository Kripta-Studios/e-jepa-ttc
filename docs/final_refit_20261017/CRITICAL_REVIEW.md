# Revisión crítica del refit y de las reglas aportadas

**Actualización:** la comparación explícita entre familias está en
`FAMILY_COMPARISON.md`. La preferencia V16 es provisional y metodológica; no
equivale a que sea la arquitectura de mayor precisión demostrada. La elección
de familia y el coste del esquema OOF son decisiones distintas.

La información aportada mantiene PHASE_ONLY como familia principal, pero cambia
la recomendación sobre las entradas de sus capas superiores. Se prefiere preparar
una construcción OOF por grupos frente al stacking secuencial in-sample. Es una
decisión metodológica para reducir sobreajuste, no una demostración de que el
resultado oficial vaya a mejorar.

## Qué cambia y qué se conserva

Se conserva el V16 original y su ZIP exacto, V18 cerrado, los resultados V19 y la
semilla 7. No se repiten L0/LW, routers, gradientes ni comparaciones con pesos ya
evaluados. Los dos encoders event A5/C2F finales ya completaron TRAIN40 y han
superado la comprobación de reutilización. Reentrenarlos desde cero no añade
exposición a esta receta.

La propuesta head-only H+V histórica no es el trabajo implementado ahora. Existen
localmente nuevos adaptadores para la cadena completa, aunque su entrenamiento
científico no ha empezado. También existe localmente el informe de 46 consultas
EvTTC en `docs/post_v18_transfer_20261017/REPORT.md`; que no esté en los HEAD
remotos revisados no implica que no tenga evidencia local. No se mezcla con la
cohorte histórica de 43 consultas.

Mi rechazo inicial del cross-fitting completo por coste y por el resultado del
GROUP_CROSSFIT de V16 era demasiado concluyente. Ese experimento sólo rotaba
contextos superiores con padres inferiores fijos; no contrastó esta propuesta
anidada. Aun así, el OOF completo no es obligatoriamente más preciso: mantiene
un cambio de distribución entre productores parciales y productores finales.

## Esquema preferido y precisión imprescindible

Cinco bloques de ocho grupos: B0 conserva V, B1 conserva H y B2/B3/B4 reparten P
con un algoritmo determinista que equilibra números de consultas, sin leer TTC
ni scores. Cada rotación utiliza 24 grupos para productores y PAIR, ocho para
refinadores y ocho para generar entradas con toda su genealogía excluida.

Cada PAIR se entrena con su propio A5. **No se mezclan los token128 de encoders
de distintos folds**: sus coordenadas latentes aprendidas no tienen una alineación
garantizada. La frontera OOF adecuada es PHASE17, cuyos campos tienen significados
físicos y diagnósticos comunes. La cabeza PHASE_ONLY utiliza veinte coordenadas
físicas/de tiempo y excluye los estados ocultos de 160 dimensiones.

Para los refinadores finales se unen PHASE17 **sin normalizar**, se ajusta un
normalizador OOF global de TRAIN40 y se entrenan E_CTX/R_CTX. Cada normalizador
auxiliar sólo ve los ocho grupos de entrenamiento de su refinador. PHASE_ONLY
recibe las salidas OOF de los refinadores auxiliares. El PAIR definitivo conserva
su entrenamiento sobre el A5 definitivo; su salida frente a las salidas de PAIR
auxiliares forma parte del cambio de distribución declarado.

No se despliegan cinco sistemas ni se selecciona la rotación que mejor puntúe.
Tampoco se convierte este material en validación ciega: la elección histórica de
arquitectura ya utilizó desarrollo, incluidos grupos que ahora generan OOF.

## Cuenta exacta; el presupuesto de tiempo aún no está cerrado

El plan conceptual tiene 49 ajustes: cinco rotaciones de ocho, más nueve
componentes finales. Si se admiten los ocho ajustes P→H→V de V13 y los dos
encoders event finales de V12, quedan **39 ajustes nuevos y 709.458 updates**.
Son **550.934 updates auxiliares más 158.524 finales**, frente a siete ajustes
de la alternativa secuencial. El cálculo se regenera con
`scripts/plan_final_refit_rotations.py` y conserva los grupos y hashes.

La reutilización de la primera rotación requiere un recibo de adopción nuevo:
los documentos históricos no se reescriben como OOF. También puede reutilizarse
su estado V sellado si se demuestra exactamente la exclusión P/H/V. Estos datos
no habilitan la ejecución automática de 39 ajustes.

No se presupuestan estas rotaciones con tiempos batch1 de V19. Falta integrar
entrenamiento, extracción de historiales, características, almacenamiento,
checkpoints y recuperación en un límite de recursos verificable. El runner
all-TRAIN40 actual tampoco debe presentarse como runner OOF generalizado.

## Reglas públicas aportadas por el usuario

El texto de GarlTTC confirma RGB+eventos, predicción escalar por sample_token,
TTC privado y ranking ponderado 0,5/0,3/0,1/0,1. No contiene un límite temporal
de 200 ms ni prohíbe expresamente 650 ms. Tampoco confirma expresamente el uso de
contexto adicional a la ventana anotada. Se declara esta incertidumbre; no se
inventa una prohibición ni se certifica una elegibilidad que no se ha comprobado.

Los 200 FPS son una característica reportada de Garl, no un requisito de envío.
La imagen Docker del evaluador no obliga a entrenar con Python 3.7. La competición
de detección 3D, su formato y su ZIP son diferentes y no se usan para este envío.
El historial de la cuenta sigue sin verificar: no hay navegador conectado en
esta sesión y el acceso web automatizado no devolvió una página fiable de la
competición. No se ha realizado ningún envío.

## Trabajo técnico ejecutado y límite actual

El preflight verificó metadata, cobertura y hashes de checkpoints; la comprobación
de payload completo de cachés grandes se distingue de sus admisiones históricas.
Se están construyendo las cachés RGB faltantes, reutilizables por ambos diseños.
La prueba de CUDA Graphs sufrió MemoryError y no produjo una admisión completa.
Se conserva el fallo; no se afirma una aceleración medida. La ruta prevista
mantiene caché sin pérdida y precarga con límites de memoria, sin activar graphs.

Referencias: [documentación oficial de stacking](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html)
y [release de GarlTTC](https://github.com/NAIL-HNU/Garl-TTC).
