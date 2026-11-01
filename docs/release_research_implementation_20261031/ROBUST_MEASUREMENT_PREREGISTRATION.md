# Control de lectura robusta después de O0

O0 mostró q algebraicamente estimable con signo erróneo y ocho ajustes sin
grados de libertad residuales. Antes de atribuirlo todo a la representación,
contrastar la lectura OLS existente con `fit_affine_normal_flow` histórico.

Usar exactamente los planos guardados de la última ventana, sus coordenadas,
normales y velocidades. No volver a seleccionar parches ni variar parámetros.
Conservar el contrato histórico: mínimo 32 planos, cinco iteraciones IRLS,
umbral de condición 1e6 y fallback isotrópico explícito. Este último introduce
una hipótesis de movimiento; no se llamará observabilidad afín sin prior.

Primero sellar ambos q y sus emisiones sin leer targets. Transportar al tiempo
de consulta cuando proceda, rechazar singularidades y conservar ausencia.
Después comparar q, signo, TTC y cobertura contra los targets de desarrollo
ya expuestos; informar solo-medida y reemplazo con fallback R0, todos los casos.
No cambiar la regla O0, no reentrenar el gate ni sustituir un resultado negativo
por el subconjunto admitido. Es un control de lectura, no un candidato elegido.
