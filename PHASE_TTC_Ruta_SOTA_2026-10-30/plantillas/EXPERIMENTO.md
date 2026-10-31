# Ficha de experimento — completar antes de ejecutar

**Estado de esta plantilla:** sin resultados. Todos los campos pendientes deben sustituirse por evidencia o por `unknown` con motivo; no inventar valores para completar el formulario.

## Identidad

| Campo | Valor |
|---|---|
| experiment_id / familia | PENDIENTE |
| fecha y responsable | PENDIENTE |
| git_commit / estado del árbol | PENDIENTE |
| baseline / candidato / padres | PENDIENTE |
| hash configuración / modelo / normalizadores | PENDIENTE |
| protocolo / manifest / split hash | PENDIENTE |
| semilla / dispositivo / entorno | PENDIENTE |
| directorio de salida nuevo | PENDIENTE |

## Pregunta e hipótesis

- Observación que motiva el experimento:
- Mecanismo propuesto:
- Única diferencia principal frente al control:
- Resultado que refutaría la hipótesis:
- Qué no podría concluirse aunque mejorase:

## Datos y exposición

| Rol | Grupos | N consultas | Uso histórico | ¿Puede elegir candidato? |
|---|---:|---:|---|---|
| Ajuste | PENDIENTE | PENDIENTE | PENDIENTE | Solo entrenamiento |
| Selección | PENDIENTE | PENDIENTE | PENDIENTE | Sí, según contrato |
| Reserva | PENDIENTE | PENDIENTE | PENDIENTE | No |
| Externo/competición expuesta | PENDIENTE | PENDIENTE | PENDIENTE | No |

Excluir/agrupación aplicada:
Genealogía upstream auditada:
¿OOF necesario? Explicar según si upstream se ajustó sobre estos datos:
Información auxiliar disponible en inferencia:

## Contratos y métricas

- Instante/unidad/geometría del target:
- Máscara de target elegible, fijada sin mirar predicción:
- Tratamiento de signo, no finitos, censura y ausencia:
- Métrica primaria y agregación:
- Margen práctico preregistrado y motivo:
- Límites de regresión por dominio/TTC corto/negativos:
- Cobertura exigida:
- Runtime y frontera de medida:
- Política de desempate:

## Presupuesto y parada

- Configuraciones permitidas, incluidas alternativas de inestabilidad:
- Semillas piloto/confirmación:
- Máximo actualizaciones, épocas o coste medido:
- Selección/early stopping:
- Condiciones de abortar por contrato:
- Condiciones de cerrar por hipótesis negativa:

## Pruebas antes de medir precisión

- [ ] Identidades y grupos.
- [ ] Causalidad y estado entre secuencias.
- [ ] Referencia física y unidades pertinentes.
- [ ] Serialización/carga y paridad cuando corresponda.
- [ ] Cobertura y razones de ausencia.
- [ ] Instrumentación no altera baseline.

## Resultado — completar después

Comando exacto ejecutado:
Salida y exit code:
Recibos/artefactos:
Todos los intentos, incluidos fallidos:
Métricas generadas y diferencias por grupo:
Incertidumbre:
Incumplimientos o desviaciones:

## Decisión

PROMOVER / RECHAZAR / INCONCLUSO / BLOQUEADO_POR_DATOS

Evidencia que apoya la decisión:
Siguiente acción concreta:
Revisión independiente:
