# Comparación con el sistema event completo

El contrato en `system_compare/REGISTRATION.json` admite E_CTX_MATCHED mediante
los hashes de sus cuatro componentes, normalizador H y predicciones V históricas
sin filtrar. La admisión se ejecutó antes de obtener endpoints o métricas nuevas
V18 y no calculó resultados V. Incluye 16.432 consultas originales.

`operational.joint_measurement.system_compare` acepta sólo los candidatos ya
admitidos en `PRIMARY_SELECTION_FINAL.json` usando H, como máximo dos. Exige los
cuatro endpoints primarios congelados, predicciones V completas de los candidatos
y el coste raw E2E emparejado con los mismos padres. No realiza una selección
nueva sobre V.

Se reutiliza sin cambios `operational.range_transfer.analyze.gate`:

- MiDc al menos un 10 % menor y mejora en al menos el 60 % de los grupos cruciales.
- Agregado sin deterioro; por banda s/l/n, aumento máximo de max(2 MiD, 5 %).
- MAE como máximo 1,02 veces la referencia; error de signo como máximo +0,5 pp.
- P95 de sobreestimación crucial como máximo 1,05 veces la referencia.
- Aviso tardío y falsa alerta como máximo +1 pp, con al menos 30 casos en cada
  denominador; soporte insuficiente impide declarar elegibilidad.
- Cobertura completa de predicción y MiD, preservando salidas inválidas.

Las tablas se comparan por identidad y orden exactos, grupos y targets originales
float64. El CSV histórico se lee con conversión round-trip. Los intervalos se
calculan mediante bootstrap emparejado de grupos de adquisición, conservando
las réplicas sin soporte de todas las bandas como indefinidas.

El informe añade ratios de p50/p95/p99/media del coste raw E2E ya medido. No
introduce un umbral de latencia posterior ni confunde coste de cabeza con coste
completo. Declara la diferencia de contexto: P2 frente a los expertos e historial
nativos de E_CTX. Es evidencia interna seed7; no promociona automáticamente el
sistema ni confirma SOTA. Réplicas e interpretación del coste siguen siendo
necesarias cuando la evidencia habilite esas fases.

Orden posterior a la matriz y su selección H:

```text
python -B -m operational.joint_measurement.evaluate predict --fit-id FIT_ADMITIDO --role V
python -B -m operational.joint_measurement.raw_cost
python -B -m operational.joint_measurement.system_compare
```

Reanudar `raw_cost` mientras devuelva PARTIAL. Si no hay candidato admitido por H,
`system_compare` deja constancia sin abrir outcomes V y la investigación continúa
con las ramas P/H autorizadas. El informe real todavía no se ha ejecutado.

Seis pruebas verifican que se rechacen cohortes omitidas, duplicadas, reordenadas
o con grupos/targets alterados, y que las predicciones inválidas permanezcan
visibles para el gate de cobertura. No equivalen a una comparación científica.

## Extensión RGB preparada el 14 de octubre

`operational.joint_measurement.rgb_system_compare` conecta los candidatos RGB
admitidos por `rgb/SELECTION_FINAL.json` con R_CTX (RGB) y F_TRUE (RGB+event).
Revalida la selección P/H y coste preparado, los endpoints, y la admisión
histórica de pesos, normalizadores y predicciones. Comprueba identidad, hashes,
cohorte completa y targets antes de calcular los gates originales y bootstrap
por grupos. Conserva predicciones inválidas y no reordena candidatos por V.
Si H no admite ningún candidato, registra ese resultado sin abrir outcomes V.

La comparación es de precisión interna: declara tanto el contexto P2 frente
al historial de los expertos como la modalidad adicional de F_TRUE. No atribuye
coste de cabeza al pipeline ni promociona el sistema. El coste E2E, réplica y
decisiones científicas habilitadas siguen siendo pasos separados.

```text
python -B -m operational.joint_measurement.rgb_system_compare --register-only
# Sólo después de la selección H final y las predicciones V del candidato admitido:
python -B -m operational.joint_measurement.rgb_system_compare
```

Validación: 11 tests nuevos y 6 tests de cohorte existentes aprobados. Incluyen
un informe completo con ambos comparadores sobre fixtures, ausencia de lecturas
V sin candidato, rechazo de procedencia/bytes/targets alterados y preservación
del candidato congelado pese a resultados inválidos. Ruff y formato aprobados.
Estas pruebas no son resultados científicos RGB V: su entrenamiento sigue vivo
y todavía no se ejecuta esta comparación sobre sus predicciones.
