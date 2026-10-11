# 03 — Protocolos, cohortes y significado de las métricas

## Poblaciones distintas

- **eAP TRAIN40 V12:** 88.744 consultas de 40 secuencias. Es entrenamiento del
  sistema propio, con las cuatro bandas TTC. Permite calcular overall_MiD,
  pero no demostrar generalización.
- **V de V13:** 16.432 consultas, ocho grupos reservados por la receta V13.
  Usa agregación macro por grupo y banda. No es test privado Garl ni se debe
  comparar numéricamente con TRAIN40 como si fueran la misma población.
- **Dev32:** transferencia exploratoria EvTTC, 32 secuencias con 32 consultas
  cada una. La campaña final tiene 1.024 predicciones y 946 etiquetas finitas;
  78 quedan sin GT. RGB disponible en todas las consultas.
- **FCWD:** tres secuencias, 630 consultas y 597 etiquetas finitas; 33 sin GT.
  El registro RGB se acepta en 495 consultas, de las que 472 tienen GT.
  Las 125 consultas etiquetadas sin RGB usan fallback E_CTX para F_TRUE/F_ZERO.
- **test12 GarlTTC:** 6.762 entradas públicas y TTC privado. El recibo V12 es
  `PACKAGED_NOT_SUBMITTED`: dos juegos completos H8/Garl event-only, sin score
  CodaBench acreditado. El archivo sample_submission no contiene GT.

Fuentes V12 y sus predicciones: `V12/docs/agent_handoff_20261010/` dentro del
ZIP, o el enlace inmutable del índice. Fuentes actuales: los dos CSV en
`docs/{fcwd_rgb_registration_20261012,dev32_transfer_20261012}/`.

## Fórmula y dominio

Para delta=0,1 s:

```text
phi(T) = -log(1 - 0.1/T)
MiD(T, pred) = 10^4 * abs(phi(T) - phi(pred))
abs(d phi / dT) = 0.1 / abs(T * (T - 0.1))

c: (0,3]      s: (3,6]      l: (6,10]      n: (-10,0]
overall_MiD = 0.5*MiDc + 0.3*MiDs + 0.1*MiDl + 0.1*MiDn
```

La fase real es finita para T<0 o T>0,1, no para cualquier valor de la banda c.
T=0, T=0,1, TTC no finito y el intervalo positivo hasta 0,1 requieren el manejo
de fallos del protocolo. Las tablas externas muestran el promedio de fases
finitas **junto al número de casos inválidos**; esa media sola no puede ocultar
fallos. Un TTC de signo incorrecto puede tener fase finita: por eso también se
cuentan errores de signo.

MAE promedia error en segundos. RTE divide error absoluto por TTC real absoluto.
MiD mide error en fase, con sensibilidad mucho mayor cerca del contacto. Por
eso un modelo puede ganar MAE y perder MiD crucial. Añadir otra pérdida llamada
«MiD» no aporta automáticamente una señal ausente: H8 ya aprende en fase.

En FCWD y Dev32 no hay banda negativa etiquetada. Su overall oficial completo
queda **no disponible**. No se reemplaza MiDn por cero ni se renormalizan los
pesos bajo el nombre oficial. `mean_MiD` histórico promedia filas válidas y
depende de su distribución; no es el overall ponderado entre bandas.

## V13 V: macro por grupo

`operational/rgb_port/evaluate.py` calcula medias por grupo dentro de cada
banda y luego aplica 0,5/0,3/0,1/0,1. El campo se llama
`group_macro_bucket_MiD`. Una secuencia con muchas ventanas no recibe
automáticamente más peso por tener más filas. La tabla 04 mantiene ese nombre.
Los gates V usan bootstrap por grupo, ocho grupos y 2.000 réplicas; el gate
R_CTX frente a R_H1 falla el mínimo de mejora del 5%, aunque su diferencia
media sea favorable. No se presenta como un gate aprobado.

## Pesos efectivos y exposición

En V12 `simplex_t/cache.py::training_mass` reparte masa por secuencia/banda con
pesos relativos 0,5/0,3/0,1/0,1. En V13
`operational/rgb_port/prepare.py::_role_payload` construye masa uniforme por
grupo: `1/(numero_grupos * filas_grupo)`. Esto cambia el objetivo efectivo
aunque ambos sistemas usen fase. No demuestra un bug ni explica retrospectivamente
FCWD V12. Antes de un experimento se debe verificar la masa materializada de H,
su manifest, el muestreo realizado y la contribución al gradiente por episodio.
Esta publicación no ejecuta una ablation de pesos ni altera los fits terminados.

## Reglas para comparar las tablas

1. Comparar dentro de la misma cohorte y política de salida. Los CSV finales
   incluyen cada comparador alineado por query_id. No sustituir una fila por
   un número más favorable de un informe anterior.
2. Las tablas V12 históricas y la transferencia V13 tienen runners, historiales,
   soporte GT y políticas de salida que deben contrastarse. Las cifras Garl
   Dev32 difieren entre campañas; este handoff no atribuye esa diferencia a una
   mejora del checkpoint Garl ni presume igualdad de población efectiva.
3. Mantener resultados nativos separados de análisis con clipping común ±60 s.
   Las tablas actuales recalculan los valores almacenados, sin nuevo clipping.
4. F_TRUE FCWD recibe RGB cuando está disponible; Garl FCWD disponible es
   event-only. Es una comparación entre sistemas, con distinta información.
5. La vista de 472 consultas RGB válidas permite comparar sobre iguales filas,
   pero no sustituye la población completa: selecciona casos más fáciles para
   la ruta de registro. Deben conservarse fallback, cobertura y errores.
6. Las tres semillas de cabeza V12 comparten productores. No equivalen a tres
   entrenamientos independientes de todo el sistema. V13 no acredita aquí tres
   réplicas completas por endpoint.

Los protocolos externos son exploratorios y ya expuestos. Haber separado labels
durante inferencia evita una fuga directa; no convierte análisis repetidos sobre
estos conjuntos en evaluación ciega para seleccionar el siguiente modelo.
