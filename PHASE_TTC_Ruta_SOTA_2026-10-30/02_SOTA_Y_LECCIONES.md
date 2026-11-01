# 02 — SOTA verificable y lecciones para PHASE-TTC

**Fecha de corte: 30 de octubre de 2026.** La recomendación es mantener la arquitectura actual durante el diagnóstico y probar primero una fusión pequeña que incorpore calidad de observación y estado temporal causal. La literatura reciente respalda investigar representaciones sensibles al movimiento si esos cambios no bastan; no demuestra que sustituir PHASE por un JEPA mayor vaya a mejorar TTC.

La búsqueda incluyó artículos completos, páginas de benchmarks, repositorios oficiales y búsquedas directas en arXiv. El buscador web general de esta sesión falló; las fuentes primarias se recuperaron directamente. La cobertura es documentada y actualizada, pero no exhaustiva en el sentido de garantizar que no exista ningún trabajo adicional.

## 1. Las comparaciones que deben permanecer separadas

| Problema | Referencia adecuada | Lo que no permite concluir |
|---|---|---|
| TTC geométrico firmado por objeto en eAP | Test12 GarlTTC, mismas 6.762 consultas, MiD y fallos oficiales | Que el modelo generaliza a otro sensor |
| Transferencia eAP → EvTTC sin ajuste | Checkpoint source-only, adaptador declarado, evaluación compatible | Que una variante entrenada en EvTTC sigue siendo zero-shot |
| Adaptación supervisada a EvTTC | Grupos de desarrollo independientes y evaluación reservada | Que el resultado sea comparable con métodos sin adaptación |
| TTC denso por píxel | EV-TTC/T²CEF, ETTCM u otro protocolo denso | Que sus métricas equivalgan a TTC por objeto |
| Anticipación de accidente o contacto humano | CCD/DoTA/DAD o Ego4D, respectivamente | Que se haya resuelto el TTC geométrico de carretera |

El **35,80 de PHASE-TTC** debe describirse aquí como resultado registrado en el recibo de GarlTTC del 28 de octubre aportado por el proyecto. No se verificó una clasificación pública en vivo durante esta revisión. La página y API del leaderboard se intentaron consultar, sin obtener una verificación suficiente del estado actual. Véase [01 — Estado](01_ESTADO_Y_DIAGNOSTICO.md).

## 2. Fuentes que cambian las decisiones

| Fuente primaria | Fecha verificada | Qué aporta | Aplicación y límite |
|---|---|---|---|
| [EvTTC: competición y resultados](https://nail-hnu.github.io/EvTTC/competition/) | Consulta 30-10-2026 | Diez secuencias y resultados por secuencia | Referencia directa; página histórica, no inventario de evaluaciones privadas |
| [eAP/Garl-TTC](https://arxiv.org/abs/2603.16303) y [release](https://github.com/NAIL-HNU/Garl-TTC) | Marzo 2026; release consultada 30-10 | Geometría, fusión tardía, supervisión de foreground, TTC firmado | Baseline aprendido obligatorio; su tabla EvTTC usa solo tres secuencias |
| [REACT](https://arxiv.org/abs/2609.19204) | 16-09-2026 | Estado continuo por evento, tiempo físico y incertidumbre | Relevante para memoria/latencia; protocolo no equiparado al test oficial de diez secuencias |
| [STRTTC, Event-Aided TTC](https://arxiv.org/abs/2407.07324), [código](https://github.com/NAIL-HNU/event_aided_ttc) | ECCV 2024 | Inicialización geométrica y registro espaciotemporal | Control de señal geométrica recuperable; hipótesis de movimiento restrictivas |
| [ETTCM](https://github.com/neuromorphic-paris/ETTCM) | ICCV 2023 | TTC denso a partir de profundidad inversa y movimiento | Baseline geométrico; requiere correspondencia con el target por objeto |
| [EV-TTC](https://github.com/anthonytec2/EV-TTC) | RA-L 2025 | Representación multiescala y red EV-Slim; T²CEF sobre M3ED | Posible comprobación externa; distinto del dataset EvTTC de HNU |
| [V-JEPA 2.1](https://arxiv.org/abs/2603.14482), [código](https://github.com/facebookresearch/vjepa2) | v1 15-03; v3 11-06-2026 | Features densas y temporales; ViT-B de 80M disponible | Probar encoder congelado; no hay superioridad TTC demostrada |
| [MotionJEPA](https://arxiv.org/abs/2609.23881), [código](https://github.com/mkarmann/motion-jepa) | 20-09-2026 | Evitar que el aprendizaje latente suprima movimiento rápido | Inspira un objetivo dinámico controlado, no una mejora automática |
| [LeWorldModel](https://arxiv.org/abs/2603.19312) | Marzo–junio 2026 | JEPA compacto y regularización de embeddings | Receta de investigación; no prueba suficiencia de 24 grabaciones |
| [Fast LeWorldModel](https://arxiv.org/abs/2606.26217) | Junio; v2 28-09-2026 | Planificación con predicción de prefijos de acciones | Prioridad baja para un estimador TTC sin acciones |
| [From Contrast to Consistency](https://arxiv.org/abs/2605.25570) | Mayo 2026 | Coherencia estructural/temporal en flujo continuo | Transferir un objetivo físico pequeño; revisar bidireccionalidad |
| [E-WAVE](https://arxiv.org/abs/2609.34346) / [SICAF](https://arxiv.org/abs/2610.29158) | Septiembre / octubre 2026 | Trayectorias continuas / saliencia temporal en flujo | Ideas auxiliares; sus resultados no son TTC |
| [CollideNet](https://arxiv.org/abs/2604.16240) | 17-04-2026 | Forecast de accidente multiescala | MSE en CCD/DoTA/DAD; tarea diferente |

## 3. Los baselines geométricos siguen siendo exigentes

La [tabla oficial EvTTC consultada](https://nail-hnu.github.io/EvTTC/competition/) contiene estos errores. Sus métodos usan modalidades y entornos de ejecución diferentes; hay que mantenerlos visibles al comparar.

| Secuencia | Image’s FoE (%) | CMax (%) | STRTTC (%) |
|---|---:|---:|---:|
| CCRs1-low, rotulada slow en la tabla | 5,37 | 2,56 | 7,54 |
| CCRs1-medium | 5,12 | 3,44 | 9,64 |
| CCRs1-high | 5,12 | 7,39 | 7,78 |
| CCRs2-low | 3,76 | 5,04 | 8,15 |
| CCRs2-medium | 3,85 | 11,20 | 11,83 |
| CCRs2-high | 2,92 | 11,10 | 13,68 |
| CCRm-low | 5,60 | 9,27 | 9,87 |
| CCRm-medium | 3,86 | 14,73 | 10,36 |
| Slider-750 | 11,35 | 4,16 | 8,95 |
| Slider-1000 | 7,55 | 2,74 | 12,43 |
| **Media aritmética propia de estas filas** | **5,450** | **7,163** | **10,023** |

La última fila es un cálculo descriptivo, **no una clasificación global oficial** ni una media ponderada por consultas. No promediar estos números con MiD de GarlTTC. CMax tarda aproximadamente 0,85–3,83 s por consulta en los entornos publicados; FoE, aproximadamente 0,012–0,036 s. La comparación final debe medir precisión y latencia con el mismo alcance del pipeline, no convertir esos tiempos históricos en una competición de hardware.

La utilidad inmediata es diagnóstica: si una señal geométrica recupera expansión en Slider y PHASE falla, se debe localizar la discrepancia en preparación, ROI, productor o fusión. No hay evidencia para atribuir todo el error de Slider a la complejidad insuficiente del encoder.

El [artículo Garl-TTC](https://arxiv.org/html/2603.16303v1) reporta transferencia sin fine-tuning en **tres** secuencias EvTTC: 8,31 % en CCRs2-medium, 10,56 % en CCRs2-high y 12,93 % en CCRm-medium; media 10,60 %. Su evaluación FCWD informa 5,2/6,1/5,4 % en tres secuencias. Estos resultados son referencias de transferencia; no cubren los dos Slider ni las diez secuencias oficiales completas.

## 4. REACT: avance pertinente, comparación todavía incompleta

[REACT](https://arxiv.org/html/2609.19204v1) mantiene un estado actualizado por evento, incorpora el intervalo temporal real y usa dos bloques de 128 canales. En EvTTC reduce coordenadas a 160×90, aplica un refractario de 10 ms por píxel grueso/polaridad y entrena ventanas de 32.768 tokens tras decimación, aproximadamente 0,8 s. La red suma 3,7M parámetros; gran parte corresponde al embedding espacial.

Su aportación transferible es **memoria causal, reloj físico y confianza**. Una GRU pequeña o un filtro de estado puede probar esas propiedades antes de adoptar una arquitectura spiking. La cifra destacada por los autores, 9,59 ± 0,74 %, exige estas reservas:

| Aspecto | Evidencia del artículo | Interpretación correcta |
|---|---|---|
| Evaluación principal | Tabla II: validación en dominio; cinco semillas | No atribuirla al test oficial completo |
| Split | Figura CCR1 medium-overlap-50; manifest completo no identificado en el texto | Exigir lista de grabaciones y grupos antes de reproducir |
| Slider | Inclusión no documentada explícitamente | No afirmar que el método resuelve Slider |
| Generalización | FCWD 25,7 %; ablación reservada 34,2 % full-field y 28,3 % con crops | La cifra de validación no resume transferencia |
| Baseline Garl | Tabla II: 9,44 % | No coincide con la tabla de tres secuencias eAP; requiere explicación de protocolo |
| Negativos | Salida positiva; evaluación sin casos negativos | No copiar esta cabeza para TTC firmado |
| Recursos | 4,6 ms reportados; energía basada en proxy de 45 nm | No tratar energía como medida real ni asumir mismo hardware |
| Código | El manuscrito anuncia publicación futura del código | Reproducción pendiente de artefactos suficientes |

La cabeza estima media y log-varianza de \(\log(1/\tau)\), adecuada para su evaluación positiva. PHASE necesita conservar el signo de alejamiento. Además, el texto consultado no especifica suficientemente el dispositivo del 4,6 ms. Esta revisión no puede convertir ese número en una promesa de latencia en la RTX local.

REACT también muestra que el arranque importa: su prueba informa errores medianos de 27 % con 100 ms, 20 % con 200 ms y 12,9 % alrededor de 400 ms de observación, con degradación posterior. Esto respalda medir arranque y duración efectiva, no eliminar las primeras consultas.

## 5. MiD no significa lo mismo en todos los artículos

El [artículo eAP](https://arxiv.org/html/2603.16303v1) define:

\[
\mathrm{MiD}=10^4|\log\hat\eta-\log\eta|,
\qquad \eta=1-\Delta t/\tau.
\]

El score agrega bandas crucial, pequeña, grande y negativa con pesos 0,5/0,3/0,1/0,1. El código oficial debe resolver el instante de referencia, intervalos, dominio del logaritmo y política de fallos. Los valores publicados por bandas son 53,1/37,6/40,6/31,3; su agregado calculado sobre esos números redondeados es **45,02**, frente al **45,0** redondeado del artículo. El **45,03** del recibo de leaderboard es otra cifra, calculada con mayor precisión: no se deduce exactamente de las cuatro bandas redondeadas a una décima. La [release](https://github.com/NAIL-HNU/Garl-TTC) confirma 6.762 consultas y que el test público carece de TTC de referencia.

REACT llama \(\eta\) a **\(1/\tau\)** y define un \(\mathrm{MiD}_w\) de log-ratios de TTC inverso, sin el factor \(10^4\) y sin negativos. No comparar su 0,0986 con 35,80 o 45,03. La coincidencia de símbolos no demuestra equivalencia matemática.

El error porcentual de EvTTC tampoco es MiD. Puede crecer con un pequeño desplazamiento absoluto cuando el TTC real es pequeño. Una predicción positiva y acotada, como en Slider-1000, puede seguir siendo muy incorrecta por una referencia temporal, geométrica o dinámica inadecuada. Véase [04 — Evaluador](04_EVALUADOR_Y_BASELINES.md).

## 6. Ventana de historia, arranque y latencia

Usar 650 ms de historia pasada no obliga a esperar 650 ms adicionales en cada consulta. En régimen continuo, la historia ya está disponible; en arranque puede faltar. El tiempo de exposición, la espera hasta la última observación necesaria y el cálculo son magnitudes diferentes.

La comparación de latencia de REACT suma integración a runtime en otros métodos. Antes de adoptar esa convención, hay que fijar qué dato corresponde al instante de consulta y cuándo está disponible. Para PHASE se medirán por separado: historia requerida, frecuencia de actualización, latencia desde la última muestra necesaria, latencia desde la consulta y tiempo hasta primera predicción. También se distinguirá la actualización en streaming de la inferencia por lotes. Véase [11 — Runtime](11_RUNTIME_RECURSOS_Y_REPRODUCIBILIDAD.md).

## 7. Qué técnicas merecen pasar a experimentación

**Primera línea:** contratos correctos, confianza por productor, estado causal y fusión pequeña. Permite decidir si el fallo era de entrada, soporte, memoria o combinación, conservando las piezas que ya funcionan.

**Segunda línea:** objetivo de movimiento y equivarianza temporal. [MotionJEPA](https://arxiv.org/abs/2609.23881) muestra que una representación puede evitar el colapso global y aun así suprimir elementos rápidos. El control pertinente es un mismo encoder con y sin el objetivo dinámico, acompañado de probes de expansión y signo; no basta reducir la pérdida JEPA.

**Tercera línea:** [V-JEPA 2.1-B congelado](https://github.com/facebookresearch/vjepa2) como referencia de representación visual, con la misma cabeza e historia causal. No se entrenará un foundation model desde cero en 12 GB. La ventaja debe justificar memoria, runtime y complejidad.

Los trabajos [STSC](https://arxiv.org/abs/2605.25570), [E-WAVE](https://arxiv.org/abs/2609.34346) y [SICAF](https://arxiv.org/abs/2610.29158) motivan conservar trayectorias y saliencia temporal. Sus mejoras de flujo no prueban una mejora TTC; las redes bidireccionales necesitan revisar acceso a futuro.

## 8. Datos externos y alcance de una afirmación SOTA

[EV-TTC](https://github.com/anthonytec2/EV-TTC), con guion, es otro proyecto: usa T²CEF construido sobre M3ED y TTC denso. Su README reporta 3,3 ms para el **filtro** a 75 millones de eventos/s en Jetson Orin NX, no necesariamente la latencia completa de TTC. Puede aportar una evaluación externa tras alinear geometría y unidades; no es una sustitución directa de EvTTC.

FCWD solo será un test independiente si no se usó anteriormente para seleccionar PHASE o sus decisiones. Si ya fue explorado, deberá declararse y complementarse con capturas nuevas. Para la debilidad actual, un conjunto Slider nuevo y reservado puede aportar más información que muchas arquitecturas sobre dos clips conocidos.

La afirmación final deberá especificar tarea, modalidades, cajas, historia, datos de entrenamiento, comparadores, cobertura y fecha. Una mejora interna es una mejora interna; una posición oficial es una posición en ese leaderboard. La campaña de [08](08_JEPA_Y_ARQUITECTURAS_FUTURAS.md) y [10](10_CONFIRMACION_ESTADISTICA_Y_SOTA.md) convierte esas exigencias en experimentos y condiciones de cierre.
