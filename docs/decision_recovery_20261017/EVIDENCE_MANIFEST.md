# Manifiesto de evidencia local V19

Captura de identidad: 2026-10-17T10:40:39+00:00.

Este manifiesto acompaña al [informe de ejecución y resultados](INFORME_EJECUCION_Y_RESULTADOS.md). Los 38 archivos se leyeron y sus SHA-256 se calcularon al redactarlo. No se regeneraron resultados ni se modificaron registros científicos.

Todas las rutas de la tabla son relativas a `artifacts/decision_recovery_20261017/` en el worktree V19. **Los artefactos, datos y checkpoints permanecen locales; este commit publica documentación, no una copia de esos archivos.** Los recibos enlazan internamente las predicciones, padres, fuentes y configuraciones. El manifiesto identifica la evidencia: no sustituye disponer de ella ni reproduce por sí solo el experimento.

El cierre de fuentes ejecutado es `44f64831e93f9a09d351cc80149fd223b13aa805`. Los recibos individuales conservan sus revisiones originales. Las correcciones CI posteriores (`f96a4be`, formato; `593ed6c`, aislamiento de test) no reescriben esos hashes. Usar la revisión declarada por cada registro al hacer verificaciones estrictas.

Los archivos de presupuesto y estado son snapshots al corte, no promesas de inmutabilidad futura. La reserva histórica de 900 s en GPU_BUDGET no es un proceso activo ni tiempo de kernels medido. CONTINUATION_STATE documenta el bloque completado; los estados intermedios antiguos no anulan sus recibos finales.

| Ruta local relativa | Bytes | SHA-256 |
|---|---:|---|
| `REGISTRATION.json` | 19907 | `4470a3d4d8b334e8c7e0122b306ab0ea3662155e56fd4efa6ebd18f392a75489` |
| `training/REGISTRATION.json` | 88665 | `8ce5bb95085a4e7bfaf5dc9e84b5153e0b6156cc9d27cb36e08e815235d350da` |
| `training/ACCOUNTING.json` | 125 | `244c1ee2f571ddad3b44a62ca416631b329d125a70c8362b710874d432380e27` |
| `training/GPU_BUDGET.json` | 4728 | `a9895edee5c9bf83308773f1727c10c92262786809738e592a47e7bf05f645c1` |
| `training/qa/RESULT.json` | 6069 | `91eeff31d089fb4fdaa7daef41e53b0ef6e756e0053dcf73f634aa7e46cc9374` |
| `training/ADMISSION.json` | 305 | `6f10fc4e7d5e76a3547f4370d8a52cadbd731918428ace4c0143aab3f41bc84f` |
| `evaluation/REPORT.json` | 46512 | `1073ab21e307ae266221816fd52422f5117f23b0fb7927429e89970a6d996927` |
| `evaluation/L0_COMPLETE.json` | 1545 | `7863f5f6d61bf4b3a24da563aa791d6de54283fab7122b031ddfb385e2848e3a` |
| `evaluation/LW_COMPLETE.json` | 1545 | `db7da259f02e7c1b287403ef3f65ab25fa6284669355b32aa892fa583988ea35` |
| `evaluation/exports/L0/COMPLETE.json` | 485 | `dbed2837fd81742a76b9a4231b9b0131490c446bbca46c7e07f2ff083814f550` |
| `evaluation/exports/LW/COMPLETE.json` | 485 | `4171bff20be43799b100f4cadee6bd79a6b1ad69dece69c302fe1c4cf1b026fb` |
| `diagnostics/COMPLETE.json` | 8861 | `abb817e4ffd6d1ab435a1e449a310f46042e6d07c52e0d8d70944aafab2fda13` |
| `diagnostics/P.json` | 997186 | `fb9a335e688a40b8421416817eba5c0bb642919460cce5729def4c3695285a3f` |
| `diagnostics/H.json` | 371304 | `ddb1380ba52a48b29aa8ec56c034a444ac8f1db5a633733875f77ff5a3668d89` |
| `router/DECISION.json` | 2130 | `55d0bc15741d123d44519082d21ac185d2ede9bde2ecf857dac94a3fc9085fc5` |
| `router/NATIVE__FULL/RESULT.json` | 41770 | `1cefa0f5af96558c5d02f95656e4031e2f91e3d7df2a01011eca90cf8d133e52` |
| `router/A5__FULL/RESULT.json` | 34065 | `896ae3be1867d35cdbae2a21b4c69108dfa569b86e400b604e22c316413a5b8e` |
| `router/NATIVE__CORR/RESULT.json` | 41706 | `4d8df93a07fc32928122a5ada97371836971609724e55231ffcbad4020fd944a` |
| `loss_negative_analysis/REPORT.json` | 214265 | `5c78a490afd9323824666421368b6c2f242fc70954138cdddd4c59aa9cfa578d` |
| `loss_negative_analysis/P_FAILURE_PROFILE.json` | 196223 | `781195d78fd199e473011e0e08b8a5683f3fd8dc24eda05c256904e79b326c93` |
| `loss_pair_oracle/REPORT.json` | 65024 | `211c00aa995e71e5e9702ec89540a61df6e52cb049a16eed1d3a31d1ff4f0f82` |
| `diagnostics_gradients/RESULT.json` | 44203 | `3129b769eb073a510a016e14f8aeb17ce4663f5a24a82e2a9cb4206d9d500cec` |
| `sign_probe_v2/REGISTRATION.json` | 96267 | `cf7f6b9baf939cc76b2a3a66980836bc1f8a5af9a2016625ca51b432495135f5` |
| `sign_probe_v2/TECHNICAL_RECOVERY.json` | 42946 | `b3f3311e973df5248cc2efb706bd534d9140c620ade1f70b3cbbda201f29a665` |
| `sign_probe_v2/features/P/COMPLETE.json` | 20457 | `241cba8bc2c3c4caab0afcfd0b626b0144cd8d2de6f5bb7c7adffcf331db2d70` |
| `sign_probe_v2/features/H/COMPLETE.json` | 7528 | `6ee039103dbea95bf9e9935abe5b82b76aba84a5c61e4a051140deb9c5c63db4` |
| `sign_probe_v2_continuation/REGISTRATION.json` | 4147 | `142a454e82c4233fe6542626f6ecb999c7eaf194469ab884fbd445f9beef8a2d` |
| `sign_probe_v2_continuation/ORIGINAL_FAILURE.json` | 505 | `af9108978a077764b4d7646ef41f1f330b748bc5c4603493e4e07d79216bdf7e` |
| `sign_probe_v2_continuation/RESULT.json` | 55482 | `a3657eea27900759b54af6eeaa2e8d781477c34d32c9f940336e9f5a234bc13f` |
| `sign_probe_v2_continuation/COMPLETE.json` | 272 | `387ca6b93ee506cd7b5de1e332c5964bfca557c39ea0c656915337c7a7c76d92` |
| `feature_router_r1/REGISTRATION.json` | 15271 | `41b51c0f489fcd802ce1fbab97353af030cb7890f3be24c23a9cd5c204947902` |
| `feature_router_r1/MATERIALIZATION.json` | 442 | `55bdf81df70cc8d5f223f5cb8f9cbb90dd36662c2c61caab6cf2fbf1175d9ac4` |
| `feature_router_r1/RESULT.json` | 80069 | `d9309f7aa9763b66b192c26fec2d88c1a9e96870a870703ce61ecb639876897c` |
| `feature_router_r1/COMPLETE.json` | 2281 | `e887ec0d00f19473432f33f337e91ec80dea2fec0808a2328f4048d3d70c2d50` |
| `cost/batch4_native_and_batch1_descriptive_P2_to_TTC_v5/REGISTRATION.json` | 16807 | `8e52692797e837903f1006c2b459ef6adc942976292546866b111edf8ddbb5ad` |
| `cost/batch4_native_and_batch1_descriptive_P2_to_TTC_v5/attempts/8e52692797e83790/RESULT.json` | 19252 | `56bf767122a81a9b198319bf2655fea38ba94429dc9173bd640ece93628d0fd5` |
| `cost/batch4_native_and_batch1_descriptive_P2_to_TTC_v5/COMPLETE.json` | 443 | `dcd93cb47c868d4af77dd2c8f420b68f72840e4617ed22656e158c4f09491275` |
| `CONTINUATION_STATE.json` | 732 | `731a0237660e3fad3e71d8a863306746c100b23722c1a9585ddafec6d52b24c9` |

## Evidencia de CI remota

[GitHub Actions 39024446650](https://github.com/Kripta-Studios/e-jepa-ttc/actions/runs/39024446650), commit `593ed6c25ca528c063754e6c6b13024538729dbe`: success, 17 de octubre de 2026, 10:25:43 UTC. Total pytest: 4.415 aprobadas y 23 omitidas. Esta ejecución verifica el código y fixtures; no reproduce los entrenamientos ni carga los artefactos locales anteriores. Un push documental posterior tendrá su propia ejecución de CI.

## Comprobación de identidad sin volver a ejecutar modelos

Desde la raíz del worktree, PowerShell permite comparar cualquier entrada de la tabla:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath artifacts/decision_recovery_20261017/evaluation/REPORT.json
```

Para métricas y decisiones, consultar los JSON identificados y sus informes especializados; no inferir que un SHA correcto demuestra por sí solo validez científica o generalización.
