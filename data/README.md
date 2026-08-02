# Política de datos

Los datasets crudos no se almacenan en Git.

## IDs permitidos

```text
EVTTC32_LABELLED
EAP_LOCAL_40_OF_46
GARLTTC_PUBLIC_LABELS
BENCHMARK10_SEALED
```

No se aceptan nuevas raíces sin una revisión explícita del protocolo.

## Contenido versionable

- manifests pequeños;
- splits legibles;
- hashes;
- fixtures sintéticos;
- documentación.

Cada entrada de manifest debe incluir fuente, ruta local, tamaño, metadata de
split y hash cuando esté disponible.

## Rutas operativas

```text
datasets/evttc
datasets/evttc_official_benchmark_sealed
E:\eAP_dataset
E:\GarlTTC_dataset
E:\Garl-TTC
```

La raíz sellada no se inspecciona durante desarrollo. Las tres raíces E: son de
solo lectura. CARLA fue retirado del inventario activo tras transferencia negativa;
su dataset local fue eliminado y solo se conservan resúmenes compactos.

## Prohibiciones

- datos crudos en Git;
- eAP test;
- extracción masiva de TAR RGB;
- voxel cache global;
- cache Garl high-resolution full (~455 GiB);
- logits SAM densos;
- hidden states DINO de todas las capas;
- pseudo-TTC presentado como ground truth.
