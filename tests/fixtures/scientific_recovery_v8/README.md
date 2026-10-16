# V8 freezer fixtures

`dino_teacher_manifest.json` preserves the historical metadata bytes referenced by the grouped A5 configuration (SHA-256 `6ee4b205d29a07d3aabf7137f2a312376711b5e5d9d7930ea36328390023eab3`). It contains metadata and checksums, not cached tensors, images, or model weights.

The V8 freezer tests generate synthetic OOF rows and checksum-only checkpoint files in a temporary repository. They exercise the real signature, hash, population, fold and tamper checks. Those files are test inputs, never research results. Production configuration files are not rewritten by the tests.
