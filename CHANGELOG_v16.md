# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.6.0] - 2025-08-22
### Added
- **Warp‑aggregated atomics** for distinguished point (DP) emission in GPU kernels.
- **Spanish README** updates mirroring English documentation.
- **Backward compatible reader** for legacy `.dat` v1.5 files.

### Changed
- **New compact `.dat` format (v1.6 / `TMBM16`)**: DP record size reduced from 32B → **28B** (X tail 5B, distance 22B, type 1B).
- **Improved memory coalescing** when writing DPs, enabling more efficient PCIe transfers.
- **Documentation**: Updated `README.md` and `README_es.md` with a “What’s New in v1.6” section, benchmarks, and build tips.

### Performance
- Throughput improvements of **+10–30%** (GPU/`-dp` dependent). Example: RTX 3060 `-dp 16` ~**+16%** over v1.5.

### Compatibility
- v1.6 binaries **read** both `.dat` **v1.5** and **v1.6** formats.
- v1.6 **writes** `.dat` in the new v1.6 format by default.

### Migration
- No action required for existing `.dat` v1.5 users; they continue to load.
- New runs will generate `.dat` v1.6 files (smaller on disk).

---

## [1.5.0] - 2024-XX-XX
- Initial public release by RetiredC.
- GPU implementation of Pollard Kangaroo with DP infrastructure.

---

# Registro de cambios (ES)

## [1.6.0] - 2025-08-22
### Añadido
- **Atómicas warp‑aggregadas** para la emisión de DPs en los kernels de GPU.
- **README en español** actualizado en paralelo al inglés.
- **Lectura retrocompatible** de archivos `.dat` v1.5.

### Cambiado
- **Nuevo formato `.dat` (v1.6 / `TMBM16`)**: registro DP de 32B → **28B** (cola de X 5B, distancia 22B, tipo 1B).
- **Mejor coalescencia de memoria** al escribir DPs, con transferencias PCIe más eficientes.
- **Documentación**: `README.md` y `README_es.md` incluyen “Novedades v1.6”, benchmarks y flags de compilación.

### Rendimiento
- Mejora de **+10–30%** (según GPU y `-dp`). Ejemplo: RTX 3060 `-dp 16` ~**+16%** vs v1.5.

### Compatibilidad
- Los binarios v1.6 **leen** archivos `.dat` **v1.5** y **v1.6**.
- v1.6 **escribe** por defecto el nuevo formato `.dat` v1.6.

### Migración
- No se requiere acción para usuarios con `.dat` v1.5; siguen cargando.
- Nuevas ejecuciones generarán `.dat` v1.6 (más pequeño en disco).
