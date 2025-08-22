# RCKangaroo v1.5 (build local)

Implementación de *Kangaroo ECDLP* acelerada por GPU con varias mejoras
orientadas a **tiempo de cómputo** y **uso de memoria / I/O**.

> Probado con CUDA 12.x y NVIDIA RTX 3060 (SM 8.6).  
> Esta rama mantiene la **CLI original** y añade banderas/guías para *tame tuning* y *benchmarks*.


## Novedades técnicas (V1.5)

1) **Coordenadas Jacobianas en GPU** (opt-in)
   - Suma/doblado en Jacobiano para evitar inversiones modulares por paso.
   - Conversión a afin **solo cuando hace falta** (p.ej. para DPs o salida).
   - Add mixta (*Jacobian + Affine precomp*) para puntos de salto.
   - Conmutador de compilación: `USE_JACOBIAN=1` (habilitado en build por defecto de esta rama).

2) **Inversión por lotes (Truco de Montgomery)**
   - Se invierten muchos `Z` con **una sola inversión** y productos hacia delante/atrás.
   - Útil en compactación/normalización de estados y/o verificaciones masivas.

3) **TAMES v1.5 – formato compacto**
   - **~30–35% menos tamaño** vs. formato clásico en nuestros tests (p.ej. 84 MB → 57 MB).
   - Layout contiguo + compresión ligera (delta + varint/RLE) y lectura *streaming*.
   - Carga más rápida y menos *pressure* de caché/L2/PCIe.
   - **Compatibilidad**: el binario sigue aceptando el formato anterior; si el archivo no es v1.5, se lee por la ruta clásica.

4) **Menos I/O y binario optimizado**
   - Flags `-ffunction-sections -fdata-sections` (host) y `-Xfatbin=-compress-all` (device).
   - Caché L1/tex en *ptxas* via `-Xptxas -dlcm=ca` en `build.sh`.

> Nota: la *Montgomery Ladder* también está disponible en el código, pero no se fuerza por CLI; se usó Jacobiano + ventanas clásicas/mixtas, que mostraron mejor relación velocidad/uso de recursos en Ampere.

---

---

## 🚀 Novedades en v1.6

### Mejoras en GPU
- **Atómicas warp-aggregadas en emisión de DPs**: reduce de 32 atómicas por warp a 1, con escrituras coalescentes. **+10–30% rendimiento** según GPU y -dp.
- **Mejor coalescencia de memoria** en DPs y transferencias PCIe.

### Nuevo formato `.dat` (v1.6)
- **28B por registro DP** (vs 32B en v1.5).
  - Cola de X: 5 bytes (antes 9).
  - Distancia: 22 bytes.
  - Tipo: 1 byte.
- **Etiqueta de archivo `TMBM16`** identifica el nuevo formato.
- **Compatibilidad hacia atrás**: lectura de v1.5 y v1.6.

### Benchmarks (RTX 3060)
- v1.5: ~750 MKey/s @ -dp 16.
- v1.6: ~870 MKey/s @ -dp 16.
- ~16% más rápido y ~12.5% menos tamaño en `.dat`.



## Archivos modificados / añadidos

- **`RCGpuCore.cu`**  
  Implementaciones Jacobianas (doble/suma mixta), camino de *batch inversion* y selección de kernels según `USE_JACOBIAN`.

- **`RCGpuUtils.h`**  
  Primitivas de campo y helpers para Jacobiano (doble / add mixed).

- **`utils.h`, `utils.cpp`**  
  - Nueva ruta de **lectura/escritura TAMES v1.5** (streaming, compacta).  
  - Limpieza de utilidades y helpers varios.

- **`GpuKang.cpp`, `GpuKang.h`**  
  - Parámetros de *tame tuning* (ratio y bits) expuestos para benchs controlados.
  - Generación de distancias y partición *tame/wild* estable.

- **`RCKangaroo.cpp`**  
  - Parsing de CLI y *guard-rails* (mensajes de error consistentes).  
  - Modo *bench* más verboso.

- **`Makefile`**  
  - Objetivo directo para `rckangaroo` (sin librerías intermedias).  
  - Soporte `SM`, `USE_JACOBIAN`, `PROFILE` y *linking* determinista.

- **Scripts de apoyo**  
  - `build.sh` – *wrapper* de compilación multi-SM.
  - `bench_grid.sh` – *sweep* de parámetros (dp / tame-bits / tame-ratio) con repetición y logs.
  - `bench_rck.sh` – *benchmark* de A/B rápido.
  - `summarize_bench.py` – parser de logs → CSV (speed, tiempo real, RSS, parámetros).

---

## Árbol del proyecto (esta rama)

```
.
├── logs/                          # salida de bench_grid.sh
├── bench_grid.sh
├── bench_rck.sh
├── build.sh
├── Makefile
├── defs.h
├── Ec.cpp
├── Ec.h
├── GpuKang.cpp
├── GpuKang.h
├── RCGpuCore.cu
├── RCGpuUtils.h
├── RCKangaroo.cpp
├── rckangaroo                   # binario (tras build)
├── summarize_bench.py
├── tames71.dat                  # ejemplo formato clásico
├── tames71_v15.dat              # ejemplo formato v1.5 (compacto)
├── utils.cpp
└── utils.h
```

---

## Compilación

### Opción A – `build.sh` (recomendada)
```bash
# Sintaxis: ./build.sh <SM> <USE_JACOBIAN 0|1> <profile: release|debug>
./build.sh 86 1 release     # RTX 3060 (SM 8.6), Jacobiano ON
./build.sh 86 0 release     # Jacobiano OFF (afin) para A/B
```
Genera `./rckangaroo` en el directorio actual.

### Opción B – `make`
```bash
# Variables: SM, USE_JACOBIAN, PROFILE=(release|debug)
make SM=86 USE_JACOBIAN=1 PROFILE=release -j
```

> Requisitos: CUDA 12+, `g++` C++17, driver suficiente para la SM objetivo.


---

## Modo de uso (CLI)

Ejemplo mínimo (con TAMES v1.5):
```bash
./rckangaroo \
  -pubkey 0290e6900a58d33393bc1097b5aed31f2e4e7cbd3e5466af958665bc0121248483 \
  -range 71 \
  -dp 16 \
  -start 0 \
  -tames tames71_v15.dat
```

Parámetros útiles de *tame tuning* (se pasan por CLI y se reflejan en logs de bench):
```
  -tame-bits <N>      # bits usados para los saltos "tame" (p.ej. 4–7)
  -tame-ratio <PCT>   # porcentaje de canguros tame (p.ej. 25–50)
```
Ejemplo:
```bash
./rckangaroo ... -tame-bits 4 -tame-ratio 33
```

> Sugerencia: buscar combinaciones que **maximicen MKeys/s** pero con **menor tiempo real** y **memoria** aceptable.


---

## Benchmarks automatizados

### Barrido de parámetros (grilla)
```bash
# Editar cabezal del archivo para ajustar PUBKEY/RANGE/DP/TAMES/etc.
chmod +x bench_grid.sh summarize_bench.py

# Ejecutar barrido (graba todo en logs/)
./bench_grid.sh

# Resumir a CSV y visualizar
python3 summarize_bench.py logs > summary.csv
column -s, -t < summary.csv | less -S
```
Comparativa Jacobiano OFF/ON:
```bash
# Jacobiano ON
./build.sh 86 1 release && MODE_TAG="j1" ./bench_grid.sh
python3 summarize_bench.py logs > summary_j1.csv

# Jacobiano OFF
./build.sh 86 0 release && MODE_TAG="j0" ./bench_grid.sh
python3 summarize_bench.py logs > summary_j0.csv
```

> **TIP**: Dejá `REPEATS>=5` para mitigar jitter; el parser reporta **medianas** por combinación.


---

## Resultados de referencia (orientativos)

En pruebas rápidas de 71 bits en RTX 3060:
- **TAMES v1.5**: 84 MB → **57 MB** (~32% menor).  
- **Tiempo real**: ~100 s → **~65 s** (Jacobiano + v1.5 + mismos parámetros).  
- **RSS**: ligera reducción (≈ -20–30 MB según corrida).

> Los números varían por DP, *tame-bits*, *tame-ratio*, reloj de la GPU y versión de driver.


---

## Compatibilidad y notas

- El binario mantiene la lectura del formato de TAMES **clásico** y del **v1.5** (detectados por cabecera / heurística).  
- Si necesitás convertir masivamente a v1.5, se recomienda regenerar con el *pipeline* que usás para crear los tames, apuntando al escritor v1.5 (ver `utils.cpp`).


---

## Solución de problemas

- **`Unknown option -ffunction-sections` en NVCC**: usá `build.sh` (pasa por `-Xcompiler`).  
- **`No rule to make target 'RCGpuCore.o'`**: asegurate de usar este repositorio / Makefile o `./build.sh`.
- **`CUDA error / cap mismatch`**: compila con `./build.sh <tu SM> ...` (p.ej. 75 para Turing, 86 para Ampere).


---

## Licencia

Mantiene la licencia del proyecto original (ver `LICENSE.TXT` si aplica).  
Autorizado a ser usado con fines de investigación y educativos.
