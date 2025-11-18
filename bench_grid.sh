#!/usr/bin/env bash
set -euo pipefail

# ========= Config por defecto (sobre-escribibles por variables de entorno) =========
PUBKEY="${PUBKEY:-0290e6900a58d33393bc1097b5aed31f2e4e7cbd3e5466af958665bc0121248483}"
RANGE="${RANGE:-71}"
START="${START:-0}"
TAMES_FILE="${TAMES_FILE:-tames71_v15.dat}"   # Cambiá si querés otro archivo
REPEATS="${REPEATS:-5}"
DP_LIST="${DP_LIST:-14 15 16}"
TAME_BITS_LIST="${TAME_BITS_LIST:-4 5}"
TAME_RATIO_LIST="${TAME_RATIO_LIST:-25 33 40}"
MODE_TAG="${MODE_TAG:-j1}"   # usa j1/j0 para distinguir Jacobiano ON/OFF
LOGDIR="${LOGDIR:-logs}"

mkdir -p "$LOGDIR"

echo "== Bench grid =="
echo "PUBKEY=$PUBKEY"
echo "RANGE=$RANGE  START=$START"
echo "TAMES_FILE=$TAMES_FILE"
echo "REPEATS=$REPEATS"
echo "DP_LIST=$DP_LIST"
echo "TAME_BITS_LIST=$TAME_BITS_LIST"
echo "TAME_RATIO_LIST=$TAME_RATIO_LIST"
echo "MODE_TAG=$MODE_TAG"
echo "LOGDIR=$LOGDIR"
echo

# Chequeos
if [[ ! -x "./rckangaroo" ]]; then
  echo "ERROR: ./rckangaroo no existe o no es ejecutable. Compilá primero." >&2
  exit 1
fi
if [[ ! -f "$TAMES_FILE" ]]; then
  echo "ERROR: no se encontró $TAMES_FILE" >&2
  exit 1
fi

run_one() {
  local dp="$1"
  local tb="$2"
  local tr="$3"
  local r="$4"
  local of="$LOGDIR/${MODE_TAG}_dp${dp}_tb${tb}_tr${tr}_run${r}.log"

  echo ">> dp=$dp  tame-bits=$tb  tame-ratio=$tr  run=$r"
  /usr/bin/time -f "%E real  %Mk RSS  %I in KB  %O out KB" \
    ./rckangaroo -pubkey "$PUBKEY" -range "$RANGE" -dp "$dp" -start "$START" \
                 -tames "$TAMES_FILE" -tame-bits "$tb" -tame-ratio "$tr" \
      |& tee "$of"
}

for dp in $DP_LIST; do
  for tb in $TAME_BITS_LIST; do
    for tr in $TAME_RATIO_LIST; do
      for r in $(seq 1 "$REPEATS"); do
        run_one "$dp" "$tb" "$tr" "$r"
      done
    done
  done
done

echo
echo "== Listo. Logs guardados en $LOGDIR"
echo "Sugerencia: python3 summarize_bench.py $LOGDIR > summary_${MODE_TAG}.csv"
