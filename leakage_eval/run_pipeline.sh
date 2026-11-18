cd "$(dirname "$0")/.." || exit 1

if [ $# -lt 2 ]; then
  sed -n '2,/^cd /p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
  exit 1
fi

LOG="${1%/}"; PKL="$2"; shift 2
TAG=""; K=3; MATCH=(); MH=()
if [ $# -gt 0 ] && [[ "$1" != -* ]]; then TAG="$1"; shift; fi
while [ $# -gt 0 ]; do
  case "$1" in
    -k) K="$2"; shift 2 ;;
    --multihop) MH=(--multihop); MATCH+=("$1"); shift ;;
    *)  MATCH+=("$1"); shift ;;
  esac
done
TAG="${TAG:-$(basename "$LOG")}"

[ -e "$LOG" ] || { echo "log not found: $LOG" >&2; exit 1; }
[ -e "$PKL" ] || { echo "checkpoint not found: $PKL" >&2; exit 1; }

T=leakage_eval/tmp
mkdir -p "$T"
MEM="$T/memory_$TAG.out"; EVI="$T/evidence_$TAG.out"; SUR="$T/surfaced_$TAG.out"
MEM_R="$T/mem_ratio_$TAG.out"; SUR_R="$T/surfaced_ratio_$TAG.out"

run() { echo "+ $*" >&2; "$@"; }

run python leakage_eval/print_router_memory.py "$PKL"                       > "$MEM" || exit 1
run python leakage_eval/print_local_evidence.py "$LOG" "${MH[@]}"           > "$EVI" || exit 1
run python leakage_eval/surfaced_from_logs.py "$LOG" "$PKL" -k "$K" "${MH[@]}" > "$SUR" || exit 1
run python leakage_eval/match_memory.py "$MEM" "$EVI" "${MATCH[@]}"         > "$MEM_R" || exit 1
run python leakage_eval/match_surfaced_memory.py "$EVI" "$SUR" "${MATCH[@]}" > "$SUR_R" || exit 1

echo
echo "== memory ratio ($MEM_R) =="
tail -n 4 "$MEM_R"
echo
echo "== surfaced ratio ($SUR_R) =="
tail -n 4 "$SUR_R"
