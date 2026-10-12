#!/usr/bin/env bash
# =====================================================================
#  compare_algorithms.sh
#
#  Benchmarks the four outer-loop algorithms (SIMPLEC, SIMPLER,
#  SIMPLERR, IDEAL) on the SAME 64^3 sphere setup and reports, for each:
#
#      K (m^2)     final Darcy permeability
#      iters       number of outer iterations actually run
#      time (s)    wall-clock time of the run (this machine)
#      converged   whether RMR dropped below simpletol
#
#  The setup is shared by every run (tests/compare_algorithms.ini); only
#  the 'method' line is overridden per algorithm, so the comparison is
#  apples-to-apples.
#
#  Usage:
#      compare_algorithms.sh <path-to-NS3D> <working-dir> <base-ini>
#
#  Environment:
#      OMP_NUM_THREADS  number of OpenMP threads (default 4). The same
#                       value is used for every algorithm, so the timing
#                       comparison is fair.
#
#  Exit status: 0 if every algorithm produced a finite, positive
#  permeability; non-zero otherwise (or if any run failed).
# =====================================================================
set -euo pipefail

NS3D="${1:?usage: compare_algorithms.sh <NS3D> <workdir> <base-ini>}"
WORKDIR="${2:?missing working directory}"
BASE_INI="${3:?missing base ini}"
THREADS="${OMP_NUM_THREADS:-4}"

METHODS=(SIMPLEC SIMPLER SIMPLERR IDEAL)

cd "$WORKDIR"

# --- small helpers ------------------------------------------------------
now() { date +%s.%N; }
sub() { awk -v a="$1" -v b="$2" 'BEGIN{printf "%.3f", a-b}'; }
# Extract the final Darcy permeability (last 'Permeability K' line).
final_K() { grep 'Permeability K' "$1" 2>/dev/null | tail -1 | awk '{print $NF}'; }
# Extract the number of outer iterations from the termination line.
final_iters() { grep -E 'terminated in [0-9]+ iteration' "$1" 2>/dev/null | tail -1 \
                 | sed -E 's/.*terminated in ([0-9]+) iteration.*/\1/'; }
# Whether the log reports convergence.
converged() { grep -q 'converged' "$1" 2>/dev/null; }

# --- header -------------------------------------------------------------
printf '\nAlgorithm comparison on %s  (threads: %s)\n' "$(basename "$BASE_INI")" "$THREADS"
printf '%-10s %16s %10s %12s %12s\n' "method" "K (m^2)" "iters" "time (s)" "converged"
printf '%-10s %16s %10s %12s %12s\n' "--------" "--------" "-----" "--------" "---------"

fail=0
declare -A K IT T CONV

for m in "${METHODS[@]}"; do
    ini="$(mktemp "${TMPDIR:-/tmp}/ns3d_cmp.XXXXXX.ini")"
    # Override only the 'method' line; keep every other parameter.
    sed -E "s/^method[[:space:]]*=.*/method = $m/" "$BASE_INI" > "$ini"

    log="cmp_${m}.log"
    start="$(now)"
    run_ok=0
    if OMP_NUM_THREADS="$THREADS" "$NS3D" "$ini" > "$log" 2>&1; then
        run_ok=1
    fi
    t="$(sub "$(now)" "$start")"
    rm -f "$ini"

    if [[ "$run_ok" -eq 1 ]]; then
        K[$m]="$(final_K "$log")"
        IT[$m]="$(final_iters "$log")"
        T[$m]="$t"
        if converged "$log"; then CONV[$m]="yes"; else CONV[$m]="no"; fi
    else
        K[$m]="?"; IT[$m]="?"; T[$m]="$t"; CONV[$m]="?"
        fail=1
    fi

    printf '%-10s %16s %10s %12s %12s\n' \
        "$m" "${K[$m]}" "${IT[$m]}" "${T[$m]}" "${CONV[$m]}"
done

# --- validation ---------------------------------------------------------
printf '\n'
for m in "${METHODS[@]}"; do
    k="${K[$m]:-}"
    if [[ "$k" == "?" ]] || ! awk -v v="$k" 'BEGIN{exit !(v+0 > 0)}'; then
        printf '  ERROR: %s did not produce a finite, positive permeability (K=%s)\n' "$m" "$k"
        fail=1
    fi
done

nconv=0
for m in "${METHODS[@]}"; do [[ "${CONV[$m]:-}" == "yes" ]] && nconv=$((nconv+1)); done
printf 'Converged (RMR < simpletol): %d / %d\n' "$nconv" "${#METHODS[@]}"
printf 'Per-run logs: %s\n' "cmp_SIMPLEC.log cmp_SIMPLER.log cmp_SIMPLERR.log cmp_IDEAL.log"

exit "$fail"
