#!/usr/bin/env bash
# Runs every probe in this directory through XPPAUT 8.0 (when present) and
# through pc-cli, and compares the results with the header lines of the probe.
#
#   XPPAUT=/path/to/xppaut   XPPAUT program (default: xppaut on PATH)
#   PC_CLI=/path/to/pc-cli   pc-cli binary (default: target/debug/pc-cli)
#
# Exit status 1 if any check fails, else 0.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../.." && pwd)
pc_cli=${PC_CLI:-$root/target/debug/pc-cli}
xpp=${XPPAUT:-xppaut}

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

fail=0

# Names of the columns after t: states ("# states: N", default 1), then aux
# names in file order. A range such as "aux w[1..2]" counts as w1, w2.
columns() {
  local file=$1 n
  n=$(sed -n 's/^# states:[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$file" | head -n 1)
  awk -v nstates="${n:-1}" '
    BEGIN { for (i = 1; i <= nstates; i++) print "@state" i }
    {
      line = tolower($0)
      if (match(line, /^aux[ \t]+[a-z0-9_]+(\[[0-9]+\.\.[0-9]+\])?/)) {
        s = substr(line, RSTART, RLENGTH)
        sub(/^aux[ \t]+/, "", s)
        if (match(s, /\[[0-9]+\.\.[0-9]+\]/)) {
          base = substr(s, 1, RSTART - 1)
          r = substr(s, RSTART + 1, RLENGTH - 2)
          split(r, ab, "[.][.]")
          a = ab[1] + 0; b = ab[2] + 0
          step = (a <= b) ? 1 : -1
          for (k = a; ; k += step) { print base k; if (k == b) break }
        } else {
          print s
        }
      }
    }' "$file"
}

# 0 if got matches expected: nan and inf by name, numbers by relative 1e-6
# (absolute 1e-6 when the expected value is below 1 in size).
same() {
  awk -v e="$1" -v g="$2" '
    function special(s) { s = tolower(s); if (s ~ /^[-+]?nan/) return "nan"; if (s ~ /^\+?inf/) return "inf"; if (s ~ /^-inf/) return "-inf"; return "" }
    function abs(x) { return x < 0 ? -x : x }
    BEGIN {
      se = special(e); sg = special(g)
      if (se != "" || sg != "") exit !(se == sg)
      d = abs(g - e); m = abs(e); if (m < 1) m = 1
      exit !(d <= 1e-6 * m)
    }'
}

# ---------------------------------------------------------------- XPPAUT half
check_xpp() {
  local file=$1 id spec
  id=$(basename "$file" .ode)
  spec=$(sed -n 's/^# xpp:[[:space:]]*//p' "$file" | head -n 1 | sed 's/[[:space:]]*$//')
  if [ -z "$spec" ]; then
    echo "FAIL $id expected (no # xpp: line) got -"
    fail=1
    return
  fi
  local dir="$work/x-$id"
  mkdir -p "$dir"
  cp "$file" "$dir/p.ode"
  local rc=0
  (cd "$dir" && timeout 10 "$xpp_bin" p.ode -silent -outfile o.dat >/dev/null 2>&1) || rc=$?
  local have=1
  if [ ! -s "$dir/o.dat" ]; then have=0; fi

  if [ "$spec" = "error" ]; then
    if [ "$have" = 0 ]; then
      echo "PASS $id expected error got error"
    else
      echo "FAIL $id expected error got $(head -n 1 "$dir/o.dat" | tr -s ' ' | sed 's/^ //;s/ $//')"
      fail=1
    fi
    return
  fi
  if [ "$have" = 0 ]; then
    echo "FAIL $id expected $spec got error"
    fail=1
    return
  fi

  local cols
  cols=$(columns "$file")
  local ok=1 gots="" tok key val row colname idx got
  for tok in $spec; do
    case $tok in
      *=*) key=${tok%%=*}; val=${tok#*=} ;;
      *) key=; val=$tok ;;
    esac
    row=1
    if [[ $key == *@* ]]; then row=${key#*@}; key=${key%@*}; fi
    if [ -z "$key" ]; then
      # single value: the first aux column
      idx=$(printf '%s\n' "$cols" | awk '!/^@state/ { print NR + 1; exit }')
    elif [ "$key" = t ]; then
      idx=1
    else
      colname=$key
      idx=$(printf '%s\n' "$cols" | awk -v n="$colname" '$0 == n { print NR + 1; exit }')
    fi
    if [ -z "${idx:-}" ]; then
      got="nocolumn"
    else
      got=$(awk -v r="$row" -v c="$idx" 'NR == r { print $c; exit }' "$dir/o.dat")
      [ -n "$got" ] || got="norow"
    fi
    if [ -n "$key" ]; then
      gots="$gots ${tok%%=*}=$got"
    else
      gots="$gots $got"
    fi
    if ! same "$val" "$got" 2>/dev/null; then ok=0; fi
  done
  gots=${gots# }
  if [ "$ok" = 1 ]; then
    echo "PASS $id expected $spec got $gots"
  else
    echo "FAIL $id expected $spec got $gots"
    fail=1
  fi
}

# -------------------------------------------------------------------- pc half
check_pc() {
  local file=$1 id want json rc=0 codes got
  id=$(basename "$file" .ode)
  want=$(sed -n 's/^# pc:[[:space:]]*//p' "$file" | head -n 1 | sed 's/[[:space:]]*$//')
  if [ -z "$want" ]; then
    echo "FAIL $id pc expected (no # pc: line) got -"
    fail=1
    return
  fi
  json=$("$pc_cli" parse "$file" --json 2>/dev/null) || rc=$?
  codes=$(printf '%s\n' "$json" | grep -o '"code": "ODE[0-9]*"' | grep -o 'ODE[0-9]*' | awk '!seen[$0]++' | paste -sd, -) || true
  got=${codes:-ok}
  local want_rc=0
  if printf '%s\n' "$json" | grep -q '"severity": "error"'; then want_rc=1; fi
  if [ "$got" = "$want" ] && [ "$rc" = "$want_rc" ]; then
    echo "PASS $id pc expected $want got $got"
  else
    echo "FAIL $id pc expected $want got $got (exit $rc)"
    fail=1
  fi
}

shopt -s nullglob
files=("$here"/*.ode)
echo "${#files[@]} probes"

xpp_bin=
if command -v "$xpp" >/dev/null 2>&1; then
  xpp_bin=$(command -v "$xpp")
  for f in "${files[@]}"; do check_xpp "$f"; done
else
  echo "SKIP xppaut not found ($xpp); the XPPAUT half is not run"
fi

if [ -x "$pc_cli" ]; then
  for f in "${files[@]}"; do check_pc "$f"; done
else
  echo "SKIP pc-cli not found at $pc_cli; run 'cargo build -p pc-cli' or set PC_CLI"
fi

if [ "$fail" = 0 ]; then echo "all checks passed"; else echo "some checks failed"; fi
exit "$fail"
