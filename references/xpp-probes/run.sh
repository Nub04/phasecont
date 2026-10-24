#!/usr/bin/env bash
# Runs every probe in this directory through XPPAUT 8.0 (when present) and
# through pc-cli, and compares the results with the header lines of the probe.
#
#   XPPAUT=/path/to/xppaut   XPPAUT program (default: xppaut on PATH)
#   PC_CLI=/path/to/pc-cli   pc-cli binary (default: target/debug/pc-cli)
#
# Exit status 1 if any check fails or no probe is found, else 0. A half whose
# program is missing is skipped with a message, and the last line says so.
set -euo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
root=$(cd "$here/../.." && pwd)

# Resolve both programs to absolute paths now, before any cd.
resolve() {
  local p
  p=$(command -v "$1" 2>/dev/null) || return 1
  [ -x "$p" ] || return 1
  realpath "$p"
}
xpp_bin=$(resolve "${XPPAUT:-xppaut}") || xpp_bin=
pc_bin=$(resolve "${PC_CLI:-$root/target/debug/pc-cli}") || pc_bin=

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

fail=0
skipped=""

num_re='^[-+]?([0-9]+\.?[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?$'
special_re='^[-+]?(nan|inf)$'

# A value is a number, or nan, inf or -inf (any case; -nan is nan).
is_value() {
  local s=${1,,}
  [[ $s =~ $num_re || $s =~ $special_re ]]
}

# 0 if got matches expected: nan and inf by name, an expected 0 within an
# absolute 1e-6, anything else within a relative 1e-6. Both must be values.
same() {
  is_value "$1" && is_value "$2" || return 1
  awk -v e="$1" -v g="$2" '
    function special(s) { s = tolower(s); if (s ~ /^[-+]?nan$/) return "nan"; if (s ~ /^\+?inf$/) return "inf"; if (s ~ /^-inf$/) return "-inf"; return "" }
    function abs(x) { return x < 0 ? -x : x }
    BEGIN {
      se = special(e); sg = special(g)
      if (se != "" || sg != "") exit !(se == sg)
      d = abs(g - e)
      if (e + 0 == 0) exit !(d <= 1e-6)
      exit !(d <= 1e-6 * abs(e))
    }'
}

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

# Runs XPPAUT on $1 in directory $2; leaves $2/out.txt (stdout and stderr) and
# $2/o.dat, and sets xrc to the exit status.
run_xpp() {
  local src=$1 dir=$2
  mkdir -p "$dir"
  cp "$src" "$dir/p.ode"
  # Helper files for the #include probes: the inc/ folder and *.inc files.
  [ -d "$here/inc" ] && cp -r "$here/inc" "$dir/inc"
  cp "$here"/*.inc "$dir/" 2>/dev/null || true
  xrc=0
  (cd "$dir" && timeout 10 "$xpp_bin" p.ode -silent -outfile o.dat >out.txt 2>&1) 2>/dev/null || xrc=$?
}

# XPPAUT 8.0 refuses a bad file with exit status 0, no output file and a
# message on stdout (measured: "illegal expression", "Illegal syntax", "... is
# a duplicate name", "ERROR compiling", "Must have at least one equation", "Bad
# global"). A probe marked "# xpp: hang" passes when XPPAUT runs into the 10 s
# timeout or dies from a signal (an include cycle recurses until it crashes).
error_re='illegal|duplicate|error|must have|probably not|no variable|bad global'

# ---------------------------------------------------------------- XPPAUT half
sanity_xpp() {
  local dir="$work/sanity"
  mkdir -p "$dir"
  printf "# sanity\nx'=0\nx(0)=3\npar a=2\naux v=a\n@ total=0,dt=1\n" > "$dir/sanity.ode"
  run_xpp "$dir/sanity.ode" "$dir/run"
  local row
  row=$(head -n 1 "$dir/run/o.dat" 2>/dev/null | tr -s ' ' | sed 's/^ //;s/ $//') || row=
  [ "$xrc" = 0 ] && [ "$row" = "0 3 2" ]
}

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
  run_xpp "$file" "$dir"

  if [ "$spec" = hang ]; then
    if [ "$xrc" = 124 ] || [ "$xrc" -ge 128 ]; then
      echo "PASS $id expected hang got exit $xrc"
    else
      echo "FAIL $id expected hang got exit $xrc"; fail=1
    fi
    return
  fi
  if [ "$xrc" = 124 ]; then
    echo "FAIL $id expected $spec got timeout"; fail=1; return
  elif [ "$xrc" != 0 ]; then
    echo "FAIL $id expected $spec got exit $xrc"; fail=1; return
  fi

  if [ "$spec" = "error" ]; then
    if [ -s "$dir/o.dat" ]; then
      echo "FAIL $id expected error got $(head -n 1 "$dir/o.dat" | tr -s ' ' | sed 's/^ //;s/ $//')"
      fail=1
    elif grep -Eiq "$error_re" "$dir/out.txt"; then
      echo "PASS $id expected error got error"
    else
      echo "FAIL $id expected error got no output and no XPPAUT message"
      fail=1
    fi
    return
  fi

  local cols
  cols=$(columns "$file")
  local ok=1 gots="" tok key val row idx got why
  for tok in $spec; do
    key=; val=$tok
    if [[ $tok == *=* ]]; then key=${tok%%=*}; val=${tok#*=}; fi
    if [[ $tok == =* ]]; then
      echo "FAIL $id expected $spec got - (empty column key in '$tok')"
      fail=1
      return
    fi
    if ! is_value "$val"; then
      echo "FAIL $id expected $spec got - (malformed # xpp: value '$tok')"
      fail=1
      return
    fi
    row=1
    if [[ $key == *@* ]]; then
      row=${key#*@}; key=${key%@*}
      if ! [[ $row =~ ^[1-9][0-9]*$ ]]; then
        echo "FAIL $id expected $spec got - (malformed row in '$tok')"
        fail=1
        return
      fi
    fi
    if [ -z "$key" ]; then
      idx=$(printf '%s\n' "$cols" | awk '!/^@state/ { print NR + 1; exit }')
    elif [ "$key" = t ]; then
      idx=1
    else
      idx=$(printf '%s\n' "$cols" | awk -v n="$key" '$0 == n { print NR + 1; exit }')
    fi
    why=
    if [ -z "${idx:-}" ]; then
      why=nocolumn; got=nocolumn
    else
      got=$(awk -v r="$row" -v c="$idx" 'NR == r { print $c; exit }' "$dir/o.dat" 2>/dev/null) || got=
      if [ -z "$got" ]; then why=norow; got=norow; fi
    fi
    if [ -n "$key" ]; then gots="$gots ${tok%%=*}=$got"; else gots="$gots $got"; fi
    if [ -n "$why" ] || ! same "$val" "$got"; then ok=0; fi
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
# The codes come from the pretty-printed JSON of "pc-cli parse --json"; the
# patterns allow any blanks around the colon.
check_pc() {
  local file=$1 id want json rc=0 codes got
  id=$(basename "$file" .ode)
  want=$(sed -n 's/^# pc:[[:space:]]*//p' "$file" | head -n 1 | sed 's/[[:space:]]*$//')
  if [ -z "$want" ]; then
    echo "FAIL $id pc expected (no # pc: line) got -"
    fail=1
    return
  fi
  json=$("$pc_bin" parse "$file" --json 2>/dev/null) || rc=$?
  codes=$(printf '%s\n' "$json" | grep -Eo '"code" *: *"ODE[0-9]+"' | grep -Eo 'ODE[0-9]+' | awk '!seen[$0]++' | paste -sd, -) || true
  got=${codes:-ok}
  local want_rc=0
  if printf '%s\n' "$json" | grep -Eq '"severity" *: *"error"'; then want_rc=1; fi
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
if [ "${#files[@]}" = 0 ]; then
  echo "FAIL no probes found in $here"
  exit 1
fi

if [ -z "$xpp_bin" ]; then
  echo "SKIP xppaut not found (${XPPAUT:-xppaut}); the XPPAUT half is not run"
  skipped="xpp half skipped: xppaut not found"
elif ! sanity_xpp; then
  echo "FAIL xppaut sanity: $xpp_bin did not run a known-good model (exit $xrc); the XPPAUT half is not run"
  fail=1
  skipped="xpp half skipped: xppaut sanity run failed"
else
  for f in "${files[@]}"; do check_xpp "$f"; done
fi

if [ -z "$pc_bin" ]; then
  echo "SKIP pc-cli not found (${PC_CLI:-$root/target/debug/pc-cli}); run 'cargo build -p pc-cli' or set PC_CLI"
  skipped="${skipped:+$skipped; }pc half skipped: pc-cli not found"
else
  for f in "${files[@]}"; do check_pc "$f"; done
fi

if [ "$fail" = 0 ]; then
  echo "all checks passed${skipped:+ ($skipped)}"
else
  echo "some checks failed${skipped:+ ($skipped)}"
fi
exit "$fail"
