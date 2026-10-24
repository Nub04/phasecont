# XPPAUT probe files

This directory holds our own small `.ode` models, each of which asks XPPAUT 8.0 one question about its input language: how it ranks operators, where it accepts a prefix minus, what it does with blanks, which builtins exist, what a statement keyword really matches. The answers are the measured values below. They are the evidence behind the rules in `crates/pc-ode/docs/language.md`, and `run.sh` re-measures them whenever XPPAUT is installed. The files are written from scratch. None of them comes from the XPPAUT distribution or its manual.

Measurements: 2026-10-23, `xppaut -version` prints `XPPAUT Version 8.0`, installed at `/usr/local/sbin/xppaut`. XPPAUT computes in double precision but writes its output file with float32 precision, so every measured value carries about seven significant digits (`0.15` is written as `0.15000001`). The runner compares with a relative tolerance of 1e-6.

## How to run

```
cargo build -p pc-cli
references/xpp-probes/run.sh
```

`run.sh` runs each probe through `xppaut file.ode -silent -outfile o.dat` in a fresh temporary directory (with `timeout 10`), reads the first output row, and compares it with the `# xpp:` line. It then runs `pc-cli parse --json` on the probe and compares the diagnostic codes with the `# pc:` line. It prints one line per probe and exits 1 if any check fails. `XPPAUT` and `PC_CLI` override the two program paths. Without `xppaut` the runner prints a note and checks only the PhaseCont half; `cargo test -p pc-cli --test probes` checks that half too. XPPAUT 8.0 refuses a bad file with exit status 0, writes no `o.dat` and prints a message on stdout (`illegal expression`, `Illegal syntax`, `... is a duplicate name`, `ERROR compiling`, `Must have at least one equation`). A probe marked `# xpp: error` passes only on that signal. A timeout, a nonzero exit status, a missing column or row, or a malformed `# xpp:` value is a FAIL. Before the first probe the runner runs one known-good model, and if that fails it prints `FAIL xppaut sanity` and skips the XPPAUT half.

## Header format

Line 1 of every probe is a `#` comment, `# <id>: <question>`, because XPPAUT 8.0 fails on a continued statement in the first line. The next lines are read by the runner.

| Line | Meaning |
|---|---|
| `# xpp: <value>` | the first aux value in the first output row |
| `# xpp: error` | XPPAUT refuses the file and writes no output |
| `# xpp: a=1 b=2` | one value per named aux column |
| `# xpp: v@2=14`, `# xpp: t@2=0.5` | `<column>@N` reads row N of the output instead of row 1 (`t` is the time column) |
| `# states: N` | number of states, when it is not 1 (columns are `t`, states, then aux in file order) |
| `# pc: ok` | `pc-cli parse` gives no diagnostic |
| `# pc: ODE0010,ODE0014` | the diagnostic codes `pc-cli parse` gives, in order of first appearance |
| `# open: ...` | a value that P2 still has to decide |

## Results

In the status column, "matches" means PhaseCont agrees with XPPAUT (same value, or both reject the input), "PhaseCont choice" means PhaseCont deliberately differs or adds a diagnostic, and "mismatch, see report" means a measured value contradicts a ruling, and "open for P2" means the rule is not settled. `nan` and `inf` are matched by name. The PhaseCont column gives the diagnostic that `pc-cli parse` prints today, or, for a probe with no diagnostic, the value `pc-ode` will compute once P2 evaluates expressions.

| Probe | Question | Construct | XPPAUT 8.0 | PhaseCont | Status |
|---|---|---|---|---|---|
| `array-bare-index` | Does a bare [j+1] in an expression expand to an integer? | `aux w[1..2]=[j+1]` | w1=2 w2=3 | no diagnostic; computes w1=2 w2=3 | matches |
| `array-block` | Does a %[1..2] block copy its body per index? | `%[1..2]` block with `y[j]` and `[j]` | v1=1 v2=2 | no diagnostic; computes v1=1 v2=2 | matches |
| `array-range` | Does x[1..3]'=-x[j] with x[1..3](0)=[j] declare three states? | `x[1..3]'=-x[j]`, `x[1..3](0)=[j]`, `aux v=x3+10` | 13 | no diagnostic; computes 13 | matches |
| `at-nospace` | Does @ total=1,dt=.5 set dt? | `@ total=1,dt=.5` | t@2=0.5 | no diagnostic; computes t@2=0.5 | matches |
| `at-spaces` | Does @ total = 1, dt = .5 set dt? | `@ total = 1, dt = .5` | t@2=0.05 | ODE0025 warning, XPPAUT ignores the option; PhaseCont reads it | PhaseCont choice |
| `at-unknown` | Is an unknown @ option accepted? | `@ ...,nosuchopt=1` | 13 | ODE0024 warning, unknown option | PhaseCont choice |
| `blank-decimal` | Does XPPAUT delete a blank inside a decimal (1. 5)? | `1. 5` | 1.5 | ODE0012 error, XPPAUT joins the words; fix-it deletes the blank | PhaseCont choice |
| `blank-digits` | Does XPPAUT delete a blank between digits (1 2)? | `1 2` | 12 | ODE0012 error, XPPAUT joins the words; fix-it deletes the blank | PhaseCont choice |
| `blank-eqeq` | Does XPPAUT join = = into == (3= =3)? | `3= =3` | 1 | ODE0012 error, XPPAUT joins the words; fix-it deletes the blank | PhaseCont choice |
| `blank-le` | Does XPPAUT join < = into <= (3< =3)? | `3< =3` | 1 | ODE0012 error, XPPAUT joins the words; fix-it deletes the blank | PhaseCont choice |
| `blank-name-name` | Does a blank between two names join them (a x)? | `a x` with `par ax=11` | 11 | ODE0012 error, XPPAUT joins the words; fix-it deletes the blank | PhaseCont choice |
| `blank-name` | Does XPPAUT delete a blank inside a name (si n(0))? | `si n(0)` | 0 | ODE0039 error, undeclared name; ODE0012 error, XPPAUT joins the words; fix-it deletes the blank | PhaseCont choice |
| `blank-num-name` | Does a blank between a number and a name join them (2 x)? | `2 x` | error (no output) | ODE0012 error, XPPAUT joins the words; fix-it deletes the blank | matches |
| `blank-starstar` | Does XPPAUT join * * into ** (x * * 2)? | `x * * 2` | 9 | ODE0012 error, XPPAUT joins the words; fix-it deletes the blank | PhaseCont choice |
| `call-atan2-1` | Is atan2 with one argument accepted? | `atan2(1)` | error (no output) | ODE0048 error, wrong argument count | matches |
| `call-max3` | Is max with three arguments accepted? | `max(1,2,3)` | error (no output) | ODE0048 error, wrong argument count | matches |
| `call-sin-0` | Is sin with no argument accepted? | `sin()` | error (no output) | ODE0048 error, wrong argument count | matches |
| `case-keywords` | Is case ignored in keywords and names? | `PAR A=2`, `AUX V=a` | 2 | no diagnostic; computes 2 | matches |
| `comment-at` | Is a trailing comment after an @ list accepted, and are its options applied? | `@ total=0,dt=1 # c` | t@2=0.5 | no diagnostic; computes t@2=0.5 | matches |
| `comment-fixed` | Is a trailing comment after a fixed quantity accepted? | `b=2 # c` | error (no output) | ODE0022 error, comment after this statement | matches |
| `comment-init` | Is a trailing comment after an init list accepted? | `init x=3 # c` | error (no output) | ODE0022 error, comment after this statement | matches |
| `comment-number` | Is a trailing comment after a number list accepted? | `number z=2 # c` | 2 | no diagnostic; computes 2 | matches |
| `comment-ode` | Is a trailing comment after an ODE accepted? | `x'=0 # c` | error (no output) | ODE0022 error, comment after this statement | matches |
| `comment-par` | Is a trailing comment after a par list accepted? | `par a=2 # c` | 2 | no diagnostic; computes 2 | matches |
| `const-e-name` | Is e an ordinary name (par e=2)? | `par e=2`, `aux v=e` | 2 | no diagnostic; computes 2 | matches |
| `const-pi` | Is pi a constant? | `pi` | 3.1415927 | no diagnostic; computes 3.1415927 | matches |
| `cont-aux` | Does a backslash continue an aux line? | `aux v=1+\` then `2` | 3 | no diagnostic; computes 3 | matches |
| `cont-ode` | Does a backslash continue an ODE line (not line 1)? | `x'=0+\` then `0` | 13 | no diagnostic; computes 13 | matches |
| `cont-par` | Does a backslash continue a par line? | `par a=1,\` then `b=2` | va=1 vb=2 | no diagnostic; computes va=1 vb=2 | matches |
| `decl-init-expr` | What does x(0)=1+2 declare? | `x(0)=1+2` | 11 | ODE0023 error, value must be a number literal | PhaseCont choice |
| `decl-init-t` | What does x(0)=t+2 declare? | `x(0)=t+2` | 10 | ODE0023 error, value must be a number literal | PhaseCont choice |
| `decl-number-expr` | What does number z=1+1 declare? | `number z=1+1` | 1 | ODE0023 error, value must be a number literal | PhaseCont choice |
| `decl-par-expr` | What does par a=1+1 declare? | `par a=1+1` | 1 | ODE0023 error, value must be a number literal | PhaseCont choice |
| `div-by-zero` | What is 1/0? | `1/0` | 4.4843049e14 | no diagnostic; keeps IEEE for now | open for P2 |
| `div-zero-zero` | What is 0/0? | `0/0` | 0 | no diagnostic; keeps IEEE for now | open for P2 |
| `eqspace-aux` | Are blanks around = accepted in aux? | `aux v = 2` | 2 | no diagnostic; computes 2 | matches |
| `eqspace-dxdt` | Are blanks around = accepted in dx/dt=? | `dx/dt = 1` | v@2=14 | no diagnostic; computes v@2=14 | matches |
| `eqspace-func` | Are blanks around = accepted in a function definition? | `f(u) = u*2` | 8 | no diagnostic; computes 8 | matches |
| `eqspace-init` | Are blanks around = accepted in init? | `init x = 4` | 14 | no diagnostic; computes 14 | matches |
| `eqspace-number` | Are blanks around = accepted in number? | `number z = 5` | 5 | no diagnostic; computes 5 | matches |
| `eqspace-ode` | Are blanks around = accepted in x'=? | `x' = 1` | v@2=14 | no diagnostic; computes v@2=14 | matches |
| `eqspace-par-list` | Do blanks separate par entries (par a=1 b=2)? | `par a=1 b=2` | va=1 vb=2 | no diagnostic; computes va=1 vb=2 | matches |
| `eqspace-par` | Are blanks around = accepted in par? | `par a = 2` | 2 | no diagnostic; computes 2 | matches |
| `eqspace-x0` | Are blanks around = accepted in x(0)=? | `x(0) = 3` | 13 | no diagnostic; computes 13 | matches |
| `exp-overflow` | What is exp(710)? | `exp(710)` | inf | no diagnostic; computes inf | matches |
| `fixed-aux-above` | Does an aux above a fixed quantity see its final value? | `aux v=b` above `b=3` | 5 | no diagnostic; computes 5 | matches |
| `fixed-forward` | What does a fixed quantity read from a later line? | `b=c+1` above `c=2` | 1 | ODE0042 error, fixed quantity defined on a later line | PhaseCont choice |
| `fn-atan2` | Which argument order does atan2 use? | `atan2(1,0)`, `atan2(-1,-1)` | v1=1.5707964 v2=-2.3561945 | no diagnostic; computes v1=1.5707964 v2=-2.3561945 | matches |
| `fn-ceil` | Is ceil accepted? | `ceil(2.5)` | error (no output) | ODE0013 warning, XPPAUT 8.0 rejects this | PhaseCont choice |
| `fn-delay` | Does delay(x,1) evaluate at t=0? | `10*delay(x,1)` | 30 | ODE0054 warning, `delay` unsupported until v2 | PhaseCont choice |
| `fn-fact` | Is fact a builtin (fact(3))? | `fact(3)` | error (no output) | ODE0039 error, undeclared name | matches |
| `fn-flr` | Is flr a floor? | `flr(2)`, `flr(-2)`, `flr(-2.5)` | v1=2 v2=-2 v3=-3 | no diagnostic; computes v1=2 v2=-2 v3=-3 | matches |
| `fn-gamma` | Is gamma a builtin (gamma(3))? | `gamma(3)` | error (no output) | ODE0039 error, undeclared name | matches |
| `fn-int` | Is int a builtin (int(2.5))? | `int(2.5)` | error (no output) | ODE0039 error, undeclared name | matches |
| `fn-lgamma` | What is lgamma(5)? | `lgamma(5)` | 3.1780539 | no diagnostic; computes 3.1780539 | matches |
| `fn-ln` | Is ln natural (ln(100))? | `ln(100)` | 4.6051702 | no diagnostic; computes 4.6051702 | matches |
| `fn-log10` | What is log10(100)? | `log10(100)` | 2 | no diagnostic; computes 2 | matches |
| `fn-log` | Is log natural (log(100))? | `log(100)` | 4.6051702 | no diagnostic; computes 4.6051702 | matches |
| `fn-mod` | What does mod give for negative, fractional and zero arguments? | `mod(-7,3)`, `mod(7,-3)`, `mod(-7,-3)`, `mod(7.5,2)`, `mod(-7.5,2)`, `mod(5,2.5)`, `mod(1,0)` | v1=2 v2=1 v3=-4 v4=1.5 v5=0.5 v6=0 v7=nan | no diagnostic; computes v1=2 v2=1 v3=-4 v4=1.5 v5=0.5 v6=0 v7=nan | matches |
| `fn-shift` | Does shift(x,0) evaluate? | `10*shift(x,0)` | 30 | ODE0055 warning, `shift` unsupported | PhaseCont choice |
| `heav-negzero` | What is heav(-0)? | `heav(-0)` | 1 | no diagnostic; computes 1 | matches |
| `heav-zero` | What is heav(0)? | `heav(0)` | 1 | no diagnostic; computes 1 | matches |
| `hom-bcs` | Does XPPAUT 8.0 accept a hom_bcs line? | `hom_bcs 1` | 5 | ODE0035 warning, `hom_bcs` unsupported | PhaseCont choice |
| `if-lazy` | Does if(1)then(1)else(1/0) evaluate both branches? | `if(1)then(1)else(1/0)` | 1 | no diagnostic; keeps IEEE for now | open for P2 |
| `implicit-div` | Is implicit multiplication after a division accepted (a/2v)? | `a/2v0`, a=8 | error (no output) | ODE0013 warning, XPPAUT 8.0 rejects this | PhaseCont choice |
| `implicit-num-name` | Is implicit multiplication of a number and a name accepted (2x)? | `2x` | error (no output) | ODE0013 warning, XPPAUT 8.0 rejects this | PhaseCont choice |
| `implicit-num-paren` | Is implicit multiplication before a parenthesis accepted (2(x))? | `2(x)` | error (no output) | ODE0013 warning, XPPAUT 8.0 rejects this | PhaseCont choice |
| `kw-fixed-letters` | Are p=, a=, w=, i=, d=, g=, b=, m= fixed quantities? | `p = 5`, `a = 6`, `w = 7`, `i = 8`, `d = 9`, `g = 10`, `b = 11`, `m = 12` | vp=5 va=6 vw=7 vi=8 vd=9 vg=10 vb=11 vm=12 | no diagnostic; computes vp=5 va=6 vw=7 vi=8 vd=9 vg=10 vb=11 vm=12 | matches |
| `kw-p-nospace` | Is p=5 (no blanks) a fixed quantity? | `p=5` | 5 | no diagnostic; computes 5 | matches |
| `kw-p-par` | Does p a=5 declare a parameter? | `p a=5` | 5 | no diagnostic; computes 5 | matches |
| `log-neg` | What is log(-1)? | `log(-1)` | nan | no diagnostic; computes nan | matches |
| `mixed-ode-map` | Is a file with an ODE and a map accepted? | `x'=0` with `y(t+1)=y/2` | 17 | ODE0052 error, ODEs and maps mixed | PhaseCont choice |
| `name-10` | Is a 10-character name accepted? | `par abcdefghij=3` | 9 | no diagnostic; computes 9 | matches |
| `name-11-collide` | Do two declared names that agree in 10 characters collide when neither is used? | `par abcdefghijk=1`, `par abcdefghijl=2`, neither used | 17 | ODE0013 warning, XPPAUT 8.0 rejects this | mismatch, see report |
| `name-11-collide-used` | Can two names that agree in 10 characters be used in a formula? | `aux v=abcdefghijk+abcdefghijl` | error (no output) | ODE0013 warning, XPPAUT 8.0 rejects this | PhaseCont choice |
| `name-11` | Can a single 11-character parameter be used in a formula? | `par abcdefghijk=5`, `aux v=abcdefghijk` | error (no output) | ODE0013 warning, XPPAUT 8.0 rejects this | PhaseCont choice |
| `name-arg1-outside` | What does arg1 read as outside a function? | `arg1` | 0 | ODE0050 error, `arg1` outside a function | PhaseCont choice |
| `name-t-outside` | What does t read as in an aux at t=0? | `t` | 0 | no diagnostic; computes 0 | matches |
| `num-dot-exp` | Is 1.e2 a number? | `1.e2` | 100 | no diagnostic; computes 100 | matches |
| `num-exp-dot` | Is 1e2.5 accepted? | `1e2.5` | error (no output) | ODE0001 error, malformed number | matches |
| `num-exp-plus` | Is 1e+2 a number? | `1e+2` | 100 | no diagnostic; computes 100 | matches |
| `num-exp-upper` | Is 1E2 a number? | `1E2` | 100 | no diagnostic; computes 100 | matches |
| `num-hex` | Is 0x10 accepted? | `0x10` | error (no output) | ODE0001 error, malformed number | matches |
| `num-leading-dot` | Is .5 a number? | `.5` | 0.5 | no diagnostic; computes 0.5 | matches |
| `num-trailing-dot` | Is 5. a number? | `5.` | 5 | no diagnostic; computes 5 | matches |
| `num-underscore` | Is 1_000 accepted? | `1_000` | error (no output) | ODE0001 error, malformed number | matches |
| `op-bang` | Is a bare ! accepted (!1)? | `!1` | error (no output) | ODE0007 error, not an operator | matches |
| `op-ne` | Is != accepted (1!=2)? | `1!=2` | error (no output) | ODE0007 error, not an operator | matches |
| `op-single-eq` | Is a single = inside a formula accepted (1=2)? | `1=2` | error (no output) | ODE0008 error, write `==` | matches |
| `par-duplicate` | Is a duplicate par accepted? | `par a=1` then `par a=2` | error (no output) | ODE0041 error, duplicate name | matches |
| `pow-neg-base` | What is (-2)^3? | `(-2)^3` | -8 | no diagnostic; computes -8 | matches |
| `pow-neg-frac` | What is (-8)^(1/3)? | `(-8)^(1/3)` | nan | no diagnostic; computes nan | matches |
| `pow-starstar` | Does ** equal ^ (2**3)? | `2**3` | 8 | no diagnostic; computes 8 | matches |
| `pow-zero-zero` | What is 0^0? | `0^0` | 1 | no diagnostic; computes 1 | matches |
| `prec-add-and` | Does a+b&c read as a+(b&c)? | `a+b&c`, a=1, b=2, c=3 | 2 | ODE0014 warning, parsed as XPPAUT does; fix-it adds parentheses | PhaseCont choice |
| `prec-add-or` | Does 2+1\|0 read as (2+1)\|0? | `2+1\|0` | 1 | ODE0014 warning, parsed as XPPAUT does; fix-it adds parentheses | PhaseCont choice |
| `prec-and-add` | Does a&b+c read as (a&b)+c? | `a&b+c`, a=1, b=2, c=3 | 4 | ODE0014 warning, parsed as XPPAUT does; fix-it adds parentheses | PhaseCont choice |
| `prec-and-mul` | Does 1&1*2 read as (1&1)*2? | `1&1*2` | 2 | ODE0014 warning, parsed as XPPAUT does; fix-it adds parentheses | PhaseCont choice |
| `prec-cmp-add` | Does a comparison bind tighter than + (3>2+5)? | `3>2+5` | 6 | ODE0014 warning, parsed as XPPAUT does; fix-it adds parentheses | PhaseCont choice |
| `prec-cmp-chain` | Is a comparison chain left associative (3>2>1)? | `3>2>1` | 0 | no diagnostic; computes 0 | matches |
| `prec-eq-gt` | Do == and > share one level (2==2>1)? | `2==2>1` | 0 | no diagnostic; computes 0 | matches |
| `prec-mul-and` | Do * and & share one level (2*1&1)? | `2*1&1` | 1 | ODE0014 warning, parsed as XPPAUT does; fix-it adds parentheses | PhaseCont choice |
| `prec-mul-cmp` | Does a comparison bind tighter than * (2*3>5)? | `2*3>5` | 0 | ODE0014 warning, parsed as XPPAUT does; fix-it adds parentheses | PhaseCont choice |
| `prec-neg-and` | Does prefix minus bind tighter than & (-2&1)? | `-2&1` | 1 | no diagnostic; computes 1 | matches |
| `prec-neg-cmp` | Does prefix minus bind looser than a comparison (-x>2)? | `-x>2` | -1 | ODE0014 warning, parsed as XPPAUT does; fix-it adds parentheses | PhaseCont choice |
| `prec-neg-mul` | Does prefix minus bind tighter than * (-x*a)? | `-x*a` | -6 | no diagnostic; computes -6 | matches |
| `prec-neg-pow` | Does prefix minus bind looser than ^ (-x^2)? | `-x^2` | -9 | no diagnostic; computes -9 | matches |
| `prec-not-add` | Does not bind tighter than + (not 0+1)? | `not 0+1` | 2 | no diagnostic; computes 2 | matches |
| `prec-not-cmp` | Does not bind looser than a comparison (not 1>2)? | `not 1>2` | 1 | no diagnostic; computes 1 | matches |
| `prec-or-add` | Do \| and binary + share one level, left to right (1\|0+1)? | `1\|0+1` | 2 | ODE0014 warning, parsed as XPPAUT does; fix-it adds parentheses | PhaseCont choice |
| `prec-pow-cmp` | Does ^ bind tighter than a comparison (2^3>9)? | `2^3>9` | 0 | no diagnostic; computes 0 | matches |
| `prec-pow-left` | Is ^ left associative (2^3^2)? | `2^3^2` | 64 | ODE0014 warning, parsed as XPPAUT does; fix-it adds parentheses | PhaseCont choice |
| `prec-starstar-left` | Is ** left associative (2**3**2)? | `2**3**2` | 64 | ODE0014 warning, parsed as XPPAUT does; fix-it adds parentheses | PhaseCont choice |
| `prefix-add-neg` | Is prefix minus after + accepted (2+-1)? | `2+-1` | error (no output) | ODE0010 error, prefix only at the start, after `(` or `,`; fix-it adds parentheses | matches |
| `prefix-and-not` | What does not after & compute (0&not 0)? | `0&not 0` | 1 | ODE0010 error, prefix only at the start, after `(` or `,`; fix-it adds parentheses | PhaseCont choice |
| `prefix-comma-neg` | Is prefix minus after a comma accepted (max(1,-3))? | `max(1,-3)` | 1 | no diagnostic; computes 1 | matches |
| `prefix-double-neg` | Is a doubled prefix minus accepted (--2)? | `--2` | error (no output) | ODE0010 error, prefix only at the start, after `(` or `,`; fix-it adds parentheses | matches |
| `prefix-mul-neg` | Is prefix minus after * accepted (2*-3)? | `2*-3` | error (no output) | ODE0010 error, prefix only at the start, after `(` or `,`; fix-it adds parentheses | matches |
| `prefix-mul-not` | What does not after * compute (5*not 0)? | `5*not 0` | 1 | ODE0010 error, prefix only at the start, after `(` or `,`; fix-it adds parentheses | PhaseCont choice |
| `prefix-neg-not` | What does minus before not compute (-not 0)? | `-not 0` | 1 | ODE0010 error, prefix only at the start, after `(` or `,`; fix-it adds parentheses | PhaseCont choice |
| `prefix-not-cmp-and` | Is x>2&not x>4 accepted? | `x>2&not x>4` | 1 | ODE0010 error, prefix only at the start, after `(` or `,`; fix-it adds parentheses | PhaseCont choice |
| `prefix-not-not` | What does a doubled not compute (not not 0)? | `not not 0` | 1 | ODE0010 error, prefix only at the start, after `(` or `,`; fix-it adds parentheses | PhaseCont choice |
| `prefix-not-start` | Is not at the start of a formula accepted (not 0)? | `not 0` | 1 | no diagnostic; computes 1 | matches |
| `prefix-paren-neg` | Is prefix minus after ( accepted (2*(-3))? | `2*(-3)` | -6 | no diagnostic; computes -6 | matches |
| `prefix-plus` | Is a unary plus accepted (+2)? | `+2` | error (no output) | ODE0011 error, unary `+` | matches |
| `prefix-pow-neg` | Is prefix minus after ^ accepted (2^-1)? | `2^-1` | error (no output) | ODE0010 error, prefix only at the start, after `(` or `,`; fix-it adds parentheses | matches |
| `prefix-pow-not` | What does not after ^ compute (2^not 0)? | `2^not 0` | 1 | ODE0010 error, prefix only at the start, after `(` or `,`; fix-it adds parentheses | PhaseCont choice |
| `prefix-sub-neg` | Is x - -1 accepted? | `x - -1` | error (no output) | ODE0010 error, prefix only at the start, after `(` or `,`; fix-it adds parentheses | matches |
| `reserved-sin` | Can sin be declared as a parameter? | `par sin=2` | error (no output) | ODE0040 error, reserved name; ODE0049 error, function name without a call | matches |
| `reserved-t` | Can t be declared as a state? | `t'=0` | error (no output) | ODE0040 error, reserved name | matches |
| `sign-zero` | What is sign(0)? | `sign(0)` | 0 | no diagnostic; computes 0 | matches |
| `sqrt-neg` | What is sqrt(-1)? | `sqrt(-1)` | nan | no diagnostic; computes nan | matches |
| `stmt-abbrev` | Does only the first letter of par count (pzz a=2)? | `pzz a=2` | 2 | no diagnostic; computes 2 | matches |
| `truth-and-half` | Is 0.5 true for & (0.5&1)? | `0.5&1` | 1 | no diagnostic; computes 1 | matches |
| `truth-and-neghalf` | Is -0.5 true for & (-0.5&1)? | `-0.5&1` | 1 | no diagnostic; computes 1 | matches |
| `truth-if-neg` | Is -1 true for if (if(-1)then(7)else(8))? | `if(-1)then(7)else(8)` | 7 | no diagnostic; computes 7 | matches |

## Notes

- Prefix probes. XPPAUT accepts a prefix `-` only at the start of an expression, after `(` or after `,`. It accepts `not` after an operator but computes a wrong value there (`0&not 0` gives 1). PhaseCont reports an error with a fix-it for both,
- `name-11`, `name-11-collide`, `name-11-collide-used`. XPPAUT 8.0 accepts a declared name longer than 10 characters but cannot read it back in a formula: it stops with `illegal expression: K`, which shows it keeps only the first 10 characters when it parses a formula. Two declared names that agree in their first 10 characters did not collide in any form measured (both unused, both used in a formula, one alone, a fixed-quantity pair, and `abcdefghij` with `abcdefghijk`). Only using such a name in a formula fails. The row `name-11-collide` is marked "mismatch, see report" because the ruling in task-12-measured.md says such names collide, and the measurement does not show it. pc-ode is unchanged, and the report lists the contradiction.
- `name-arg1-outside`. XPPAUT reads `arg1` as 0 outside a function. PhaseCont reports an error (ODE0050).
- `if-lazy`, `div-by-zero`, `div-zero-zero`. Whether XPPAUT evaluates both branches of `if` cannot be seen from values. Division by zero is guarded in XPPAUT (`1/0` gives 4.4843049e14, `0/0` gives 0). P2 decides both.
