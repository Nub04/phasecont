# XPPAUT probe files

This directory holds our own small `.ode` models, each of which asks XPPAUT 8.0 one question about its input language: how it ranks operators, where it accepts a prefix minus, what it does with blanks, which builtins exist, what a statement keyword really matches. The answers are the measured values below. They are the evidence behind the rules in `crates/pc-ode/docs/language.md`, and `run.sh` re-measures them whenever XPPAUT is installed. The files are written from scratch. None of them comes from the XPPAUT distribution or its manual.

Measurements: 2026-10-23 (the `meth-*`, `map-*` and `opt-prefix-*` probes on 2026-11-14), `xppaut -version` prints `XPPAUT Version 8.0`, installed at `/usr/local/sbin/xppaut`. XPPAUT computes in double precision but writes its output file with float32 precision, so every measured value carries about seven significant digits (`0.15` is written as `0.15000001`). The runner compares with a relative tolerance of 1e-6.

## How to run

```
cargo build -p pc-cli
references/xpp-probes/run.sh
```

`run.sh` runs each probe through `xppaut file.ode -silent -outfile o.dat` in a fresh temporary directory (with `timeout 10`), reads the first output row, and compares it with the `# xpp:` line. It then runs `pc-cli parse --json` on the probe and compares the diagnostic codes with the `# pc:` line. It prints one line per probe and exits 1 if any check fails. `XPPAUT` and `PC_CLI` override the two program paths. Without `xppaut` the runner prints a note and checks only the PhaseCont half; `cargo test -p pc-cli --test probes` checks that half too. XPPAUT 8.0 refuses a bad file with exit status 0, writes no `o.dat` and prints a message on stdout (`illegal expression`, `Illegal syntax`, `... is a duplicate name`, `ERROR compiling`, `Must have at least one equation`, `Cant export ... - non existent!`). A probe marked `# xpp: error` passes only on that signal. A timeout, a nonzero exit status, a missing column or row, or a malformed `# xpp:` value is a FAIL. Before the first probe the runner runs one known-good model, and if that fails it prints `FAIL xppaut sanity` and skips the XPPAUT half.

## Header format

Line 1 of every probe is a `#` comment, `# <id>: <question>`, because XPPAUT 8.0 fails on a continued statement in the first line. The next lines are read by the runner.

| Line | Meaning |
|---|---|
| `# xpp: <value>` | the first aux value in the first output row |
| `# xpp: error` | XPPAUT refuses the file and writes no output |
| `# xpp: hang` | XPPAUT runs into the 10 s timeout or dies from a signal (an include cycle, `kw-volt-vector`) |
| `# xpp: a=1 b=2` | one value per named aux column |
| `# xpp: v@2=14`, `# xpp: t@2=0.5` | `<column>@N` reads row N of the output instead of row 1 (`t` is the time column) |
| `# xpp: rows=21` | the output file has exactly that many lines (a stop or a recorded row changes the count) |
| `# states: N` | number of states, when it is not 1 (columns are `t`, states, then aux in file order) |
| `# pc: ok` | `pc-cli parse` gives no diagnostic |
| `# pc: ODE0010,ODE0014` | the diagnostic codes `pc-cli parse` gives, in order of first appearance |
| `# ruling: ...` | a note on how PhaseCont rules this case, with a pointer to `crates/pc-ode/docs/language.md`; the runner ignores it |

The `inc/` folder and the `*.inc` files next to the probes hold the files that the `include-*` probes include. `run.sh` copies them into the run folder, since XPPAUT reads an include relative to its working directory. A `markov` variable counts as one more state column.

## Results

In the status column, "matches" means PhaseCont agrees with XPPAUT (same value, or both reject the input), "PhaseCont choice" means PhaseCont deliberately differs or adds a diagnostic, and "mismatch, see report" means a measured value contradicts a ruling. `nan` and `inf` are matched by name. The PhaseCont column gives the diagnostic that `pc-cli parse` prints today, or, for a probe with no diagnostic, the value `pc-ode` computes.

| Probe | Question | Construct | XPPAUT 8.0 | PhaseCont | Status |
|---|---|---|---|---|---|
| `active-comment` | Does a line that starts with a double quote load and change nothing? | `"this is an active comment` | 1 | ODE0038 warning, line kept as raw text | matches |
| `array-bare-index` | Does a bare [j+1] in an expression expand to an integer? | `aux w[1..2]=[j+1]` | w1=2 w2=3 | no diagnostic; computes w1=2 w2=3 | matches |
| `array-block` | Does a %[1..2] block copy its body per index? | `%[1..2]` block with `y[j]` and `[j]` | v1=1 v2=2 | no diagnostic; computes v1=1 v2=2 | matches |
| `array-range` | Does x[1..3]'=-x[j] with x[1..3](0)=[j] declare three states? | `x[1..3]'=-x[j]`, `x[1..3](0)=[j]`, `aux v=x3+10` | 13 | no diagnostic; computes 13 | matches |
| `at-cont` | Does a backslash-newline in an @ list join `dt=\` with `.5` on the next line? | `@ total=1,dt=\` then `.5` | t@2=0.5 | no diagnostic; computes t@2=0.5 | matches |
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
| `comment-at` | Is a trailing comment after an @ list accepted, and are its options applied? | `@ total=1,dt=.5 # c` | t@2=0.5 | no diagnostic; computes t@2=0.5 | matches |
| `comment-fixed` | Is a trailing comment after a fixed quantity accepted? | `b=2 # c` | error (no output) | ODE0022 error, comment after this statement | matches |
| `comment-init` | Is a trailing comment after an init list accepted? | `init x=3 # c` | error (no output) | ODE0022 error, comment after this statement | matches |
| `comment-number` | Is a trailing comment after a number list accepted? | `number z=2 # c` | 2 | no diagnostic; computes 2 | matches |
| `comment-ode` | Is a trailing comment after an ODE accepted? | `x'=0 # c` | error (no output) | ODE0022 error, comment after this statement | matches |
| `comment-par` | Is a trailing comment after a par list accepted? | `par a=2 # c` | 2 | no diagnostic; computes 2 | matches |
| `const-e-name` | Is e an ordinary name (par e=2)? | `par e=2`, `aux v=e` | 2 | no diagnostic; computes 2 | matches |
| `const-pi` | Is pi a constant? | `pi` | 3.1415927 | no diagnostic; computes 3.1415927 | matches |
| `backslash-active-comment` | Does a backslash at the end of an active comment line swallow the next line? | `"an active comment \` then `x'=1`, `x'=2` | v@3=2 (`x'=1` is lost) | ODE0038 warning only; the next line joins the kept text silently (no ODE0076) | PhaseCont choice |
| `backslash-blanks` | Does a backslash followed by two blanks continue the line (XPPAUT 8.2.2)? | `x'=1+\` and two blanks, then `2` | v@3=3 | no diagnostic; computes v@3=3 | matches |
| `backslash-comment` | Does a backslash at the end of a `#` comment line swallow the next line? | `# a note ... \` then `x'=1`, `x'=2` | v@3=2 (`x'=1` is lost) | ODE0076 warning naming `x'=1`; computes v@3=2 | matches |
| `backslash-comment-blank` | Does a comment backslash followed by a blank line leave the line after the blank intact? | `# a note \`, a blank line, `x'=1` | v@3=1 | no diagnostic; computes v@3=1 | matches |
| `backslash-comment-chain` | Does a swallowed comment line that ends in a backslash swallow the line after it? | `# first \`, `# second \`, `x'=1`, `x'=2` | v@3=2 | one ODE0076 warning; computes v@3=2 | matches |
| `backslash-comment-crlf` | Does a CRLF file behave like an LF file for a comment that ends in a backslash? | the `backslash-comment` file with CRLF line ends | v@3=2 | ODE0076 warning; computes v@3=2 | matches |
| `backslash-comment-inc` | Does a comment backslash on line 1 of an included file swallow line 2? | `#include inc/bs-comment.ode`, then `x'=2` | v@3=2 (the include's `x'=1` is lost) | ODE0076 warning in the included file; computes v@3=2 | matches |
| `backslash-comment-line1` | Does a backslash at the end of line 1 of the main file swallow line 2? | line 1 ends in `\`, line 2 is `x'=1` | v@3=1 (line 1 is exempt) | no diagnostic; computes v@3=1 | matches |
| `backslash-comment-mid` | Does a backslash in the middle of a `#` comment drop the rest of it and swallow the next line? | `# a path like a\b in a comment`, then `x'=1`, `x'=2` | v@3=2 | ODE0076 warning; computes v@3=2 | matches |
| `backslash-eof` | Is a backslash at the very end of the file, with no newline, accepted (XPPAUT 8.2.2)? | last line `@ total=1,dt=0.5,meth=euler\` | v@3=1 | no diagnostic; computes v@3=1 | matches |
| `backslash-eof-text` | Is text after a backslash on the last line, with no newline, dropped? | last line `@ total=1,dt=0.5,meth=euler\ junk` | v@3=1 | ODE0075 warning; computes v@3=1 | matches |
| `backslash-join-digits` | Does a continuation join `1\` and `2` into 12 or into `1 2`? | `x'=1\` then `2` | v@3=12 (no blank at the join) | ODE0012 error (blank between words) | PhaseCont choice |
| `backslash-mixed-blanks` | Does a backslash followed by a space, a tab and a space continue the line (XPPAUT 8.2.2)? | `x'=1+\` and ` \t `, then `2` | v@3=3 | no diagnostic; computes v@3=3 | matches |
| `backslash-tab` | Does a backslash followed by a tab continue the line (XPPAUT 8.2.2)? | `x'=1+\` and a tab, then `2` | v@3=3 | no diagnostic; computes v@3=3 | matches |
| `backslash-text` | What does XPPAUT 8.2.2 do with text after a backslash (`\5`)? | `x'=1+\5` then `2` | v@3=3 (continues, drops the `5`) | ODE0075 warning quoting `5`; computes v@3=3 | matches |
| `backslash-text-comment` | What does XPPAUT 8.2.2 do with a comment after a backslash? | `x'=1+\ # five` then `2` | v@3=3 | ODE0075 warning that calls the text a comment; computes v@3=3 | matches |
| `backslash-text-crlf` | Does a CRLF file behave like an LF file for text after a backslash? | the `backslash-text` file with CRLF line ends | v@3=3 | ODE0075 warning; computes v@3=3 | matches |
| `backslash-trailing-comment-at` | What does a trailing `#` comment ending in a backslash do on an `@` line? | `@ total=1 # note \` then `,dt=0.5,meth=euler` | error | ODE0076 warning; the `@` line reads `@ total=1` | PhaseCont choice |
| `backslash-trailing-comment-par` | What does a trailing `#` comment ending in a backslash do on a par line? | `par a=1 # note \` then `x'=a`, `x'=2` | v@3=2 (`x'=a` joins the par line) | ODE0076 warning; computes v@3=2 | matches |
| `cont-aux` | Does a backslash continue an aux line? | `aux v=1+\` then `2` | 3 | no diagnostic; computes 3 | matches |
| `cont-ode` | Does a backslash continue an ODE line (not line 1)? | `x'=0+\` then `0` | 13 | no diagnostic; computes 13 | matches |
| `cont-par` | Does a backslash continue a par line? | `par a=1,\` then `b=2` | va=1 vb=2 | no diagnostic; computes va=1 vb=2 | matches |
| `decl-init-expr` | What does x(0)=1+2 declare? | `x(0)=1+2` | 11 | ODE0023 error, value must be a number literal | PhaseCont choice |
| `decl-init-t` | What does x(0)=t+2 declare? | `x(0)=t+2` | 10 | ODE0023 error, value must be a number literal | PhaseCont choice |
| `decl-number-expr` | What does number z=1+1 declare? | `number z=1+1` | 1 | ODE0023 error, value must be a number literal | PhaseCont choice |
| `decl-par-expr` | What does par a=1+1 declare? | `par a=1+1` | 1 | ODE0023 error, value must be a number literal | PhaseCont choice |
| `derived-func-state` | What does a derived parameter get from a function that reads a state? | `!d=g(1)`, `g(u)=u*a+x` | 5 (the initial value of x) | ODE0053 error | PhaseCont choice |
| `derived-later` | Can an aux read a derived parameter defined on a later line? | `aux v=d`, then `!d=a+1` | 4 | no diagnostic; computes 4 | matches |
| `derived-state` | What does a derived parameter read from a state? | `!d=a*x` | 6 (the initial value of x) | ODE0053 error | PhaseCont choice |
| `div-by-zero` | What is 1/0? | `1/0` | 4.4843049e14 | no diagnostic; IEEE division gives inf (`language.md`) | PhaseCont choice |
| `div-zero-zero` | What is 0/0? | `0/0` | 0 | no diagnostic; IEEE division gives NaN (`language.md`) | PhaseCont choice |
| `dy-block` | Does `dy[j]/dt` inside a `%[1..2]` block declare the derivatives of y1 and y2 (XPPAUT 8.2.2)? | `%[1..2]`, `dy[j]/dt=-y[j]`, `y[j](0)=1` | s@2=1 | ODE0019 error for the `dy[j]/dt` line, then ODE0044 and ODE0039 for `y1` and `y2` | PhaseCont choice |
| `dy-range` | Does the range line `dy[1..2]/dt` declare the derivatives of y1 and y2 (XPPAUT 8.2.2)? | `dy[1..2]/dt=-y[j]`, `y1(0)=1`, `y2(0)=1` | s@2=1 | ODE0019 error for the `dy[1..2]/dt` line, then ODE0044 and ODE0039 for `y1` and `y2` | PhaseCont choice |
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
| `aux-fixed-after` | Can an aux column share its name with an earlier fixed quantity (XPPAUT 8.2.2)? | `f=2*x+1`, `aux f=7*x`, `aux g=f` | f=7 g=3 (the aux column keeps its formula, readers get the fixed value) | no diagnostic; the IR gives the same two values | matches |
| `aux-fixed-before` | Can an aux column share its name with a later fixed quantity (XPPAUT 8.2.2)? | `aux f=7*x`, `f=2*x+1`, `aux g=f` | f=7 g=3 | no diagnostic; same values | matches |
| `aux-fixed-self` | Does `aux f=f+1` after a fixed quantity f read the fixed quantity (XPPAUT 8.2.2)? | `f=2*x+1`, `aux f=f+1` | f=4 (at x=1) | no diagnostic; computes 4 | matches |
| `aux-fixed-self-before` | Does `aux f=f+1` above a fixed quantity f read the fixed quantity (XPPAUT 8.2.2)? | `aux f=f+1`, `f=2*x+1` | f=4 | no diagnostic; computes 4 | matches |
| `aux-param-after` | Can an aux column share its name with an earlier parameter (XPPAUT 8.2.2)? | `x'=-x*a`, `par a=2`, `aux a=7*x`, `aux g=a` | a=7 g=2 (prints "duplicate name" and loads; the aux column keeps its formula, readers get the parameter) | no diagnostic; the IR gives the same values (`pc-ffi` test) | matches |
| `aux-param-before` | Can an aux column share its name with a later parameter (XPPAUT 8.2.2)? | `aux a=7*x`, `aux g=a`, `par a=2` | a=7 g=2 | no diagnostic; same values | matches |
| `aux-param-self` | Does `aux a=7*a` read the parameter a (XPPAUT 8.2.2)? | `par a=2`, `aux a=7*a` | a=14 | no diagnostic; computes 14 | matches |
| `aux-state` | Can an aux column share its name with a state (XPPAUT 8.2.2)? | `x'=-x`, `aux x=7*x`, `aux g=x` | x=7 g=1 (prints "duplicate name" and loads; the output file has two columns named x, and a reader gets the state) | ODE0041 error | PhaseCont choice (a table would need two columns with one name; owner question) |
| `aux-aux` | Can two aux quantities share a name (XPPAUT 8.2.2)? | `aux q=2*x`, `aux q=3*x`, `aux z=9` | q=2 z=9 (both q columns are written, z is the fifth column) | ODE0041 error | PhaseCont choice (same reason; owner question) |
| `aux-read-aux` | Can a formula read an aux quantity by name (XPPAUT 8.2.2)? | `aux q=2*x`, `aux g=q` | error ("illegal expression", no output) | ODE0051 error | matches |
| `fixed-forward-func` | What does a fixed quantity get from a function that reads a later fixed quantity? | `b=g(1)`, `c=5`, `g(u)=u+c` | 1 (c is still 0) | ODE0042 error | PhaseCont choice |
| `fixed-forward` | What does a fixed quantity read from a later line? | `b=c+1` above `c=2` | 1 | ODE0042 error, fixed quantity defined on a later line | PhaseCont choice |
| `fn-atan2` | Which argument order does atan2 use? | `atan2(1,0)`, `atan2(-1,-1)` | v1=1.5707964 v2=-2.3561945 | no diagnostic; computes v1=1.5707964 v2=-2.3561945 | matches |
| `fn-basic` | What do the elementary builtins give? | `tan(1)`, `asin(1)`, `sinh(1)`, `erf(1)`, `min(3,2)`, `sign(-3)` and 12 more | 18 values, see the file | no diagnostic | matches |
| `fn-bessel` | Is the order of besselj and bessely truncated toward zero? | `besselj(1.5,1)`, `besselj(-0.5,1)`, `bessely(1.5,1)` | equal to the integer order | no diagnostic | matches |
| `fn-ceil` | Is ceil accepted? | `ceil(2.5)` | error (no output) | ODE0013 warning, XPPAUT 8.0 rejects this | PhaseCont choice |
| `fn-delay` | Does delay(x,1) evaluate at t=0? | `10*delay(x,1)` | 30 | ODE0054 warning, `delay` read by `phasecont.continuation` (equilibria, Hopf curves and periodic orbits from a Hopf point) and by `pc-cli dde-eq`, `dde-branch`, `dde-hopf`, `dde-lyap`, `dde-zh`, `dde-hh` and `dde-cycle` | PhaseCont choice |
| `fn-fact` | Is fact a builtin (fact(3))? | `fact(3)` | error (no output) | ODE0039 error, undeclared name | matches |
| `fn-flr` | Is flr a floor? | `flr(2)`, `flr(-2)`, `flr(-2.5)` | v1=2 v2=-2 v3=-3 | no diagnostic; computes v1=2 v2=-2 v3=-3 | matches |
| `fn-gamma` | Is gamma a builtin (gamma(3))? | `gamma(3)` | error (no output) | ODE0039 error, undeclared name | matches |
| `fn-int` | Is int a builtin (int(2.5))? | `int(2.5)` | error (no output) | ODE0039 error, undeclared name | matches |
| `fn-lgamma` | What is lgamma(5)? | `lgamma(5)` | 3.1780539 | no diagnostic; computes 3.1780539 | matches |
| `fn-ln` | Is ln natural (ln(100))? | `ln(100)` | 4.6051702 | no diagnostic; computes 4.6051702 | matches |
| `fn-log10` | What is log10(100)? | `log10(100)` | 2 | no diagnostic; computes 2 | matches |
| `fn-log` | Is log natural (log(100))? | `log(100)` | 4.6051702 | no diagnostic; computes 4.6051702 | matches |
| `fn-mod` | What does mod give for negative, fractional and zero arguments? | `mod(-7,3)`, `mod(7,-3)`, `mod(-7,-3)`, `mod(7.5,2)`, `mod(-7.5,2)`, `mod(5,2.5)`, `mod(1,0)` | v1=2 v2=1 v3=-4 v4=1.5 v5=0.5 v6=0 v7=nan | no diagnostic; computes v1=2 v2=1 v3=-4 v4=1.5 v5=0.5 v6=0 v7=nan | matches |
| `fn-mod-more` | Does mod follow fmod plus y when the remainder is negative, for more signs and sizes? | `mod(-0.1,1)`, `mod(2.5,-1)`, `mod(-2.5,-1)`, `mod(0,-3)`, `mod(-3,3)`, `mod(123456.75,7)`, `mod(-123456.75,7)`, `mod(0.75,-1)` | v1=0.89999998 v2=0.5 v3=-1.5 v4=0 v5=-0 v6=4.75 v7=2.25 v8=0.75 | no diagnostic; computes the same values with `r = fmod(x, y)`, plus `y` when `r < 0` | matches |
| `fn-random` | What do ran(0) and poisson(0) give? | `ran(0)`, `poisson(0)` | 0 and 0 | no diagnostic | matches |
| `fn-shift` | Does shift(x,0) evaluate? | `10*shift(x,0)` | 30 | ODE0055 warning, `shift` unsupported | PhaseCont choice |
| `global-arre` | Is `arre=1`, a shortened `arret`, a global target XPPAUT refuses (XPPAUT 8.2.2)? | `global 1 x-0.55 {arre=1}` | error | ODE0058 error | matches |
| `global-arret` | Does `arret=1` in a global action list stop the run (XPPAUT 8.2.2)? | `global 1 x-0.55 {arret=1}` | rows=6 (the run ends at t=0.5 of 2) | no diagnostic; the IR refuses the model as simulation only | matches |
| `global-arret-aux` | Does an aux named `arret` take the place of the stop action (XPPAUT 8.2.2)? | `aux arret=x`, `global 1 x-0.55 {arret=1}` | rows=21 (no stop) | ODE0058 error, an aux cannot be assigned | PhaseCont choice |
| `global-arret-derived` | Does a derived parameter named `arret` leave the action as the stop (XPPAUT 8.2.2)? | `!arret=a+1`, `global 1 x-0.55 {arret=1}` | rows=6 (stop) | ODE0058 error, a derived parameter cannot be assigned | PhaseCont choice |
| `global-arret-fixed` | Is a fixed quantity named `arret` a target, so the run does not stop (XPPAUT 8.2.2)? | `arret=x+1`, `global 1 x-0.55 {arret=1}` | rows=6 (stop) | ODE0058 error, a fixed quantity cannot be assigned | PhaseCont choice |
| `global-arret-func` | Does a function named `arret` leave the action as the stop (XPPAUT 8.2.2)? | `arret(u)=u+1`, `global 1 x-0.55 {arret=1}` | rows=6 (stop) | ODE0058 error, a function cannot be assigned | PhaseCont choice |
| `global-arret-number` | Does a number named `arret` leave the action as the stop (XPPAUT 8.2.2)? | `number arret=0`, `global 1 x-0.55 {arret=1}` | rows=6 (stop) | ODE0058 error, a number cannot be assigned | PhaseCont choice |
| `global-arret-par` | Is a declared parameter `arret` an ordinary global target, with no stop (XPPAUT 8.2.2)? | `par arret=0`, `global 1 x-0.55 {arret=1}` | rows=21 | no diagnostic | matches |
| `global-arret-read` | Can a formula read `arret` when only a global action assigns it (XPPAUT 8.2.2)? | `aux a=arret` | error | ODE0039 error | matches |
| `global-arret-state` | Is a declared state `arret` an ordinary global target, with no stop (XPPAUT 8.2.2)? | `arret'=0`, `global 1 x-0.55 {arret=1}` | rows=21 | no diagnostic | matches |
| `global-braces` | Does global 1 {t-0.5} {x=5} work like the form without braces? | `global 1 {t-0.5} {x=5}`, `aux v=x` | v@2=5 | no diagnostic | matches |
| `global-empty` | Is an empty global list {} accepted? | `global 1 t-0.5 {}` | error ("Bad global") | ODE0020 error | matches |
| `global-no-interp` | Does the action name `no_interp` load and assign nothing (XPPAUT 8.2.2)? | `global 1 x-0.55 {no_interp=1}` | v@7=0.6 rows=11 | ODE0058 error, `no_interp` is not declared | PhaseCont choice |
| `global-out-put` | Does `out_put=1` in a global action list add an output row at the crossing (XPPAUT 8.2.2)? | `global 1 x-0.55 {out_put=1}`, `nout=5` | t@3=0.55, rows=6 | no diagnostic; the IR refuses the model as simulation only | matches |
| `global-par-target` | Can a global event assign a parameter? | `par a=1`, `global 1 t-0.9 {a=5}` | v@1=1 v@2=5 | no diagnostic | matches |
| `global-plain` | Does global 1 t-0.5 {x=5} set x at the crossing? | `global 1 t-0.5 {x=5}`, `aux v=x` | v@2=5 | no diagnostic | matches |
| `global-target-aux` | Is an aux a valid `global` target (XPPAUT 8.2.2)? | `aux w=x`, `global 1 x-0.55 {w=1}` | rows=21 (loads, no visible effect) | ODE0058 error, an aux cannot be assigned | PhaseCont choice |
| `global-target-fixed` | Is a fixed quantity a valid `global` target (XPPAUT 8.2.2)? | `f=x+1`, `global 1 x-0.55 {f=1}` | error (no output) | ODE0058 error, a fixed quantity cannot be assigned | matches |
| `global-target-number` | Is a number a valid `global` target (XPPAUT 8.2.2)? | `number nn=0`, `global 1 x-0.55 {nn=1}` | error (no output) | ODE0058 error, a number cannot be assigned | matches |
| `global-two` | Does global 1 t-0.5 {x=5;y=2} set both states? | `global 1 t-0.5 {x=5;y=2}` | v@2=25 | no diagnostic | matches |
| `heav-negzero` | What is heav(-0)? | `heav(-0)` | 1 | no diagnostic; computes 1 | matches |
| `heav-zero` | What is heav(0)? | `heav(0)` | 1 | no diagnostic; computes 1 | matches |
| `hom-bcs` | Does XPPAUT 8.0 accept a hom_bcs line? | `hom_bcs 1` | 5 | ODE0035 warning, `hom_bcs` unsupported | PhaseCont choice |
| `hom-bcs-func` | Is `hom_bcs` a one-argument function that returns 0 (XPPAUT 8.2.2)? | `aux h=hom_bcs(3)` | h=0 | ODE0039 error, function `hom_bcs` is not declared | PhaseCont choice |
| `hom-bcs-name` | Can a parameter be named `hom_bcs` (XPPAUT 8.2.2)? | `par hom_bcs=9` | error (no output, duplicate name) | no diagnostic | PhaseCont choice |
| `idx-global-par` | Is a `global` target `p[1]` the parameter p1 (XPPAUT 8.2.2)? | `par p1=0`, `global 1 x1-0.55 {p[1]=1}` | v@7=1 | ODE0015 error at the index | PhaseCont choice |
| `idx-global-state` | Is a `global` target `x[1]` the state x1 (XPPAUT 8.2.2)? | `x[1]'=1`, `global 1 x1-0.55 {x[1]=0}` | v@7=0.05 | ODE0015 error at the index | PhaseCont choice |
| `idx-map` | Is a map target `x[1](t+1)` the map variable x1 (XPPAUT 8.2.2)? | `x[1](t+1)=x1+1`, `init x1=0` | v@4=3 | ODE0015 error at the index, then ODE0044 for the `init` | PhaseCont choice |
| `idx-map-range` | Does a map range line `x[1..2](t+1)` declare x1 and x2 (XPPAUT 8.2.2)? | `x[1..2](t+1)=x[j]+[j]`, `init x1=0,x2=0` | v@3=6 | no diagnostic; a map with two states (SPEC 8u) | matches |
| `idx-markov` | Is a `markov` name `z[1]` the variable z1 (XPPAUT 8.2.2)? | `markov z[1] 2` | v@2=1 v@3=0 | ODE0015 error at the index | PhaseCont choice |
| `idx-markov-range` | Does a `markov` range `z[1..2]` declare z1 and z2 with one set of rows (XPPAUT 8.2.2)? | `markov z[1..2] 2` | v@2=1 v@3=0 (z2) | ODE0070 error, no range on `markov` | PhaseCont choice |
| `idx-option-total` | Is an option name `total[1]` read as total1, which sets `total` (XPPAUT 8.2.2)? | `@ total[1]=2,dt=1` | rows=3 | ODE0020 error, expected `=` after the option | PhaseCont choice |
| `idx-option-unknown` | Is an unknown option name `xx[1]` read as xx1 (XPPAUT 8.2.2)? | `@ xx[1]=2` | rows=3 (prints a message, loads) | ODE0020 error, expected `=` after the option | PhaseCont choice |
| `idx-set` | Does a `set` line whose name carries an index load (XPPAUT 8.2.2)? | `set s[1] {a=7}` | 5 | ODE0015 error at the index | PhaseCont choice |
| `idx-wiener` | Is a `wiener` name `w[1]` the variable w1 (XPPAUT 8.2.2)? | `wiener w[1]`, `aux v=0*w1+5` | v=5 | ODE0015 error at the index | PhaseCont choice |
| `idx-wiener-range` | Does a `wiener` range `w[1..2]` declare w1 and w2 (XPPAUT 8.2.2)? | `wiener w[1..2]`, `aux v=0*w2+5` | v=5 | no diagnostic; two wiener names (simulation only) | matches |
| `if-lazy` | Does if(1)then(1)else(1/0) evaluate both branches? | `if(1)then(1)else(1/0)` | 1 | no diagnostic; both branches are evaluated and one is selected, so the result is 1 (`language.md`) | matches |
| `implicit-div` | Is implicit multiplication after a division accepted (a/2v)? | `a/2v0`, a=8 | error (no output) | ODE0013 warning, XPPAUT 8.0 rejects this | PhaseCont choice |
| `implicit-num-name` | Is implicit multiplication of a number and a name accepted (2x)? | `2x` | error (no output) | ODE0013 warning, XPPAUT 8.0 rejects this | PhaseCont choice |
| `implicit-num-paren` | Is implicit multiplication before a parenthesis accepted (2(x))? | `2(x)` | error (no output) | ODE0013 warning, XPPAUT 8.0 rejects this | PhaseCont choice |
| `include-comment` | Does a trailing comment on an #include line stop the file from loading? | `#include inc/a.ode # c` | error (the name is undefined) | ODE0022 error | matches |
| `include-cycle` | What does XPPAUT do with a file that includes itself? | `#include loop.inc` in `loop.inc` | runs until it crashes | ODE0061 error | PhaseCont choice |
| `include-done` | Does done in an included file end only that file? | `done` in `inc/d.ode` | 10 | no diagnostic; computes 10 | matches |
| `include-missing` | Is an #include of a missing file ignored? | `#include inc/none.ode` | 1 | ODE0060 error | PhaseCont choice |
| `include-ok` | Does #include inc/a.ode load the file? | `#include inc/a.ode` | 3 | no diagnostic; computes 3 | matches |
| `include-path` | Is an include inside inc/sub/a.ode taken relative to that file? | `#include b.ode` in `inc/sub/a.ode` | error (taken relative to the working directory) | no diagnostic; relative to the including file | PhaseCont choice |
| `include-quotes` | Does a quoted file name in #include load the file? | `#include "inc/a.ode"` | error (the name is undefined) | ODE0059 error | matches |
| `index-fixed-bare` | What does a bare `[3]` give (XPPAUT 8.2.2)? | `aux v=[3]` | v=3 | ODE0015, a constant index can only follow a name | PhaseCont choice |
| `index-fixed-decl` | Does a constant index on a declared name name the numbered symbol (XPPAUT 8.2.2)? | `y[1]'=1`, `dz[2]/dt=2`, `init q[2]=4`, `par a[2]=3`, `number n[2]=6`, `f[3]=7`, `!d[2]=n2+1`, `g[2](u)=u*4`, `table h[2] % 3 0 2 t^2`, `aux w[2]=y1+3`, a `%[1..2]` block with `v[j]=n[2]*[j]` | s=14 w=3 e=40 m=18 | no diagnostic; same values | matches |
| `index-fixed-initval` | Does the init value `x[1](0)=3` set `x1` (XPPAUT 8.2.2)? | `x1'=1`, `x[1](0)=3`, `aux v=x1` | v=0 (not set) | ODE0015 error | PhaseCont choice |
| `index-fixed-range-line` | Does a constant index next to a range line read as the numbered name (XPPAUT 8.2.2)? | `flux[1..3]=[j]*x`, `y'=-flux[3]`, `aux r=flux[2]+flux3` | r=5, yy=-3 at t=1 | no diagnostic; same values | matches |
| `index-fixed-read` | Is `name[n]` read as `name<n>` in a formula (XPPAUT 8.2.2)? | `f3=7`, `par a2=3`, `k3=100`, `g2(u)=u*2`, `aux r=f[3]+a[2]+g[2](4)+k[03]` | r=118 | no diagnostic; computes 118 | matches |
| `kw-export-e` | Is `e`, one letter, still an `export` line (XPPAUT 8.2.2)? | `e {zz} {yy}` | 5 (ignored) | ODE0019 error, not a statement | PhaseCont choice |
| `kw-export-ex` | Is `ex`, the first two letters, still an `export` line (XPPAUT 8.2.2)? | `ex {zz} {yy}` | error (no output) | ODE0019 error, not a statement | PhaseCont choice |
| `kw-export-full` | Does an `export` line with a name that does not exist stop the load (XPPAUT 8.2.2)? | `export {zz} {yy}` | error (no output) | ODE0032 warning, kept as raw text | PhaseCont choice |
| `kw-fixed-letters` | Are p=, a=, w=, i=, d=, g=, b=, m= fixed quantities? | `p = 5`, `a = 6`, `w = 7`, `i = 8`, `d = 9`, `g = 10`, `b = 11`, `m = 12` | vp=5 va=6 vw=7 vi=8 vd=9 vg=10 vb=11 vm=12 | no diagnostic; computes vp=5 va=6 vw=7 vi=8 vd=9 vg=10 vb=11 vm=12 | matches |
| `kw-hom-bcs-prefix` | Is `hom`, a prefix of `hom_bcs`, an ignored line (XPPAUT 8.2.2)? | `hom 1` | 5 (ignored) | ODE0019 error, not a statement | PhaseCont choice |
| `kw-options-full` | Does an `options` line that names a missing file load (XPPAUT 8.2.2)? | `options none.opt` | 5 (prints `Loading new options file`) | ODE0034 warning, kept as raw text | PhaseCont choice |
| `kw-options-o` | Is `o`, one letter, still an `options` line (XPPAUT 8.2.2)? | `o none.opt` | 5 (prints `Loading new options file`) | ODE0019 error, not a statement | PhaseCont choice |
| `kw-options-ox` | Is `ox`, a word that only starts with o, still an `options` line (XPPAUT 8.2.2)? | `ox none.opt` | 5 (prints `Loading new options file`) | ODE0019 error, not a statement | PhaseCont choice |
| `kw-p-nospace` | Is p=5 (no blanks) a fixed quantity? | `p=5` | 5 | no diagnostic; computes 5 | matches |
| `kw-p-par` | Does p a=5 declare a parameter? | `p a=5` | 5 | no diagnostic; computes 5 | matches |
| `kw-set-malformed` | Does a `set` line with an unclosed brace stop the load (XPPAUT 8.2.2)? | `set s {zz` | 5 (loads; prints `added s doing $ zz`) | ODE0020 error, expected `=` | PhaseCont choice |
| `kw-set-se` | Does `se s {a=7}`, a prefix of `set`, change the parameter a (XPPAUT 8.2.2)? | `se s {a=7}` | 5 (no change in a headless run; prints `added s doing $ a=7`) | ODE0019 error, not a statement | PhaseCont choice |
| `kw-special-conv` | Does `special f=conv(zz)` load (XPPAUT 8.2.2)? | `special f=conv(zz)` | crashes (exit status 139) | ODE0031 warning, kept as raw text | PhaseCont choice |
| `kw-special-full` | Does a line that starts with the full word `special` reach the special-function parser (XPPAUT 8.2.2)? | `special f=zz` | error (Illegal special function) | ODE0031 warning, kept as raw text | PhaseCont choice |
| `kw-special-specia` | Is `specia`, the word without its last letter, still a `special` line (XPPAUT 8.2.2)? | `specia f=zz` | error (Illegal special function) | ODE0019 error, not a statement | PhaseCont choice |
| `kw-special-speci` | Is `speci`, five letters, still a `special` line (XPPAUT 8.2.2)? | `speci f=zz` | 5 (ignored) | ODE0019 error, not a statement | PhaseCont choice |
| `kw-special-spxxxa` | Does `spxxxa` reach the special-function parser, so that only SP and a sixth letter A count (XPPAUT 8.2.2)? | `spxxxa f=zz` | error (Illegal special function) | ODE0019 error, not a statement | PhaseCont choice |
| `kw-special-spxxxx` | Does `spxxxx`, with no A in sixth place, stay out of the special-function parser (XPPAUT 8.2.2)? | `spxxxx f=zz` | 5 (ignored) | ODE0019 error, not a statement | PhaseCont choice |
| `kw-unknown-word` | Is a line with an unknown first word and a number ignored (XPPAUT 8.2.2)? | `zzz 1` | 5 (ignored) | ODE0019 error, not a statement | PhaseCont choice |
| `kw-volt-full` | Does `volt w=x+1` add a state column w (XPPAUT 8.2.2)? | `volt w=x+1` | v=5, two states | ODE0028 warning, kept as raw text | PhaseCont choice |
| `kw-volt-v` | Does a one-letter `v w=x+1` add a state column w (XPPAUT 8.2.2)? | `v w=x+1` | v=5, two states | ODE0019 error, not a statement | PhaseCont choice |
| `kw-volt-vector` | Does `vector w=x+1` add a state column w like `volt` (XPPAUT 8.2.2)? | `vector w=x+1` | crashes (exit status 139) | ODE0019 error, not a statement | PhaseCont choice |
| `kw-volt-vecxxr` | Does `vecxxr`, a word with `ve` first and `r` in sixth place, reach the vectorizer like `vector` (XPPAUT 8.2.2)? | `vecxxr w=x+1` | crashes (exit status 139) | ODE0019 error, not a statement | PhaseCont choice |
| `kw-volt-vecxxx` | Does `vecxxx`, a `ve` word with no `r` in sixth place, add a state column w like `volt` (XPPAUT 8.2.2)? | `vecxxx w=x+1` | v=5, two states | ODE0019 error, not a statement | PhaseCont choice |
| `kw-volt-vol` | Does `vol w=x+1` add a state column w (XPPAUT 8.2.2)? | `vol w=x+1` | v=5, two states | ODE0019 error, not a statement | PhaseCont choice |
| `kw-volt-vx` | Does `vx w=x+1`, a word that only starts with v, add a state column w (XPPAUT 8.2.2)? | `vx w=x+1` | v=5, two states | ODE0019 error, not a statement | PhaseCont choice |
| `list-init-blank` | Do blanks separate init entries? | `init x=1 y=2` | 21 | no diagnostic; computes 21 | matches |
| `list-number-blank` | Do blanks separate number entries? | `number a=1 b=2` | 21 | no diagnostic; computes 21 | matches |
| `log-neg` | What is log(-1)? | `log(-1)` | nan | no diagnostic; computes nan | matches |
| `map-dt` | Does dt change the step of a map? | `x(t+1)=x+t`, `@ total=4,dt=0.5` | t@2=1 v@5=6 (step 1) | no diagnostic | PhaseCont choice |
| `map-meth-euler` | Does meth=euler change how a map file runs? | `x(t+1)=r*x*(1-x)`, `@ meth=euler,total=2` | v@2=0.22499999 (iterated) | no diagnostic | matches |
| `map-t-index` | Is t inside a map the iterate index, and is it the time column? | `x(t+1)=x+t`, `@ total=4` | t@5=4 v@3=1 v@5=6 | no diagnostic | matches |
| `map-t0` | Does t0 shift the t that a map reads? | `x(t+1)=x+t`, `@ total=4,t0=5` | t@1=5 v@2=5 v@3=11 | no diagnostic | matches |
| `map-t0-frac` | Does a map accept a t0 that is not an integer? | `x(t+1)=x+t`, `@ total=4,t0=2.5` | t@2=3.5 v@2=2.5 v@3=6 | no diagnostic | PhaseCont choice |
| `markov-extra` | Is extra text after n ignored? | `markov z 2 1` | v@2=1 v@3=0 | ODE0020 error with a fix-it | PhaseCont choice |
| `markov-rows` | Do the rows {0} {1} / {1} {0} flip a two-state markov variable every step? | `markov z 2` | v@2=1 v@3=0 | no diagnostic | matches |
| `meth-d` | Does meth=d iterate an ODE file as a map? | `x'=r*x*(1-x)`, `@ meth=d,total=2` | v@2=0.22499999 (iterated) | ODE0074 warning; the file is a map | matches |
| `meth-discrete` | Does meth=discrete iterate an ODE file as a map? | `x'=r*x*(1-x)`, `@ meth=discrete,total=2` | v@2=0.22499999 (iterated) | ODE0074 warning; the file is a map | matches |
| `meth-last` | Does a later meth entry override an earlier one? | `@ meth=discrete,total=2`, then `@ meth=euler` | v@2=0.11125 (Euler) | no diagnostic | matches |
| `meth-method` | Does the key method set meth? | `@ method=discrete,total=2` | v@2=0.22499999 (iterated) | ODE0074 warning; the file is a map | matches |
| `meth-ode45` | What does a meth value that names no XPPAUT method do? | `@ meth=ode45,total=2` | v@2=0.22499999 (iterated) | ODE0074 warning; the file stays an ODE | PhaseCont choice |
| `meth-rungekutta` | Does meth=rungekutta integrate? | `@ meth=rungekutta,total=2` | v@2=0.11182589 (integrated) | no diagnostic | matches |
| `meth-spaced` | Is meth = discrete with blanks around = ignored? | `@ meth = discrete,total=2` | v@2=0.11182589 (integrated) | ODE0025 warning; the file stays an ODE | matches |
| `meth-upper` | Is the meth key and value case-insensitive? | `@ METH=Discrete,total=2` | v@2=0.22499999 (iterated) | ODE0074 warning; the file is a map | matches |
| `mixed-ode-map` | Is a file with an ODE and a map accepted? | `x'=0` with `y(t+1)=y/2` | 17 | ODE0052 error, ODEs and maps mixed | PhaseCont choice |
| `name-10` | Is a 10-character name accepted? | `par abcdefghij=3` | 9 | no diagnostic; computes 9 | matches |
| `name-11-collide-used` | Can two names that agree in 10 characters be used in a formula? | `aux v=abcdefghijk+abcdefghijl` | error (no output) | ODE0013 warning, XPPAUT 8.0 rejects this | PhaseCont choice |
| `name-11-collide` | Do two declared names that agree in 10 characters collide when neither is used? | `par abcdefghijk=1`, `par abcdefghijl=2`, neither used | 17 | ODE0013 warning on each long name, no collision note | matches |
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
| `number-n` | Is the one-letter word `n` read as `number` (XPPAUT 8.2.2)? | `n a=2` | 2 | no diagnostic; computes 2 | matches |
| `number-num` | Is `num` read as `number` (XPPAUT 8.2.2)? | `num a=2` | 2 | no diagnostic; computes 2 | matches |
| `number-numx` | Is a first word that only starts with `n` read as `number` (XPPAUT 8.2.2)? | `numx a=2` | 2 | no diagnostic; computes 2 | matches |
| `op-bang` | Is a bare ! accepted (!1)? | `!1` | error (no output) | ODE0007 error, not an operator | matches |
| `op-ne` | Is != accepted (1!=2)? | `1!=2` | error (no output) | ODE0007 error, not an operator | matches |
| `op-single-eq` | Is a single = inside a formula accepted (1=2)? | `1=2` | error (no output) | ODE0008 error, write `==` | matches |
| `opt-njmp-last` | njmp=4, then nout=2 on a later @ line: which wins (XPPAUT 8.2.2)? | `@ njmp=4` then `@ nout=2` | rows=5 (nout=2 wins) | no diagnostic; the model info lists njmp=2 | matches |
| `opt-njmp-same-line` | njmp=4, then nout=2 in the same @ list: which wins (XPPAUT 8.2.2)? | `@ ...,njmp=4,nout=2` | rows=5 (nout=2 wins) | no diagnostic; the model info lists njmp=2 | matches |
| `opt-nout` | Does nout=2 set the output step like njmp=2 (XPPAUT 8.2.2)? | `@ ...,nout=2` | t@2=1 | no diagnostic; the model info lists njmp=2 | matches |
| `opt-nout-last` | nout=2, then njmp=4 on a later @ line: which wins (XPPAUT 8.2.2)? | `@ ...,nout=2` then `@ njmp=4` | rows=3 (njmp=4 wins) | no diagnostic; the model info lists njmp=4 | matches |
| `opt-nout-prefix` | Does a key that starts with nout, such as noutx, set njmp (XPPAUT 8.2.2)? | `@ ...,noutx=2` | t@2=1 | no diagnostic; the model info lists njmp=2 | matches |
| `opt-nout-same-line` | nout=2, then njmp=4 in the same @ list: which wins (XPPAUT 8.2.2)? | `@ ...,nout=2,njmp=4` | rows=3 (njmp=4 wins) | no diagnostic; the model info lists njmp=4 | matches |
| `opt-nout-short` | Does nou, a key shorter than nout, set the output step (XPPAUT 8.2.2)? | `@ ...,nou=2` | rows=9 (nothing set) | ODE0024 warning, unknown option | PhaseCont choice |
| `opt-prefix-dt` | Does a key that starts with dt, such as dtx, set dt? | `@ dtx=0.5,total=1` | t@2=0.5 | no diagnostic | matches |
| `opt-prefix-short` | Does tot, a key shorter than total, set total? | `@ tot=2,dt=1` | t@21=20 (total stays 20) | ODE0024 warning, unknown option | matches |
| `opt-prefix-t0` | Does a key that starts with t0, such as t0x, set t0? | `@ t0x=5,dt=0.5,total=1` | t@1=5 | no diagnostic | matches |
| `opt-prefix-total` | Does a key that starts with total, such as totalx, set total? | `@ totalx=2,dt=1` | t@3=2 | no diagnostic | matches |
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
| `solv-s` | Is the one-letter word `s` read as `solv` (XPPAUT 8.2.2)? | `s y=-1` | error | ODE0057 warning, ODE0030 warning, ODE0019 error | matches |
| `solv-so` | Is the two-letter word `so` read as `solv` (XPPAUT 8.2.2)? | `so y=-1` | -1 | ODE0057 and ODE0030 warnings (algebraic constraint kept as raw text) | matches |
| `solv-solve` | Is the spelled-out word `solve` read as `solv` (XPPAUT 8.2.2)? | `solve y=-1` | -1 | ODE0057 and ODE0030 warnings (algebraic constraint kept as raw text) | matches |
| `sqrt-neg` | What is sqrt(-1)? | `sqrt(-1)` | nan | no diagnostic; computes nan | matches |
| `stmt-abbrev` | Does only the first letter of par count (pzz a=2)? | `pzz a=2` | 2 | no diagnostic; computes 2 | matches |
| `table-inline` | Does table h % 11 0 1 2*t define h(0.5)=1? | `table h % 11 0 1 2*t` | 1 | no diagnostic | matches |
| `table-range` | Does an inline table interpolate linearly and extrapolate from its end segments? | `table h % 3 0 2 t^2`, then `h(0.5)`, `h(1.5)`, `h(0)`, `h(2)`, `h(-1)`, `h(3)`, `h(10)` | 0.5 2.5 0 4 -1 7 28 | no diagnostic | matches |
| `truth-and-half` | Is 0.5 true for & (0.5&1)? | `0.5&1` | 1 | no diagnostic; computes 1 | matches |
| `truth-and-neghalf` | Is -0.5 true for & (-0.5&1)? | `-0.5&1` | 1 | no diagnostic; computes 1 | matches |
| `truth-if-neg` | Is -1 true for if (if(-1)then(7)else(8))? | `if(-1)then(7)else(8)` | 7 | no diagnostic; computes 7 | matches |

## Notes

- Prefix probes. XPPAUT accepts a prefix `-` only at the start of an expression, after `(` or after `,`. It accepts `not` after an operator but computes a wrong value there (`0&not 0` gives 1). PhaseCont reports an error with a fix-it for both,
- `name-11`, `name-11-collide`, `name-11-collide-used`. XPPAUT 8.0 accepts a declared name longer than 10 characters but cannot read it back in a formula: it stops with `illegal expression: K`, which shows it keeps only the first 10 characters when it parses a formula. Two declared names that agree in their first 10 characters did not collide in any form measured (both unused, both used in a formula, one alone, a fixed-quantity pair, and `abcdefghij` with `abcdefghijk`). Only using such a name in a formula fails. PhaseCont gives an ODE0013 warning for each long name and no note about names that share their first 10 characters, so the row `name-11-collide` matches.
- `name-arg1-outside`. XPPAUT reads `arg1` as 0 outside a function. PhaseCont reports an error (ODE0050).
- `if-lazy`, `div-by-zero`, `div-zero-zero`. Whether XPPAUT evaluates both branches of `if` cannot be seen from values. Division by zero is guarded in XPPAUT (`1/0` gives 4.4843049e14, `0/0` gives 0). P2 ruled: PhaseCont division follows IEEE (`1/0` is inf, `0/0` is NaN), and `if` evaluates both branches and selects one with the test `c != 0`. See `crates/pc-ode/docs/language.md`.
- `meth-*`, `map-meth-euler`, `opt-prefix-*`. XPPAUT 8.0 picks the method from the first character of the `meth` value, ignoring case. On 2026-11-14 one probe per character a to z and 0 to 9 ran: the characters a b c e g m q r s y 2 5 8 integrate, v (Volterra) crashes XPPAUT on a plain ODE, and every other character iterates the file as a map. An empty value runs the default Runge-Kutta. The last `meth` entry wins, also across `@` lines. XPPAUT takes a key that starts with a full option name as that option, so `method` sets `meth` and `totalx` sets `total`, while a shorter key such as `met` or `tot` sets nothing. Runs of the same day showed this for `dt`, `t0`, `trans`, `njmp`, `total`, `bound`, `toler` and `dtmin`, and the longer name wins: `dtminx` acts like `dtmin`, not `dt`. The effect of `dtmax`, `atoler` and `maxstor` does not show in the output file, so they were not measured. PhaseCont follows the `meth` rule with one exception: only a value that starts with d makes the file a map, and any other value outside the integrator letters keeps the file an ODE with an ODE0074 warning (`meth-ode45`). PhaseCont follows the prefix rule for the measured names and gives no ODE0024 for such a key. It matches every other option name exactly, so a key such as `xlox` still gets ODE0024 (PhaseCont choice until a probe shows the rule for that name). The sweep files are not committed; `meth-d`, `meth-ode45` and `meth-rungekutta` keep three of the method runs, and `opt-prefix-dt`, `opt-prefix-t0`, `opt-prefix-short` and `opt-prefix-total` keep four of the option runs.
- `map-t-index`, `map-t0`, `map-dt`, `map-t0-frac`. In a map file XPPAUT 8.0 sets `t` inside the right-hand side to the iterate index, and the time column holds the same value. `t0` shifts it (t0=5 gives t = 5, 6, ... and x = 0, 5, 11, 18, 26 from `x(t+1)=x+t`), `dt` does not change the step, and a fractional `t0` is accepted (t = 2.5, 3.5, ...). In other runs of the same day `total=4.5` acted as 4, `trans=2` dropped the rows before t = 2 and `njmp=2` wrote every second row. PhaseCont iterates a map with `simulate(m, x0, [n0 n1])`, where n0 and n1 must be integers, and sets t = k inside the right-hand side (SPEC 8u). It does not read `dt`, `t0`, `total`, `trans` or `njmp` for a map, so `map-dt` and `map-t0-frac` are PhaseCont choices. The `total`, `trans` and `njmp` runs are not committed.
- `kw-*`, `hom-bcs-*`. XPPAUT 8.2.2 reads some statement keywords from fewer letters than PhaseCont does, and ignores a line whose first word it does not know (`kw-unknown-word`, `kw-hom-bcs-prefix`). Measured on 2026-11-18: `special` needs `sp` and `a` in sixth place (`kw-special-*`), `export` needs `ex` (`kw-export-*`), `volt` takes any word that starts with `v` except one that starts with `ve` and has `r` in sixth place, which XPPAUT reads as `vector` and crashes on (`kw-volt-*`: `kw-volt-vector` and `kw-volt-vecxxr` crash, `kw-volt-vecxxx` is volt), `options` takes any word that starts with `o` except one that starts with `on` (`kw-options-*`), and `set` takes `se` (`kw-set-se`; a headless run shows no effect of a set line, even a broken one in `kw-set-malformed`). The `options` and `set` rules come from messages on stdout, which `run.sh` does not check for a value probe: `Loading new options file:<name>` for `o`, `ox`, `op`, `opt` and `optionsx` as for `options`, and `added <name> doing $ <body>` for `se`, `sex` and `setx` as for `set`, and no message for `s` or an unknown word. The `# xpp:` value of those probes is the same whether the line is read or ignored. A `hom_bcs` line is not a statement in 8.2.2, but the name is a deprecated builtin that returns 0 for any argument and reserves the name (`hom-bcs-func`, `hom-bcs-name`). PhaseCont needs the full word for all of these and gives ODE0019 for a shorter one, which is a PhaseCont choice. XPPAUT 8.2.2 crashes (exit status 139) on `vector w=x+1` and on `special f=conv(zz)`; `run.sh` passes a signal exit for `# xpp: hang`, so `kw-volt-vector`, `kw-volt-vecxxr` and `kw-special-conv` record the crash that way.
- `idx-*`, `dy-*`. XPPAUT 8.2.2 folds a constant index into the name for a `map`, `markov`, `wiener`, `global` target, option key and `set` name, as it does for a state (`idx-map`, `idx-markov`, `idx-wiener`, `idx-global-par`, `idx-global-state`, `idx-option-total`, `idx-option-unknown`, `idx-set`). It also expands the range `[1..2]` for `map` and `wiener`, and for `markov` with one set of rows for both names (`idx-map-range`, `idx-wiener-range`, `idx-markov-range`). PhaseCont accepts the two ranges for `map` and `wiener` and refuses the rest with ODE0015, ODE0020 or ODE0070. A derivative written `dy[j]/dt` or `dy[1..2]/dt` works in XPPAUT, alone or inside a `%[1..2]` block (`dy-range`, `dy-block`), and is ODE0019 in PhaseCont.
- `global-arret-*`, `global-target-*`, `global-no-interp`. A `global` target is looked up among the initial-value names, then the parameter names, and then the three action names `out_put`, `arret` and `no_interp`. A name that a number, function, fixed quantity or derived parameter holds is not a target, so `arret` then stays the stop action (rows=6 in the four `global-arret-*` probes); an aux of that name is accepted as a harmless target and the run goes on (`global-arret-aux`, `global-target-aux`, rows=21). A fixed quantity or a number that is not named `arret` is an error (`global-target-fixed`, `global-target-number`). `no_interp` loads and assigns nothing (`global-no-interp`). PhaseCont gives ODE0058 for all of them. That matches XPPAUT for the two error cases and is a PhaseCont choice for the rest.
