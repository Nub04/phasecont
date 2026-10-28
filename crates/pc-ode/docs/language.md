# The `.ode` language accepted by `pc-ode`

This file is the rule source for the `pc-ode` parser, checker and writer. It covers the subset of the XPPAUT `.ode` format that PhaseCont v1 reads, and says where each rule comes from.

## Sources

Each rule carries one of these sources:

- **manual**: the XPPAUT documentation (online help pages, the v5.0 and v5.41 PDF manuals, the v8 notes). Only the documentation was read, never XPPAUT source code.
- **measured**: XPPAUT 8.0 run headless (`xppaut file.ode -silent -outfile o.dat`) on our own probe files in `references/xpp-probes/`. Each rule marked measured below names the probe that checks it, or falls under the probes in the README table of that folder (`run.sh` compares XPPAUT with `pc-cli`). XPPAUT writes its output file with float32 precision, so measured values carry about 7 significant digits (`0.15` prints as `0.15000001`).
- **PhaseCont choice**: a rule we picked where XPPAUT gives no usable answer, or where we deliberately differ.

Where XPPAUT 8.0 and the manual disagree, the measured behavior wins unless a rule says otherwise.

## Lines

- A file is a sequence of lines. A backslash at the end of a line joins it with the next line (manual). The lexer treats backslash plus newline, or backslash plus `\r\n`, as a blank, so spans point into the original text.
- A line whose first non-blank character is `#` is a comment, except `#include` (manual). Comments are kept in the AST and written back.
- A trailing `#` comment after a statement is accepted after a `par`, `number` or `@` list, kept on the statement and written back (measured: XPPAUT 8.0 ignores it there). After any other statement it is an error (measured for formulas and `init`, which XPPAUT rejects; PhaseCont choice for the statements not measured).
- A file whose first line starts with a digit is in the old XPPAUT format, which is an error (manual).
- Case is ignored in names and keywords (manual, measured). The AST keeps the spelling as written and uses the lower-case form as the lookup key.
- A stray character is an error. That includes any non-ASCII character outside a comment, and a backslash that is not the last character of a line (PhaseCont choice).
- A byte order mark (U+FEFF) at the very start of a file is ignored. Anywhere else it is a stray character (PhaseCont choice).
- A lone `\r` that is not followed by `\n` ends a line, like `\n` and `\r\n` (PhaseCont choice). A comment stops at the first `\r` or `\n`.

## Statement classification

The first word of a line decides what the line is.

- A first word is a keyword only when a blank or the end of the line follows it and the next non-blank character is not `=` (manual, measured). So `p a=5` declares a parameter, while `p=5`, `p = 5`, `a = 1`, `w = 1`, `i = 1`, `d = 1`, `g = 1`, `b = 1` and `m = 1` all define fixed quantities.
- For the nine declarations below, only the first letter is compared, whatever follows (measured: `pzz a=2` declares a parameter). The implementation must not check that the word is a prefix of the full keyword.

| First letter | Declaration |
|---|---|
| `m` | `markov` |
| `p` | `parameter` |
| `w` | `wiener` |
| `t` | `table` |
| `a` | `aux` |
| `i` | `init` |
| `b` | `bndry` (also written `bdry`) |
| `g` | `global` |
| `d` | `done` |

- These keywords need the full word (manual): `number`, `set`, `options`, `only`, `special`, `solv`, `export`, `volt`, `hom_bcs`.
- Any other line is a formula: an ODE, a map, an initial value, a derived parameter, a fixed quantity or a user function, told apart by the form of its left side.

## Statement forms

| Form | Example | Meaning | Status |
|---|---|---|---|
| ODE | `x'=-x`, `dx/dt=-x` | state equation | full |
| Map | `x(t+1)=r*x*(1-x)` | discrete map | full |
| Initial value | `x(0)=1`, `init x=1,y=2` | initial data | full |
| Parameter | `par a=1,b=2` | parameters | full |
| Number | `number c=3` | constant | full |
| Derived parameter | `!d=a*b` | parameter computed from others | full |
| Fixed quantity | `w=x+y` | named subexpression | full |
| Auxiliary | `aux e=x^2+y^2` | output quantity | full |
| User function | `f(u,v)=u*v` | inlined function, at most 9 arguments | full |
| Inline table | `table h % 101 0 6.283 sin(t)` | tabulated function | full |
| File table | `table h data.tab` | tabulated function | unsupported, warning |
| Options | `@ total=100,dt=.05` | numerical options | full |
| Include | `#include other.ode` | include a file | full |
| Set | `set low {a=1,b=2}` | named parameter set | full |
| Global flag | `global 1 x-1 {x=0}` | event | simulation only |
| Markov | `markov z 2` plus one line per state | Markov chain | simulation only |
| Wiener | `wiener w1,w2` | white noise | simulation only |
| End | `done` | end of file | full |
| Volterra | `volt` forms, `x(t)=...int...` | integral equation | unsupported, warning |
| Boundary value | `bndry`, `bdry` | BVP conditions | unsupported, warning |
| DAE | `0=...`, `solv` | algebraic constraint | unsupported, warning |
| Others | `special`, `export`, `only`, `options file`, `hom_bcs` | | unsupported, warning |

Unsupported statements stay in the AST as raw text, so the writer reproduces them.

### Lists

- `par`, `number` and `init` lists separate entries with commas or blanks (measured: `par a=1 b=2`, `number a=1 b=2` and `init x=1 y=2` each set both). Blanks around `=` are allowed in these lists and in formulas (measured: `par a = 2`, `x(0) = 3`, `x' = 1`, `dx/dt = 1`, `aux v = 2`, `f(u) = u*2` all work). No diagnostic.
- In an `@` list XPPAUT 8.0 accepts blanks around `=` but does not apply the option (measured: `@ total = 1, dt = .5` ran with the default `dt`). PhaseCont reads the option and gives a warning that XPPAUT 8.0 ignores it, with a fix-it that removes the blanks.
- `@` option names are all stored. An unknown name gives a warning (PhaseCont choice; XPPAUT ignores it silently). The AUTO options (`ntst`, `nmax`, `npr`, `dsmin`, `dsmax`, `ds`, `parmin`, `parmax`, `epsl`, `epsu`, `epss` and the rest the manual lists) are kept for P3 to use as defaults.

### Values

- A value in a `par`, `number` or `init` list, and the right side of `x(0)=`, is a signed numeric literal. Inside an array template it may also be a bare index form (`x[j](0)=[j]`). Anything else is an error (measured: XPPAUT reads only a leading number, so `par a=1+1` gives 1 and `x(0)=t+2` gives 0).
- A bare name in a `par` list (`par a`) declares the parameter with value 0 (manual).

### Statement details

These rules fix what the parser does where the manual is silent. Each rule says where it comes from: the manual, a measurement with XPPAUT 8.0 ("measured"), or a PhaseCont choice.

- `@` lines.
  - An `@` entry is `name=value`, split by commas or blanks. The value is kept as text up to the next comma, blank, comment or line end (PhaseCont choice).
  - A blank before or after `=` makes XPPAUT 8.0 ignore the option, so it gives a warning (measured). A backslash-newline continuation is not a blank: `dt=\` with `.5` on the next line sets `dt` to 0.5 and gives no warning (measured).
- `set` lines.
  - The syntax is `set name {a=1,b=2}` (manual), and `set low {a=1,b=2}` is an accepted line (PhaseCont choice: XPPAUT 8.0 loads it, but a set only changes the GUI parameter set, so no headless run output shows it and no probe can check it).
  - Blanks also separate entries, as in `set s {a=2 b=3}`, and an empty list `set s {}` loads (PhaseCont choice, for the same reason as above: not observable headlessly). Blanks around `=` give no diagnostic (PhaseCont choice).
- `global` lines.
  - The form is `global sign cond {name=expr; ...}` (manual), and `global 1 x-0.5 {x=0;y=2}` loads and works (measured: probes `global-plain`, `global-two`).
  - The condition may be in braces: `global 1 {t-0.5} {x=5}` acts like `global 1 t-0.5 {x=5}` (measured: probe `global-braces`, compared with `global-plain`). The writer prints the form without braces.
  - The sign must be the literal `-1`, `0` or `1` (PhaseCont choice).
  - An empty list `{}` is an error: XPPAUT rejects it with "Bad global" (measured: probe `global-empty`).
  - A target may be a state or a parameter (measured: probe `global-par-target`, with `par a=1`, `x'=a` and `global 1 t-0.9 {a=5}`, `a` changes from 1 to 5 after t=0.9, and a state target is reset the same way). Any other target (`t`, `pi`, a builtin, `argK`, an aux quantity, a fixed quantity, a derived parameter, a number, a function, a table, a `wiener` name or an undeclared name) is error ODE0058, "cannot be assigned" (PhaseCont choice). An undeclared target gives warning ODE0057 instead when an unsupported statement may declare it. The right side is checked like any formula.
- `markov` lines.
  - `markov name n` is followed by `n` rows, each with `n` `{expr}` groups (manual).
  - XPPAUT ignores extra text after `n`, as in `markov z 2 1` (measured: probe `markov-extra`; probe `markov-rows` checks the 0/1 rows). PhaseCont reports it with a fix-it that deletes it (PhaseCont choice). Missing rows are an error too (PhaseCont choice).
  - Blank lines and comment lines may sit between the rows. The comments are kept inside the `markov` statement in source order, and the blank lines are not stored (PhaseCont choice). An `#include` line ends the rows.
- `table name % npts lo hi expr`: `table h % 11 0 1 2*t` loads and works (measured: probe `table-inline`). The table holds `expr` at `npts` evenly spaced values of `t` from `lo` to `hi`. A call interpolates linearly between them and, outside `[lo, hi]`, extrapolates the first or last segment: with `table h % 3 0 2 t^2`, `h(1.5)` = 2.5, `h(-1)` = -1 and `h(10)` = 28 (measured: probe `table-range`). The derivative of a table call is the slope of the segment that holds the argument; at an interior node it is the slope of the segment to the right (PhaseCont choice). If `expr` reads a parameter, a state or a fixed quantity, `pc-ir` tabulates it from the parameter defaults and the initial values when the model is lowered, so changing them later leaves the table as it was until the model is lowered again (PhaseCont choice; XPPAUT also tabulates at load). A missing or bad field is one error that names the field, and the rest of the line is skipped (PhaseCont choice).
- User functions.
  - A user function has at least one parameter (PhaseCont choice).
  - Neither its name nor its parameters carry an index (PhaseCont choice).
  - The left side `x(t)=` is a Volterra equation (manual).
- `done`.
  - Text after `done` is not read, so it gives no diagnostic, but a comment right after `done` is an error like any other trailing comment (PhaseCont choice).
  - `done` inside an array block also ends the block, which is reported as unclosed (PhaseCont choice).
- Lines the parser cannot read.
  - A line that matches no statement form is an error, and parsing goes on with the next line. A statement whose form is known but whose parts are wrong is a separate error (PhaseCont choice).
  - Lexer diagnostics inside text kept raw (`@` and `set` values, unsupported statements, active comments, everything after `done`) are dropped (PhaseCont choice).
  - An unknown character gets only the lexer's diagnostic. The parser skips it without a second report (PhaseCont choice).
- Active comments: a line that starts with `"` is an XPPAUT active comment (manual). Such lines load and change nothing (measured: probe `active-comment`). PhaseCont keeps them as raw text, with an unsupported warning.
- Array blocks.
  - `%[i1..i2]` opens a block and `%` closes it (manual). Each must be alone on its line, and a comment after them is an error (PhaseCont choice).
  - A nested block, an unclosed block and a `%` that closes nothing are errors. The bounds are read as integers, and a negative bound is reported later, by the array pass (PhaseCont choice).
  - A malformed `%[` line is one error. A stray character on it (`%@`) is reported by the lexer only, and a block that is never closed is reported once. It still opens a block, so its `%` closes it without a second error and the lines in between stay inside it. The AST keeps such a block as `BadBlock`, which the array pass neither expands nor checks (PhaseCont choice).
- Includes.
  - `#include file` includes a file (manual).
  - Only lower-case `#include` followed by a blank is an include; `#includes` and `#INCLUDE` are comments, and so is `#done` (PhaseCont choice).
  - A blank followed by `#` starts a trailing comment, which is an error with a fix-it that deletes it: XPPAUT does not load the file when a comment follows the name (measured: probe `include-comment`). The include is still resolved, so the user sees one error and not a list of undefined names (PhaseCont choice).
  - Quotes around the name (`"inc.ode"` or `'inc.ode'`) are an error with a fix-it that removes them: XPPAUT 8.0 does not load the file, since the quotes become part of the name (measured: probe `include-quotes`). The name without quotes is still resolved (PhaseCont choice). An empty or blank name in quotes (`#include ""`, `#include " "`) is a malformed statement (ODE0020) with no fix-it, and nothing is loaded (PhaseCont choice).

### Names and order

- A name is a letter followed by letters, digits, `_` and `.` (manual: `aux P.E.=...`). A `.` followed by another `.` ends the name, so `x[1..3]` lexes as expected.
- XPPAUT 8.0 accepts a declared name longer than 10 characters but cannot read it back in a formula; it stops with "illegal expression" (measured: probes `name-11` and `name-11-collide-used`). Names of 10 characters work (probe `name-10`). Two declared names that agree in their first 10 characters do not collide (measured: probe `name-11-collide`). A name longer than 10 characters gives the portability warning (ODE0013), which says XPPAUT 8.0 stops where the name is used in a formula and has no note about other names. A name reported as reserved or duplicate gets no length warning.
- Reserved names cannot be declared as states, parameters, numbers, fixed quantities, aux quantities or functions: `t`, `pi`, every builtin function name, and `arg1` to `arg9` (manual "Reserved words"; measured: `t'=0` and `par sin=2` stop XPPAUT with "duplicate name").
- Declaring the same name twice is an error, including a repeated `par` (measured: XPPAUT stops with "duplicate name").
- Names may be used before their declaration, with one exception: XPPAUT evaluates fixed quantities in file order, so a fixed quantity that reads another fixed quantity defined on a later line sees 0 (measured: probe `fixed-forward`). PhaseCont makes that forward reference between fixed quantities an error. The same holds when a fixed quantity calls a user function that reads, directly or through other functions, a fixed quantity defined below the caller (measured: probe `fixed-forward-func`); the error sits on the call, with a note at the later fixed quantity. ODEs, maps and `aux` lines see the final values of all fixed quantities, wherever they are defined (measured: probe `fixed-aux-above`), so they give no diagnostic, and neither does a function body on its own.
- An `aux` name used inside a formula is an error (manual: aux quantities are output only).
- A user function may not call itself, directly or through other functions, since functions are inlined (PhaseCont choice).
- A file that mixes ODEs and maps is an error (PhaseCont choice; XPPAUT 8.0 loads it silently).
- The keyword words `not`, `if`, `then` and `else` are reserved too (PhaseCont choice). `e` is not reserved: it is an ordinary name, and Euler's number is `exp(1)`.
- A reserved name keeps its meaning when a file declares it anyway: `par sin=2` is reported once, and `sin(x)` still calls the builtin.
- Function parameters follow the same rules: a reserved parameter name is an error, a parameter named twice in one function is an error, and a parameter hides a global name with the same spelling inside the function body.
- `init x=...` and `x(0)=...` must name a state declared by an ODE, map or `markov` statement, and each state gets at most one initial value. Both are errors (PhaseCont choice). When the name is reserved and a statement tried to declare it (`t'=1` then `t(0)=2`), only the reserved name error is reported. A reserved name that nothing tried to declare (`init pi=1`) gives ODE0044 with the reason (`pi` is a constant).
- A cycle among fixed quantities (`a=b`, `b=a`) is an error, reported once at its earliest member with notes at the others. Uses inside a cycle are not also reported as forward references.
- A derived parameter (`!name=...`) may use numbers, parameters, `number` constants, other derived parameters, `pi`, builtins and user functions. It may not use `t`, states, fixed quantities, `wiener` names or `aux` quantities, since it is evaluated once, before integration (PhaseCont choice). The same holds for a user function it calls, directly or through other functions: the error sits on the call, with a note at the use inside the function (measured: probes `derived-state` and `derived-func-state`; XPPAUT 8.0 silently reads the initial value of the state). A derived parameter may use one defined on a later line (measured: probe `derived-later`, XPPAUT 8.0 gives `d=4` for a derived parameter used above its line); a cycle among derived parameters is an error.
- A function takes at most 9 parameters (manual). Inside a function body, `argK` names parameter `K` when the function has at least `K` parameters (manual: `arg1` to `arg9`). Any other use of `argK`, as a value or as a call, inside a function with fewer parameters or outside every function, is an error (PhaseCont choice; XPPAUT 8.0 reads it as 0).
- Calls are checked against the callee: a builtin against the table below, a user function against its parameter count, a table against one argument. Calling a value (`a(1)` for a parameter `a`) and using a function or table as a value (`x'=sin`) are errors.
- When a file mixes ODEs and maps, the model kind is the kind of the first ODE or map line, and the error sits on the first line of the other kind, with a note at the first line of the first kind. `markov` states do not set the kind.
- Undefined names and unsupported statements (PhaseCont choice). An undefined name is an error, unless the file has an unsupported statement whose text may declare names: a Volterra equation, an algebraic constraint (`0=...` or `solv`), `special`, a file table or `sum(..)of(..)`. Then the undefined name gives warning ODE0057 instead, with a note at the first such statement, since PhaseCont did not read the declaration that statement may hold. The same applies to an `init` name that names nothing. Active comments, boundary conditions, `export`, `only`, `hom_bcs` and `options` declare no names, so they leave the error in place.
- `check` ignores `#include` lines and `@`, `set` and `options` statements. It skips the bodies of array blocks and declarations whose name carries an index, and it never reports a name with an index as undefined; array expansion runs before it.

## Numbers

- Digits with an optional `.` and an optional exponent: `5`, `.5`, `5.`, `1.e2`, `1e+2`, `1E2` (measured).
- `e` or `E` starts an exponent only when a digit, or a sign and then a digit, follows it. So `2exp(x)` is `2` then `exp(x)`, and `2e-3x` is `2e-3` then `x`.
- A `.` followed by another `.` ends the number, so `1..10` is `1`, `..`, `10`.
- `1e2.5`, `0x10` and `1_000` are errors (measured).
- `1.2.3`, `.5.5`, `1e2.` and `00x10` are also errors (PhaseCont choice). A bad literal such as `1_000e+3` is one error that covers the whole literal, including a signed exponent.
- A literal that is too large for a double, such as `1e999`, is an error. A literal that underflows to zero, such as `1e-999`, is not (PhaseCont choice).
- `pi` is a constant. `e` is not (measured: `par e=2` is legal).

## Expressions

### Precedence

Loosest to tightest (measured; XPPAUT 8.0 differs from C here):

| Level | Operators | Associativity |
|---|---|---|
| 1 | binary `+ -`, `\|` | left, all at one level |
| 2 | `* /`, `&`, implicit multiplication | left, all at one level |
| 3 | prefix `-`, `not` | prefix; the operand is parsed at level 4 |
| 4 | `< > <= >= ==` | left, all at one level |
| 5 | `^`, `**` | left |
| 6 | number, name, call, `( e )`, `if(c)then(a)else(b)`, array index form | |

Measured examples, each of which is a parser test:

| Expression | Reads as | XPPAUT 8.0 value (`x=3`) |
|---|---|---|
| `1\|0+1` | `(1\|0)+1` | 2 |
| `2+1\|0` | `(2+1)\|0` | 1 |
| `2*1&1` | `(2*1)&1` | 1 |
| `1&1*2` | `(1&1)*2` | 2 |
| `3>2>1` | `(3>2)>1` | 0 |
| `2==2>1` | `(2==2)>1` | 0 |
| `2^3^2` | `(2^3)^2` | 64 |
| `2**3**2` | `(2**3)**2` | 64 |
| `-x^2` | `-(x^2)` | -9 |
| `-x>2` | `-(x>2)` | -1 |
| `-x*2` | `(-x)*2` | -6 |
| `-2&1` | `(-2)&1` | 1 |
| `not 0+1` | `(not 0)+1` | 2 |
| `not 1>2` | `not (1>2)` | 1 |
| `2*3>5` | `2*(3>5)` | 0 |
| `3>2+5` | `(3>2)+5` | 6 |
| `2^3>9` | `(2^3)>9` | 0 |

### Prefix operators

- XPPAUT 8.0 accepts prefix `-` only at the start of an expression, right after `(`, or right after `,` (measured: `2*-3`, `2+-1`, `2^-1`, `--2` and `x - -1` are all "Illegal syntax"). It accepts `not` after an operator but computes a wrong value there (measured: `0&not 0`, `5*not 0`, `2^not 0`, `-not 0` and `not not 0` all give 1).
- PhaseCont: a prefix `-` or `not` anywhere else is an error with a fix-it that wraps the prefix expression in parentheses (`2*(-3)`). The parser still builds the node and keeps going, so later errors are reported too.
- Unary `+` is an error (measured).

### Blanks inside formulas

XPPAUT deletes every blank in a formula before parsing it (measured: `1 2` is 12, `si n(0)` is 0, `x * * 2` is 9, `3< =3` is 1, `1. 5` is 1.5). It handles `not` before that (`not 0` is 1).

PhaseCont treats blanks as separators. Wherever deleting a blank would join its neighbors into one token, it reports an error with a fix-it that deletes the blank: a name or number followed by a name or number (`2 x`, `a x`, `1 2`, `si n`), `*` followed by `*`, and `<`, `>` or `=` followed by `=`. The words `not`, `if`, `then` and `else` are exempt as the left neighbor.

### Errors

- `!=`, `!` and `=` inside an expression are errors (measured: "illegal expression"). The `!=` message suggests `not(a==b)`.
- A call with the wrong number of arguments is an error (measured: `max(1,2,3)`, `atan2(1)`, `sin()`). Builtin arity is checked by `check`, using the builtin table below.

### Implicit multiplication

- A number literal followed directly by a name or `(`, with no blank between them, multiplies at level 2: `2v^2(1-w)` is `2*(v^2)*(1-w)` (PhaseCont choice, approved). Nothing else multiplies implicitly; `x(1)` stays a call and `)(` is an error.
- XPPAUT 8.0 rejects `2x` and `2(x)` (measured), so every use gives the portability warning with a fix-it that inserts `*`. After `/` the warning also says that `a/2v` reads as `(a/2)*v`.
- The AST marks implicit multiplication, and the writer always writes it back without `*`. Where the right operand would merge with the literal, the writer puts the right operand's first operand in parentheses, which the parser also reads as implicit: a number (`2(3)`), a bare index (`2([j])`), a name that starts with `e` or `E` and a digit (`2(e1)`, since `2e1` is one number), and a name that starts with `x` or `X` and a hex digit after a literal made only of zeros (`0(x1)`, since `0x1` is a malformed hex number). Only a hand-built product whose left side does not end with a literal is written with `*`.

### Precedence warnings

One warning code, with a fix-it that adds the parentheses XPPAUT's reading implies (PhaseCont choice), for:

- a comparison written without parentheses as a direct operand of `+ - * /` or prefix `-` (`2*x>1`);
- `&` or `|` and an arithmetic operator (`+ - * /`, implicit multiplication) without parentheses where one is a direct operand of the other, in either direction (`a|b+c`, `a+b|c`, `a&b+c`, `a+b&c`, `a*b&c`);
- a chain of `^` or `**` without parentheses (`a^b^c`).

Comparisons as operands of `&` and `|` give no warning.

### Portability warning

One warning code, "XPPAUT 8.0 rejects this", covers implicit multiplication, `ceil` and names longer than 10 characters. Each message names the construct.

## Arrays

Arrays are expanded while loading, after includes and before `check` (manual v8, pages 13–21; measured).

- **Range lines.** A statement with one `name[i1..i2]` is copied once per index `j` from `i1` to `i2`. `i1` may be larger than `i2` (the copies count down); both are non-negative integers.
  - The range may be on the declared name (`x[1..10]'=-x[j]`, `aux r[1..10]=...`, `c[1..10]=...`, `par a[1..5]=.25`, `wiener w[1..5]`, `number z[1..5]=...`) or inside an expression (`global 1 x[1..5]-1 {x[j]=0}`).
  - Other names on the same line use `[j]`: `par a[1..2]=.25,b[j]=.3` gives `par a1=.25,b1=.3` and `par a2=.25,b2=.3`.
  - A plain name on a shared range line is declared once per copy, so `par a[1..2]=.5, c=.1` declares `c` twice. That gives the duplicate-name error plus the note "`c` does not change with the index of this array line, so every copy declares it again". The note appears only when two copies of the same template, with different indices, declare the name at the same place; `b[j*0]` gets it too. A duplicate between two names of one copy (`par a[1..2]=.5, a[j]=.1`) or between two inclusions of one file is a plain duplicate.
- **Index forms.** `[j]`, `[j+n]`, `[j-n]` and `[j*n]` with an integer `n`. The expanded name is the root with the decimal index appended (`x[j-1]` at `j=3` is `x2`).
- **Index limit.** One limit, 0 to 4294967295 (`u32::MAX`), applies to every index, at two places. The parser reads the integers of a range on a name or in an expression (`x[1..10]`) and the `n` of `[j+n]`, `[j-n]` and `[j*n]` as plain digits that fit in a `u32`, 0 to 4294967295. A larger literal, a sign, or any other character makes the whole index form unsupported (ODE0015), so `x[0..4294967296]` and `y[j+4294967296]` give ODE0015 and `x[0..4294967295]` does not. The bounds of a block line `%[i1..i2]` are read as `i64`, and a block bound above 4294967295 gives ODE0069 as the Errors bullet says. A block bound that does not fit in `i64` (`%[0..9223372036854775808]`) makes the block line malformed (ODE0027). The value of an index form at some `j` is checked against 0 and 4294967295 after the copies are planned (ODE0068, ODE0069). A range within these limits can still pass the copy and work caps (ODE0071).
- **Bare index forms.** An index form on its own in an expression or a value position (`[j]`, `[j+1]`) expands to the integer (measured).
- **Blocks.** `%[i1..i2]` on its own line, then statements that use `[j]`, then `%` on its own line. The block is copied in index order: all of `j=i1`, then all of the next index.
- **Errors.** More than one range on a line (ODE0065, at the second range, with a note at the first); a range inside a block (ODE0066); `[j]` with no range in scope (ODE0067, once per statement, at the first index form); an index form not listed above (the parser reports it); a negative block bound, or an index form below 0 at some `j` (ODE0068, so `x[j-1]` on a line that starts at `j=0`); a block bound or index form above 4294967295 (ODE0069, `MAX_INDEX`, the same limit as a range line; PhaseCont choice); `markov` inside a block (the manual warns it fails) or with a range anywhere in it (ODE0070, PhaseCont choice, since the rows would be copied with the name).
- **A statement with an array error makes no copies** (PhaseCont choice). On a range line the whole line is dropped. In a block only the body statement with the error is dropped, and the rest of the body is copied; a negative or too large block bound drops the whole block. Each index form that goes out of bounds is reported once, naming the `j` where it does, not once per copy. The names a dropped statement would have declared (its ODE, map or initial-value target, or its `aux`, fixed, derived, `par`, `number`, `wiener`, `markov`, `table`, function or `set` names; every body statement of a dropped block) are recorded in `Model::suppressed`. `check` then reports no undefined name, and no undeclared `global` target, for a use of such a name: a plain name matches as written, and a name with an index matches its root followed by a decimal index (`0`, or digits with no leading zero). So `x[0..2]'=x[j-1]` then `y'=x1+x2` gives only the ODE0068 error (PhaseCont choice).
- An expanded name that is undefined, such as `x0` from `x[j-1]` at `j=1`, is the ordinary undefined-name error, pointing at the template line and naming the index.
- **Copy note.** Every copy keeps the template's spans plus its index value (`Stmt::array_index`) and its template id (`Stmt::array_template`: one id per range line, or per body statement of a block, in one load, so a file included twice gives two templates), so diagnostics point at the line the user wrote. Every `check` diagnostic about a copy carries the note "in the copy with j = N of this array line" at the template line (PhaseCont choice).
- **Collapse.** When several copies of one template give the same diagnostic (same code, severity, span and message, ignoring the copy note), only the one from the first copy is kept (PhaseCont choice). So `aux v=1` then `x[1..20]'=v` gives one error, while `x[1..3]'=y[j]` gives three, one per undefined name. Copies of a file included twice are separate templates, so each inclusion keeps its own error. The name length warning (ODE0013) from copies of one template at one place collapses to the first copy even though the names differ, with the note "N more copies of this array line give the same warning".
- **Cap.** One load makes at most 100000 statements as copies (`MAX_ARRAY_COPIES`) and does at most 2000000 units of work (`MAX_ARRAY_WORK`; both PhaseCont choices). Each copy costs the weight of its template: 1 for the statement, plus 1 per name, value and expression node in it, plus 1 per started 64 bytes of the template statement (its whole span, trailing comment included). So `x[j]'=-x[j]` weighs 4 + 1 = 5, a comment line of 4002 bytes in a block weighs 1 + 63 = 64, and a block costs the sum of its body statements per index, each weighed by its own span. The range line or block that would pass either limit is an error (ODE0071), reported once with the limit it passes, and makes no copies; every later range line and block also makes no copies, with no further message. Plain statements after it are kept. The count and the work are computed before any copy is made, so `%[0..4000000000]` fails at once, and so does a 4 KB `par` line with 400 indexed entries and 100000 copies. A block with an empty body makes nothing and never counts.
- **Blocks and other passes.** A malformed block that the parser kept as a bad block is kept as written. An `#include` inside a block is reported by the include pass and dropped by the expansion with no second message. A function whose name has an index (the parser reports it) is dropped.
- The writer writes the template form; `write_expanded` writes the flat form.

## Includes

`#include name.ode` is kept in the AST, and the parser itself never reads files (PhaseCont choice). A separate pass, `resolve_includes`, replaces each include with the statements of the named file, which it gets through a loader: `FsLoader` reads the file system and `MapLoader` holds text in memory. The template model keeps its `#include` lines for the writer; the resolved model is what array expansion and `check` see.

- **Paths.** A relative path is taken relative to the directory of the including file, and an absolute path is used as given (PhaseCont choice). XPPAUT 8.0 takes the path relative to the working directory instead (measured: probe `include-path`, a file in `sub/` that includes `inc.ode` loads `./inc.ode` when XPPAUT runs from the parent directory and `sub/inc.ode` when it runs from `sub/`). Inside MATLAB the working directory has no relation to the model file, so PhaseCont uses the including file. The one exception is the top file given to `FsLoader::load_top`, which may be relative to the working directory.
- **Opened file and display name.** `FsLoader` opens the joined path as it is, so the system follows symbolic links. The display name, used in messages and as the base for the file's own includes, is the joined path with only `.` components dropped; `..` is kept, since folding it as text after a symbolic link would name a different file (PhaseCont choice). The display name therefore always names the opened file.
- **Drive-relative paths.** On Windows, a name with a drive but no root folder (`C:inc.ode`) is an error, since the system reads it relative to that drive's working directory (PhaseCont choice).
- **Files.** Each included file gets its own file id, so its diagnostics point into it. A file is read, added and parsed once per `resolve_includes` call, however often it is included: its statements are copied in at every include, and its parse diagnostics are reported once. The includes inside a shared file are resolved at each site against that site's own display name, so one file reached under two names (through a symbolic link, say) can include different files at each; the statements copied in at a later site carry the file id of the first, which is harmless since the text is the same. Errors from the include pass itself (cycle, depth, quoted name, array block) are reported at each include that hits them. The count error (ODE0064) is the exception: it is reported once per call.
- **`done`.** A `done` line ends the included file only; the including file goes on after the `#include` line (measured: probe `include-done`).
- **Unreadable file.** A file the loader cannot read is an error at the `#include` line (PhaseCont choice; XPPAUT 8.0 ignores a missing include file without a message, measured: probe `include-missing`). The include is dropped.
- **Cycles.** A file that includes itself, directly or through other files, is an error at the `#include` line that closes the cycle (PhaseCont choice; XPPAUT 8.0 loops until it crashes, measured: probe `include-cycle`). Files are compared by the loader's identity key: `FsLoader` uses the canonical path where the system can give one and the display name with `dir/..` folded as text otherwise, so this is best effort; `MapLoader` uses the name as written. With `MapLoader`, a top file whose map name differs from the key it is included under (top `/m/model.ode` including `model.ode`) is copied in once before the cycle is found one level down. The include is dropped.
- **Depth.** The top file is at depth 0 and a file it includes is at depth 1. An include that would open a file at depth 17 or more is an error, and that file is not read (PhaseCont choice). The depth is checked before the file is read, so a cycle that would close at depth 17 is reported as too deep, not as a cycle.
- **Count.** One `resolve_includes` call reaches at most 4096 `#include` statements (`MAX_INCLUDES`), which bounds the number of files read and the number of include sites (PhaseCont choice). Every statement reached counts, resolved or not: unreadable files, cycles, quoted and empty names, includes in array blocks and every repeat of a file included more than once. The first statement past the limit is an error (ODE0064), reported once per call; it and every later include statement are dropped with no other diagnostic and nothing loaded.
- **Size.** The statements that included files add are also bounded, counted once per copy: at most 100000 statements (`MAX_INCLUDED_STATEMENTS`) and 1 MiB of statement text (`MAX_INCLUDED_BYTES`; both PhaseCont choices). The include whose file would pass either limit is an error (ODE0072), reported once per call; it and every later include are not resolved. The 4096 count alone does not bound the work, since a file with many statements included 4096 times is large. At the limits, a debug-build `pc-cli parse` took 0.1 to 0.8 s and 120 to 430 MB.
- **Same file twice.** Including one file twice without a cycle is allowed; `check` then reports the duplicate names as usual (PhaseCont choice).
- **Array blocks.** An `#include` inside a `%[..]` block is an error and is not resolved, since the block would copy the file once per index (PhaseCont choice).
- The errors for an unreadable file, a cycle and too deep a nesting name the chain of files, as in `a.ode -> b.ode -> a.ode`, and carry a note at each `#include` line that led to the file. Every include error in a nested file carries those notes. Control characters in names are escaped in messages (a tab shows as `\t`).

## Builtins

Arity `n` means the function takes exactly `n` arguments. "Smooth" means the function has the derivatives continuation needs on its domain. A model that uses a non-smooth function, `if`, a comparison (`<`, `<=`, `>`, `>=`, `==`), `&`, `|`, `not`, or any table, is flagged non-smooth, since comparisons and logic are step functions like `heav`. Every formula counts, including the body of a function that is never called. A model that uses a simulation-only function, or `global`, `markov` or `wiener`, is flagged simulation-only.

| Name | Arity | Meaning | Smooth | Sim. only | Source |
|---|---|---|---|---|---|
| `sin cos tan` | 1 | trigonometric, radians | yes | no | manual, measured (`tan(1)` = 1.5574077) |
| `asin acos atan` | 1 | inverse trigonometric | yes | no | manual, measured (`asin(1)` = 1.5707964) |
| `atan2` | 2 | `atan2(y,x)`, C argument order | yes | no | measured (`atan2(1,0)` = 1.5707964, `atan2(-1,-1)` = -2.3561945); the v5.41 manual's order is wrong for 8.0 |
| `sinh cosh tanh` | 1 | hyperbolic | yes | no | manual, measured |
| `exp` | 1 | exponential | yes | no | manual, measured (`exp(710)` = inf) |
| `ln`, `log` | 1 | natural logarithm | yes | no | measured (`log(100)` = 4.6051702); `log(-1)` is NaN |
| `log10` | 1 | base-10 logarithm | yes | no | measured (`log10(100)` = 2) |
| `sqrt` | 1 | square root | yes | no | manual, measured (`sqrt(-1)` is NaN) |
| `abs` | 1 | absolute value | no | no | manual, measured |
| `heav` | 1 | 1 for `x >= 0`, else 0 | no | no | manual, measured (`heav(0)` = `heav(-0)` = 1) |
| `sign` | 1 | -1, 0 or 1 | no | no | manual, measured (`sign(0)` = 0) |
| `mod` | 2 | `r = fmod(x,y)`, plus `y` when `r < 0` | no | no | measured (`mod(-7,3)` = 2, `mod(7,-3)` = 1, `mod(-7,-3)` = -4, `mod(-7.5,2)` = 0.5, `mod(1,0)` is NaN; probe `fn-mod-more`: `mod(-0.1,1)` = 0.9, `mod(2.5,-1)` = 0.5, `mod(-2.5,-1)` = -1.5, `mod(-123456.75,7)` = 2.25, `mod(0.75,-1)` = 0.75) |
| `min max` | 2 | minimum, maximum | no | no | manual, measured |
| `flr` | 1 | floor | no | no | measured (`flr(-2.5)` = -3, `flr(-2)` = -2) |
| `ceil` | 1 | ceiling | no | no | manual; XPPAUT 8.0 rejects it, so portability warning |
| `erf erfc` | 1 | error function, complement | yes | no | manual, measured (`erf(1)` = 0.84270078) |
| `lgamma` | 1 | `ln\|Γ(x)\|` | yes, away from poles | no | measured (`lgamma(5)` = 3.1780539, `lgamma(-0.5)` = 1.2655121) |
| `besselj bessely` | 2 | `J_n(x)`, `Y_n(x)`; order truncated toward zero; `\|order\| > 10000` gives NaN | yes in `x` | no | manual, measured (`besselj(1.5,1)` = `besselj(1,1)`, `besselj(-0.5,1)` = `besselj(0,1)`) |
| `ran` | 1 | uniform on `[0, a]` | no | yes | manual, measured only at 0 (probe `fn-random`: `ran(0)` = 0; the draws cannot be compared) |
| `normal` | 2 | normal with mean `m` and variance `v` | no | yes | manual |
| `poisson` | 1 | Poisson sample | no | yes | manual, measured only at 0 (probe `fn-random`: `poisson(0)` = 0) |
| `delay` | 2 | `delay(x,tau)` | | | manual; warning, v2 |
| `shift`, `del_shft`, `sum(..)of(..)` | | | | | manual; unsupported, warning |
| `if(c)then(a)else(b)` | | `a` when `c` is nonzero, else `b` | no | no | manual, measured (`if(-1)` takes `then`) |

`delay(x,tau)` gives warning ODE0054 (planned for v2), and `shift` and `del_shft` give ODE0055 and ODE0056; the calls are kept as written. `ceil` gives the portability warning, since XPPAUT 8.0 rejects it. The `if(c)then(a)else(b)` and `sum(..)of(..)` rows are not calls: `if` is an expression form that makes the model non-smooth, and the parser keeps a `sum(..)of(..)` line as an unsupported statement. A bare call `sum(...)` gives warning ODE0037. `crates/pc-ode/src/builtins.rs` holds the same table, and a test checks that the two agree.

Operator semantics (measured):

- Comparisons return 1 or 0. `&`, `|`, `not` and `if` treat any nonzero value as true (`0.5&1` = 1, `-0.5&1` = 1).
- `**` is the same as `^`. `0^0` = 1, `(-2)^3` = -8, `(-8)^(1/3)` is NaN.
- `t` and `arg1` read as 0 outside the context that defines them.
- Division: XPPAUT 8.0 guards it (`1/0` = 4.4843049e14, `0/0` = 0). PhaseCont uses IEEE division, so `1/0` is inf and `0/0` is NaN (PhaseCont choice). XPPAUT's guard puts a jump with an undocumented constant into the model.
- `if(c)then(a)else(b)` evaluates both branches and picks one, which cannot be told apart from lazy evaluation since there are no floating-point traps (PhaseCont choice). The test is `c != 0`, so a NaN condition takes `then`. `min(x,y)` is `x < y ? x : y` and `max(x,y)` is `x > y ? x : y`, so a NaN in `x` returns `y`.
- `pc-ir` folds `x*0` to 0, `x*1` and `x+0` to `x`, `x-x` to 0 and `-(-x)` to `x`, and picks the branch of an `if` whose condition is constant (PhaseCont choice). A NaN or inf in a term that folding removes therefore does not reach the result: `x*0` gives 0 even when `x` is inf. Folding never reorders operations, so a finite nonzero result keeps its bits unless it reads the sign of a zero that folding changed. The sign of a zero can change: `x*0` folds to `+0` even when `x` is negative, and `0-x` folds to `-x`, which is `-0` for `x = +0` where the unfolded `0-0` is `+0`. This matters only where the sign of a zero is read, for example `atan2(x*0,-1)` at `x = -1`, which gives `+pi` in the DAG and the JIT (the `x*0` became `+0`) and `-pi` in the `Fallback` interpreter (`-1*0` is `-0`). A test in `crates/pc-ir/src/diff_tests.rs` pins this.

## Writer

`write` (`crates/pc-ode/src/writer.rs`) turns a model back into `.ode` text, and `parse` of that text gives the same model, spans aside (PhaseCont choice; property tests in `crates/pc-ode/tests/round_trip.rs`). `write_expanded` is the same function, called on the expanded model.

- **Canonical text.** One statement per line. Keywords are written in full and in lower case (`par`, `init`, `aux`, `wiener`, `table`, `global`, `markov`, `done`). Lists use `,` with no blanks (`par a=1,b=2`), `@` lines use `, `, `global` assignments use `;`. ODEs use `x'=` unless they were written `dx/dt=`.
- **Expressions.** Parentheses where the precedence above needs them, and also where they are not needed for the tree: where the parser would give a precedence warning (a comparison or `&`/`|` as an operand of arithmetic, arithmetic as an operand of `&`/`|`, `-` of a comparison: `2*(x > 1)`, `a|(b*c)`, `-(a < b)`), around the left comparison or power of a chain (`(a < b) < c`, `(a^b)^c`), and around a whole operand that starts with a prefix `-` or `not` and does not start the expression (after `=`, `(` or `,`): `a + (-2x)`, `a*(-b)`. Binary `+`, `-` and the comparisons get a blank on each side; `* / ^ ** & |` get none.
- **Spelling kept.** Names keep their spelling. Number literals keep the text they were written with (`1.50E+2`, `.5`). A literal without text gets the shortest digits that read back as the same value, in plain notation when the decimal exponent is in -5..=15 and in `e` notation otherwise.
- **Implicit multiplication.** Always written as juxtaposition (`2x`, `2(x + 1)`). When the right side would merge with the literal (a number, a bare index, a name that starts with `e`/`E` and a digit, or after an all-zero literal a name that starts with `x`/`X` and a hex digit), the first operand of the right side goes in parentheses: `2(3)`, `2([j])`, `2(e1)^2`, `0(x1)`. The warning-only parentheses are left out of the left side, since `(a<2)x` is not implicit, so `a<2x` is written `a < 2x` with the same precedence warning as the source. Only a hand-built product whose left side does not end with a literal is written with `*`.
- **Line starts and ends.** A text whose first line would start with a digit (`0=x-y`) gets a blank first line, since a leading digit marks an old-format file. A line that would end in `\` outside a `#` comment gets a blank after the `\`, so it does not continue the line.
- **Kept as written.** `#include` lines, range lines, `%` blocks, comments, trailing comments and unsupported statements (as their raw text).
- **Errors.** A model with errors is not guaranteed to round-trip. Error nodes are written as `?`, a malformed `%[..]` line as `%[]`, and a literal that is not finite as `1e999`; each reparses with an error.
- **Other limits.** A negative `Expr::Num` value (hand-built only) is written as `-2`, or `(-2)` where a prefix needs parentheses, and reparses as `-` of `2`. `format_number(-0.0)` gives `-0`. A comment or include path that ends in a lone `\r` loses it on reparse.

## Diagnostic codes

Every diagnostic has a stable code, `ODE` plus four digits. The table lists them in order, with the default severity from the `CODES` table in `crates/pc-ode/src/diag.rs`. New codes are appended and a code is never renumbered or reused. A test checks that this table and `CODES` have the same codes, in the same order, with the same severities. Each code has a sample file under `crates/pc-ode/tests/invalid/`, except ODE0018.

Each pass (include, expand, check) keeps at most 10000 diagnostics (`MAX_DIAGNOSTICS`, a PhaseCont choice) and counts the rest without storing them, so a hostile file cannot use memory through its diagnostics. `load` keeps at most 10000 in total, sorted by position, and ends with one error, ODE0073, "N more diagnostics not shown". The summary can push the list to 10001. Copies of one array line that give the same diagnostic collapse as they arrive, so they do not fill the list. Lexer and parser diagnostics are bounded by the input size and have no cap of their own.

| Code | Severity | Meaning |
|---|---|---|
| ODE0001 | error | A number literal is malformed, such as `1.2.3`, `0x10` or `1_000` ([Numbers](#numbers)). |
| ODE0002 | error | A number literal is too large for a double, such as `1e999` ([Numbers](#numbers)). |
| ODE0003 | error | A character that no token can start with, including a non-ASCII character outside a comment ([Lines](#lines)). |
| ODE0004 | error | An expression is missing where one is required, as in `x'=`. |
| ODE0005 | error | Two operands follow each other with no operator, as in `(1)(2)`. The fix-it inserts `*` ([Implicit multiplication](#implicit-multiplication)). |
| ODE0006 | error | A token that cannot appear at this point of an expression, such as the `,` in `(1,2)`. |
| ODE0007 | error | `!` or `!=` inside an expression ([Errors](#errors)). |
| ODE0008 | error | A single `=` inside an expression ([Errors](#errors)). |
| ODE0009 | error | A `(` or `[` has no closing partner. |
| ODE0010 | error | A prefix `-` or `not` where XPPAUT 8.0 does not accept it, as in `2*-3`. The fix-it adds parentheses ([Prefix operators](#prefix-operators)). |
| ODE0011 | error | A unary `+` ([Prefix operators](#prefix-operators)). |
| ODE0012 | error | A blank sits inside what XPPAUT 8.0 reads as one token, such as `1 2` or `x * * 2`. The fix-it deletes the blank ([Blanks inside formulas](#blanks-inside-formulas)). |
| ODE0013 | warning | XPPAUT 8.0 rejects the construct: implicit multiplication, a builtin XPPAUT 8.0 lacks (`ceil`), or a name longer than 10 characters, which XPPAUT 8.0 cannot read in a formula ([Portability warning](#portability-warning)). |
| ODE0014 | warning | An operator mix whose XPPAUT 8.0 precedence differs from C, such as `2*x>1` or `a^b^c`. The fix-it adds parentheses ([Precedence warnings](#precedence-warnings)). |
| ODE0015 | error | An array index form that is not `[i1..i2]`, `[j]`, `[j+n]`, `[j-n]` or `[j*n]` with unsigned integers that fit in 32 bits, or a range `[i1..i2]` used as a value instead of after a name ([Arrays](#arrays)). |
| ODE0016 | error | An `if(c)then(a)else(b)` expression that is missing a part or a closing parenthesis. |
| ODE0017 | error | An expression nested more than 200 levels deep. |
| ODE0018 | error | The first line of the file starts with a digit, so the file is in the old XPPAUT format ([Lines](#lines)). |
| ODE0019 | error | A line that matches no statement form ([Statement details](#statement-details)). |
| ODE0020 | error | A line with a known statement form whose parts are wrong, such as `table` with no name or an `#include` with an empty quoted name ([Statement details](#statement-details)). |
| ODE0021 | error | Text after the end of a statement, as in `x'=1 )`. |
| ODE0022 | error | A trailing comment after any statement other than a `par`, `number` or `@` list. The fix-it deletes the comment ([Lines](#lines)). |
| ODE0023 | error | The value in a `par`, `number` or `init` list, or after `x(0)=`, is not a number literal or a bare index form ([Values](#values)). |
| ODE0024 | warning | An `@` option name that PhaseCont does not know ([Lists](#lists)). |
| ODE0025 | warning | An `@` option with blanks around `=`, which XPPAUT 8.0 ignores. The fix-it removes the blanks ([Lists](#lists)). |
| ODE0026 | error | A `markov` statement whose rows are wrong: too few rows, a row with the wrong number of `{expr}` groups, or an unclosed `{`; a row after the last one is read as a separate line (ODE0019) ([Statement details](#statement-details)). |
| ODE0027 | error | A malformed array block: a bad `%[i1..i2]` line, a nested block, an unclosed block, or a `%` that closes nothing ([Statement details](#statement-details)). |
| ODE0028 | warning | A Volterra equation, which is kept as raw text and not supported ([Statement forms](#statement-forms)). |
| ODE0029 | warning | A boundary value statement (`bndry`, `bdry`), kept as raw text and not supported. |
| ODE0030 | warning | An algebraic constraint (`0=...`, `solv`), kept as raw text and not supported. |
| ODE0031 | warning | A `special` statement, kept as raw text and not supported. |
| ODE0032 | warning | An `export` statement, kept as raw text and not supported. |
| ODE0033 | warning | An `only` statement, kept as raw text and not supported. |
| ODE0034 | warning | An `options` file statement, kept as raw text and not supported. |
| ODE0035 | warning | A `hom_bcs` statement, kept as raw text and not supported. |
| ODE0036 | warning | A table read from a file, kept as raw text and not supported. |
| ODE0037 | warning | A `sum(..)of(..)` form, kept as raw text and not supported ([Builtins](#builtins)). |
| ODE0038 | warning | An active comment, a line that starts with `"`. It is kept as raw text and changes nothing ([Statement details](#statement-details)). |
| ODE0039 | error | A name that no statement declares ([Names and order](#names-and-order)). |
| ODE0040 | error | A reserved name (`t`, `pi`, a builtin, `argK`, `not`, `if`, `then`, `else`) is declared, or used as a function parameter ([Names and order](#names-and-order)). |
| ODE0041 | error | A name is declared twice ([Names and order](#names-and-order)). |
| ODE0042 | error | A fixed quantity reads another fixed quantity that is defined on a later line, directly or through a user function ([Names and order](#names-and-order)). |
| ODE0043 | error | Fixed quantities or derived parameters that depend on each other in a cycle ([Names and order](#names-and-order)). |
| ODE0044 | error | An initial value for a name that is not a state ([Names and order](#names-and-order)). |
| ODE0045 | error | A state has more than one initial value ([Names and order](#names-and-order)). |
| ODE0046 | error | A user function with more than 9 parameters ([Names and order](#names-and-order)). |
| ODE0047 | error | A user function that calls itself, directly or through other functions ([Names and order](#names-and-order)). |
| ODE0048 | error | A call with the wrong number of arguments ([Errors](#errors)). |
| ODE0049 | error | A function, table or builtin used as a value, or a value (`t`, `pi`, a parameter) called as a function ([Names and order](#names-and-order)). |
| ODE0050 | error | `argK` outside a function that has at least `K` parameters ([Names and order](#names-and-order)). |
| ODE0051 | error | An `aux` quantity used inside a formula ([Names and order](#names-and-order)). |
| ODE0052 | error | The file has both ODEs and maps ([Names and order](#names-and-order)). |
| ODE0053 | error | A derived parameter uses something other than parameters, numbers, derived parameters, builtins and functions ([Names and order](#names-and-order)). |
| ODE0054 | warning | A call of `delay`, which is kept and planned for v2 ([Builtins](#builtins)). |
| ODE0055 | warning | A call of `shift`, which is kept and not supported ([Builtins](#builtins)). |
| ODE0056 | warning | A call of `del_shft`, which is kept and not supported ([Builtins](#builtins)). |
| ODE0057 | warning | A name that no statement declares, in a file with an unsupported statement that may declare it. It replaces ODE0039 there, and ODE0044 or ODE0058 for an undeclared `init` or `global` name ([Names and order](#names-and-order)). |
| ODE0058 | error | A `global` assignment to a name that is not a state or a parameter ([Statement details](#statement-details)). |
| ODE0059 | error | An `#include` file name in quotes. The fix-it removes the quotes ([Includes](#includes)). |
| ODE0060 | error | An included file that the loader cannot read ([Includes](#includes)). |
| ODE0061 | error | A file that includes itself, directly or through other files ([Includes](#includes)). |
| ODE0062 | error | Includes nested more than 16 levels deep ([Includes](#includes)). |
| ODE0063 | error | An `#include` inside a `%[..]` array block ([Includes](#includes)). |
| ODE0064 | error | More than 4096 `#include` statements reached in one load ([Includes](#includes)). The statement and size limits give ODE0072. |
| ODE0065 | error | More than one range in a statement ([Arrays](#arrays)). |
| ODE0066 | error | A range inside a `%[..]` array block ([Arrays](#arrays)). |
| ODE0067 | error | An index form such as `[j]` with no range in scope ([Arrays](#arrays)). |
| ODE0068 | error | A negative block bound, or an index form that is below 0 for some `j` ([Arrays](#arrays)). |
| ODE0069 | error | A block bound, or an index form at some `j`, above 4294967295 ([Arrays](#arrays)). |
| ODE0070 | error | A `markov` statement inside an array block or with a range ([Arrays](#arrays)). |
| ODE0071 | error | Array expansion would pass the copy limit or the work limit ([Arrays](#arrays)). |
| ODE0072 | error | The included files add more than 100000 statements or 1 MiB of text in one load ([Includes](#includes)). |
| ODE0073 | error | More than 10000 diagnostics in one pass or one load. The message gives the number not shown ([Diagnostic codes](#diagnostic-codes)). |
