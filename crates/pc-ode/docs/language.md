# The `.ode` language accepted by `pc-ode`

This file is the rule source for the `pc-ode` parser, checker and writer. It covers the subset of the XPPAUT `.ode` format that PhaseCont v1 reads, and says where each rule comes from.

## Sources

Each rule carries one of these sources:

- **manual**: the XPPAUT documentation (online help pages, the v5.0 and v5.41 PDF manuals, the v8 notes). Only the documentation was read, never XPPAUT source code.
- **measured**: XPPAUT 8.0 run headless (`xppaut file.ode -silent -outfile o.dat`) on our own probe files in `references/xpp-probes/`. XPPAUT writes its output file with float32 precision, so measured values carry about 7 significant digits (`0.15` prints as `0.15000001`).
- **PhaseCont choice**: a rule we picked where XPPAUT gives no usable answer, or where we deliberately differ.
- **open**: not settled. These do not affect parsing; they block P2, where `pc-ir` evaluates expressions.

Where XPPAUT 8.0 and the manual disagree, the measured behavior wins unless a rule says otherwise.

## Lines

- A file is a sequence of lines. A backslash at the end of a line joins it with the next line (manual). The lexer treats backslash plus newline, or backslash plus `\r\n`, as a blank, so spans point into the original text.
- A line whose first non-blank character is `#` is a comment, except `#include` (manual). Comments are kept in the AST and written back.
- A trailing `#` comment after a statement is accepted after a `par`, `number` or `@` list, kept on the statement and written back (measured: XPPAUT 8.0 ignores it there). After any other statement it is an error (measured for formulas and `init`, which XPPAUT rejects; PhaseCont choice for the statements not measured).
- A file whose first line starts with a digit is in the old XPPAUT format, which is an error (manual).
- Case is ignored in names and keywords (manual, measured). The AST keeps the spelling as written and uses the lower-case form as the lookup key.
- A stray character is an error. That includes any non-ASCII character outside a comment, and a backslash that is not the last character of a line (PhaseCont choice).
- A byte order mark (U+FEFF) at the very start of a file is ignored. Anywhere else it is a stray character (PhaseCont choice).
- A lone `\r` that is not followed by `\n` is a blank (PhaseCont choice).

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
  - The syntax is `set name {a=1,b=2}` (manual), and `set low {a=1,b=2}` loads and works (measured).
  - Blanks also separate entries, as in `set s {a=2 b=3}`, and an empty list `set s {}` loads (both measured). Blanks around `=` give no diagnostic (PhaseCont choice).
- `global` lines.
  - The form is `global sign cond {name=expr; ...}` (manual), and `global 1 x-0.5 {x=0;y=2}` loads and works (measured).
  - The sign must be the literal `-1`, `0` or `1` (PhaseCont choice).
  - An empty list `{}` is an error: XPPAUT rejects it with "Bad global" (measured).
- `markov` lines.
  - `markov name n` is followed by `n` rows, each with `n` `{expr}` groups (manual).
  - XPPAUT ignores extra text after `n`, as in `markov z 2 1` (measured). PhaseCont reports it with a fix-it that deletes it (PhaseCont choice). Missing rows are an error too (PhaseCont choice).
  - Blank lines and comment lines may sit between the rows. The comments are kept inside the `markov` statement in source order, and the blank lines are not stored (PhaseCont choice). An `#include` line ends the rows.
- `table name % npts lo hi expr`: `table h % 11 0 1 2*t` loads and works (measured). A missing or bad field is one error that names the field, and the rest of the line is skipped (PhaseCont choice).
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
- Active comments: a line that starts with `"` is an XPPAUT active comment (manual). Such lines load and change nothing (measured). PhaseCont keeps them as raw text, with an unsupported warning.
- Array blocks.
  - `%[i1..i2]` opens a block and `%` closes it (manual). Each must be alone on its line, and a comment after them is an error (PhaseCont choice).
  - A nested block, an unclosed block and a `%` that closes nothing are errors. The bounds are read as integers, and a negative bound is reported later, by the array pass (PhaseCont choice).
  - A malformed `%[` line is one error. It still opens a block, so its `%` closes it without a second error and the lines in between stay inside it. The AST keeps such a block as `BadBlock`, which the array pass neither expands nor checks (PhaseCont choice).
- Includes.
  - `#include file` includes a file (manual).
  - Only lower-case `#include` followed by a blank is an include; `#includes` and `#INCLUDE` are comments, and so is `#done` (PhaseCont choice).
  - A blank followed by `#` starts a trailing comment, which is an error with a fix-it that deletes it: XPPAUT does not load the file when a comment follows the name (measured). Quotes around the name are kept as written for now.

### Names and order

- A name is a letter followed by letters, digits, `_` and `.` (manual: `aux P.E.=...`). A `.` followed by another `.` ends the name, so `x[1..3]` lexes as expected.
- XPPAUT 8.0 truncates names to 10 characters, so names that agree in their first 10 characters collide (measured). A name longer than 10 characters gives the portability warning. When two declared names agree in their first 10 characters, ignoring case, the warning adds a note.
- Reserved names cannot be declared as states, parameters, numbers, fixed quantities, aux quantities or functions: `t`, `pi`, every builtin function name, and `arg1` to `arg9` (manual "Reserved words"; measured: `t'=0` and `par sin=2` stop XPPAUT with "duplicate name").
- Declaring the same name twice is an error, including a repeated `par` (measured: XPPAUT stops with "duplicate name").
- Names may be used before their declaration, with one exception: XPPAUT evaluates fixed quantities in file order, so a fixed quantity that reads another fixed quantity defined on a later line sees 0 (measured). PhaseCont makes that forward reference between fixed quantities an error. ODEs, maps, `aux` lines and user functions see the final values of all fixed quantities, wherever they are defined (measured: an `aux` above `b=...` sees the value of `b`), so they give no diagnostic.
- An `aux` name used inside a formula is an error (manual: aux quantities are output only).
- A user function may not call itself, directly or through other functions, since functions are inlined (PhaseCont choice).
- A file that mixes ODEs and maps is an error (PhaseCont choice; XPPAUT 8.0 loads it silently).
- The keyword words `not`, `if`, `then` and `else` are reserved too (PhaseCont choice). `e` is not reserved: it is an ordinary name, and Euler's number is `exp(1)`.
- A reserved name keeps its meaning when a file declares it anyway: `par sin=2` is reported once, and `sin(x)` still calls the builtin.
- Function parameters follow the same rules: a reserved parameter name is an error, a parameter named twice in one function is an error, and a parameter hides a global name with the same spelling inside the function body.
- `init x=...` and `x(0)=...` must name a state declared by an ODE, map or `markov` statement, and each state gets at most one initial value. Both are errors (PhaseCont choice).
- A cycle among fixed quantities (`a=b`, `b=a`) is an error, reported once at its earliest member with notes at the others. Uses inside a cycle are not also reported as forward references.
- A derived parameter (`!name=...`) may use numbers, parameters, `number` constants, other derived parameters, `pi`, builtins and user functions. It may not use `t`, states, fixed quantities, `wiener` names or `aux` quantities, since it is evaluated once, before integration (PhaseCont choice). A derived parameter may use one defined on a later line; a cycle among derived parameters is an error.
- A function takes at most 9 parameters (manual). Inside a function body, `argK` names parameter `K` when the function has at least `K` parameters (manual: `arg1` to `arg9`). Any other use of `argK`, inside a function with fewer parameters or outside every function, is an error (PhaseCont choice; XPPAUT 8.0 reads it as 0).
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
- The AST marks implicit multiplication, and the writer writes it back without `*`. The one exception: when the right operand starts with `e` or `E` followed by a digit, juxtaposition would re-lex as one number (`2` then `e1` would read as `2e1`), so the writer writes `*`. The parser never produces that case; only hand-built ASTs lose the flag.

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
  - A plain name on a shared range line is declared once per copy, so `par a[1..2]=.5, c=.1` declares `c` twice. That gives the duplicate-name error plus a note that explains the copying.
- **Index forms.** `[j]`, `[j+n]`, `[j-n]` and `[j*n]` with an integer `n`. The expanded name is the root with the decimal index appended (`x[j-1]` at `j=3` is `x2`).
- **Bare index forms.** An index form on its own in an expression or a value position (`[j]`, `[j+1]`) expands to the integer (measured).
- **Blocks.** `%[i1..i2]` on its own line, then statements that use `[j]`, then `%` on its own line. The block is copied in index order: all of `j=i1`, then all of the next index.
- **Errors.** More than one range on a line; a range inside a block; `[j]` with no range in scope; an index form not listed above; a negative index; `markov` inside a block (the manual warns it fails). An expanded name that is undefined, such as `x0` from `x[j-1]` at `j=1`, is the ordinary undefined-name error, pointing at the template line and naming the index.
- Every copy keeps the template's span plus its index value, so diagnostics point at the line the user wrote. The writer writes the template form; `write_expanded` writes the flat form.

## Includes

`#include name.ode` is kept in the AST. A separate pass resolves it through a loader, relative to the directory of the including file, with cycle detection and a depth limit. A `done` line ends the included file only. The parser itself never reads files (PhaseCont choice; the manual says nothing about paths).

## Builtins

Arity `n` means the function takes exactly `n` arguments. "Smooth" means the function has the derivatives continuation needs on its domain. A model that uses a non-smooth function, or any table, is flagged non-smooth. A model that uses a simulation-only function, or `global`, `markov` or `wiener`, is flagged simulation-only.

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
| `mod` | 2 | `r = fmod(x,y)`, plus `y` when `r < 0` | no | no | measured (`mod(-7,3)` = 2, `mod(7,-3)` = 1, `mod(-7,-3)` = -4, `mod(-7.5,2)` = 0.5, `mod(1,0)` is NaN); formula fits the data, open for P2 |
| `min max` | 2 | minimum, maximum | no | no | manual, measured |
| `flr` | 1 | floor | no | no | measured (`flr(-2.5)` = -3, `flr(-2)` = -2) |
| `ceil` | 1 | ceiling | no | no | manual; XPPAUT 8.0 rejects it, so portability warning |
| `erf erfc` | 1 | error function, complement | yes | no | manual, measured (`erf(1)` = 0.84270078) |
| `lgamma` | 1 | `ln\|Γ(x)\|` | yes, away from poles | no | measured (`lgamma(5)` = 3.1780539, `lgamma(-0.5)` = 1.2655121) |
| `besselj bessely` | 2 | `J_n(x)`, `Y_n(x)`; order truncated toward zero | yes in `x` | no | manual, measured (`besselj(1.5,1)` = `besselj(1,1)`, `besselj(-0.5,1)` = `besselj(0,1)`) |
| `ran` | 1 | uniform on `[0, a]` | no | yes | manual, measured |
| `normal` | 2 | normal with mean `m` and variance `v` | no | yes | manual |
| `poisson` | 1 | Poisson sample | no | yes | manual, measured |
| `delay` | 2 | `delay(x,tau)` | | | manual; warning, v2 |
| `shift`, `del_shft`, `sum(..)of(..)` | | | | | manual; unsupported, warning |
| `if(c)then(a)else(b)` | | `a` when `c` is nonzero, else `b` | no | no | manual, measured (`if(-1)` takes `then`) |

`delay(x,tau)` gives warning ODE0054 (planned for v2), and `shift` and `del_shft` give ODE0055 and ODE0056; the calls are kept as written. `ceil` gives the portability warning, since XPPAUT 8.0 rejects it. The `if(c)then(a)else(b)` and `sum(..)of(..)` rows are not calls: `if` is an expression form that makes the model non-smooth, and the parser keeps a `sum(..)of(..)` line as an unsupported statement. A bare call `sum(...)` gives warning ODE0037. `crates/pc-ode/src/builtins.rs` holds the same table, and a test checks that the two agree.

Operator semantics (measured):

- Comparisons return 1 or 0. `&`, `|`, `not` and `if` treat any nonzero value as true (`0.5&1` = 1, `-0.5&1` = 1).
- `**` is the same as `^`. `0^0` = 1, `(-2)^3` = -8, `(-8)^(1/3)` is NaN.
- `t` and `arg1` read as 0 outside the context that defines them.
- Division: XPPAUT 8.0 guards it (`1/0` = 4.4843049e14, `0/0` = 0). PhaseCont keeps IEEE division for now. **Open for P2.**
- Whether `if` evaluates both branches cannot be seen from values, since XPPAUT has no traps. **Open for P2.**

## Diagnostic codes

Filled in by Task 13 of Phase 1. Every code in the `CODES` table in `src/diag.rs` is listed here with its severity and meaning.
