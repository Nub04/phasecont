import assert from "node:assert/strict";
import { test, before } from "node:test";
import * as fs from "node:fs";
import * as path from "node:path";
import {
  INITIAL,
  Registry,
  parseRawGrammar,
  type IGrammar,
} from "vscode-textmate";
import { createOnigScanner, createOnigString, loadWASM } from "vscode-oniguruma";

const root = path.resolve(__dirname, "..", "..");
const grammarFile = path.join(root, "syntaxes", "ode.tmLanguage.json");

let grammar: IGrammar;

before(async () => {
  const wasm = fs.readFileSync(
    require.resolve("vscode-oniguruma/release/onig.wasm"),
  );
  await loadWASM({ data: wasm });
  const registry = new Registry({
    onigLib: Promise.resolve({ createOnigScanner, createOnigString }),
    loadGrammar: async (scope) => {
      if (scope !== "source.ode") {
        return null;
      }
      return parseRawGrammar(fs.readFileSync(grammarFile, "utf8"), grammarFile);
    },
  });
  const g = await registry.loadGrammar("source.ode");
  assert.ok(g, "grammar source.ode did not load");
  grammar = g;
});

interface Tok {
  text: string;
  scopes: string[];
}

/** Tokenizes one line, or several lines joined by "\n". */
function tokenize(src: string): Tok[][] {
  let state = INITIAL;
  return src.split("\n").map((line) => {
    const r = grammar.tokenizeLine(line, state);
    state = r.ruleStack;
    return r.tokens.map((t) => ({
      text: line.slice(t.startIndex, t.endIndex),
      scopes: t.scopes,
    }));
  });
}

/** The innermost scope of the `nth` token whose text is `text`. */
function scopeOf(src: string, text: string, nth = 0, line = 0): string {
  const hits = tokenize(src)[line]!.filter((t) => t.text === text);
  const tok = hits[nth];
  assert.ok(tok, `no token ${JSON.stringify(text)} (#${nth}) in ${src}`);
  return tok.scopes[tok.scopes.length - 1]!;
}

function hasScope(src: string, text: string, scope: string, nth = 0): boolean {
  const hits = tokenize(src)[0]!.filter((t) => t.text === text);
  const tok = hits[nth];
  assert.ok(tok, `no token ${JSON.stringify(text)} in ${src}`);
  return tok.scopes.some((s) => s === scope);
}

test("comments", () => {
  assert.equal(scopeOf("# hello", "# hello"), "comment.line.number-sign.ode");
  assert.equal(
    scopeOf("  # indented", "# indented"),
    "comment.line.number-sign.ode",
  );
  assert.equal(
    scopeOf("x'=-x # decay", "# decay"),
    "comment.line.number-sign.ode",
  );
  assert.equal(
    scopeOf('"an active comment', '"an active comment'),
    "comment.line.double-quote.ode",
  );
});

test("#include is lower case with a blank after it", () => {
  assert.equal(
    scopeOf("#include other.ode", "#include"),
    "keyword.control.import.include.ode",
  );
  assert.equal(
    scopeOf("#include other.ode", "other.ode"),
    "string.unquoted.filename.ode",
  );
  assert.equal(
    scopeOf("#includes other.ode", "#includes other.ode"),
    "comment.line.number-sign.ode",
  );
  assert.equal(
    scopeOf("#INCLUDE other.ode", "#INCLUDE other.ode"),
    "comment.line.number-sign.ode",
  );
});

test("declarations compare only the first letter", () => {
  for (const line of [
    "par a=1",
    "param a=1",
    "pzz a=2",
    "aux v=x",
    "init x=1",
    "wiener w",
    "table h % 101 0 6.283 sin(t)",
    "markov z 2",
    "bdry x-1",
    "global 1 x-1 {x=0}",
    "done",
    "done # finished",
  ]) {
    const first = line.split(/[ \t]/)[0]!;
    assert.equal(
      scopeOf(line, first),
      "keyword.control.declaration.ode",
      line,
    );
  }
});

test("a first word followed by = is an assignment, not a declaration", () => {
  const toks = tokenize("p = 3")[0]!;
  assert.ok(
    !toks.some((t) => t.scopes.includes("keyword.control.declaration.ode")),
  );
  const toks2 = tokenize("amp=3")[0]!;
  assert.ok(
    !toks2.some((t) => t.scopes.includes("keyword.control.declaration.ode")),
  );
});

test("full-word keywords need the whole word", () => {
  for (const word of [
    "number",
    "set",
    "options",
    "only",
    "special",
    "solv",
    "export",
    "volt",
    "hom_bcs",
  ]) {
    assert.equal(
      scopeOf(`${word} a=1`, word),
      "keyword.control.declaration.ode",
      word,
    );
  }
  assert.ok(
    !tokenize("numbers a=1")[0]!.some((t) =>
      t.scopes.includes("keyword.control.declaration.ode"),
    ),
  );
});

test("ODE, map and initial-value left sides name a state", () => {
  assert.equal(scopeOf("x'=-x", "x"), "entity.name.variable.state.ode");
  assert.equal(scopeOf("x' = -x", "x"), "entity.name.variable.state.ode");
  assert.equal(scopeOf("dx/dt=-x", "dx"), "entity.name.variable.state.ode");
  assert.equal(
    scopeOf("x(t+1)=r*x*(1-x)", "x"),
    "entity.name.variable.state.ode",
  );
  assert.equal(scopeOf("x(0)=1", "x"), "entity.name.variable.state.ode");
  assert.equal(scopeOf("x[j]'=-x[j]", "x"), "entity.name.variable.state.ode");
  assert.equal(scopeOf("x[j](0)=1", "x"), "entity.name.variable.state.ode");
});

test("derived parameter and function definition", () => {
  assert.equal(scopeOf("!d=2*a", "d"), "variable.other.derived.ode");
  assert.equal(
    scopeOf("f(u,v)=u*v", "f"),
    "entity.name.function.definition.ode",
  );
  assert.equal(scopeOf("f(u,v)=u*v", "u"), "variable.parameter.function.ode");
  assert.equal(
    scopeOf("f(u,v)=u*v", "v", 0),
    "variable.parameter.function.ode",
  );
});

test("@ options", () => {
  const line = "@ total=100,dt=.05";
  assert.equal(scopeOf(line, "total"), "entity.other.attribute-name.option.ode");
  assert.equal(scopeOf(line, "100"), "constant.numeric.ode");
  assert.equal(scopeOf(line, "dt"), "entity.other.attribute-name.option.ode");
  assert.equal(scopeOf(line, ".05"), "constant.numeric.ode");
  assert.equal(
    scopeOf("@ meth=rungekutta", "rungekutta"),
    "string.unquoted.option-value.ode",
  );
});

test("expressions: constants, time, builtins, operators", () => {
  const src = "x'=sin(x)+pi*t**2-e";
  assert.equal(scopeOf(src, "sin"), "support.function.builtin.ode");
  assert.equal(scopeOf(src, "pi"), "constant.language.pi.ode");
  assert.equal(scopeOf(src, "t"), "variable.language.time.ode");
  assert.equal(scopeOf(src, "**"), "keyword.operator.arithmetic.ode");
  assert.equal(scopeOf(src, "e"), "variable.other.ode");
  assert.equal(scopeOf(src, "2"), "constant.numeric.ode");
});

test("names are not split into keywords", () => {
  assert.equal(scopeOf("x'=api+tau+sinx", "api"), "variable.other.ode");
  assert.equal(scopeOf("x'=api+tau+sinx", "tau"), "variable.other.ode");
  assert.equal(scopeOf("x'=api+tau+sinx", "sinx"), "variable.other.ode");
  assert.equal(scopeOf("x'=a.b+1", "a.b"), "variable.other.ode");
  assert.equal(scopeOf("x'=sinh(x)", "sinh"), "support.function.builtin.ode");
  assert.equal(scopeOf("x'=log10(x)", "log10"), "support.function.builtin.ode");
  assert.equal(scopeOf("x'=atan2(x,y)", "atan2"), "support.function.builtin.ode");
  assert.equal(scopeOf("x'=foo(x)", "foo"), "entity.name.function.call.ode");
  assert.equal(scopeOf("x'=arg1+arg9", "arg1"), "variable.language.argument.ode");
});

test("conditionals and logic", () => {
  const src = "x'=if(x>1)then(1)else(0)";
  assert.equal(scopeOf(src, "if"), "keyword.control.conditional.ode");
  assert.equal(scopeOf(src, "then"), "keyword.control.conditional.ode");
  assert.equal(scopeOf(src, "else"), "keyword.control.conditional.ode");
  assert.equal(scopeOf(src, ">"), "keyword.operator.comparison.ode");
  assert.equal(
    scopeOf("x'=not(a)&b|c", "not"),
    "keyword.operator.logical.word.ode",
  );
  assert.equal(scopeOf("x'=not(a)&b|c", "&"), "keyword.operator.logical.ode");
});

test("number forms", () => {
  for (const n of ["5", ".5", "5.", "1.e2", "1e+2", "2.5e-3"]) {
    assert.equal(scopeOf(`x'=${n}*x`, n), "constant.numeric.ode", n);
  }
});

test("1..10 is 1, .., 10", () => {
  const toks = tokenize("x[1..10]'=-x[1..10]")[0]!.map((t) => t.text);
  assert.ok(toks.includes("1"), JSON.stringify(toks));
  assert.ok(toks.includes(".."), JSON.stringify(toks));
  assert.ok(toks.includes("10"), JSON.stringify(toks));
  assert.ok(!toks.includes("1."), JSON.stringify(toks));
  assert.equal(scopeOf("x[1..10]'=0", ".."), "keyword.operator.range.ode");
});

test("array blocks", () => {
  const lines = tokenize("%[1..3]\nx[j]'=-x[j-1]\n%");
  assert.equal(lines[0]![0]!.text, "%");
  assert.ok(
    lines[0]![0]!.scopes.includes("keyword.control.array.begin.ode"),
  );
  assert.ok(
    lines[2]![0]!.scopes.includes("keyword.control.array.end.ode"),
  );
  assert.equal(scopeOf("x[j+1]'=0", "j"), "variable.language.index.ode");
});

test("every builtin in builtins.rs is highlighted as a builtin", () => {
  const src = fs.readFileSync(
    path.join(root, "..", "..", "crates", "pc-ode", "src", "builtins.rs"),
    "utf8",
  );
  const table = src.slice(0, src.indexOf("#[cfg(test)]"));
  const names = [...table.matchAll(/\bf\("([a-z_0-9]+)"/g)].map((m) => m[1]!);
  assert.ok(names.length >= 30, `found only ${names.length} builtins`);
  for (const name of names) {
    assert.equal(
      scopeOf(`x'=${name}(x)`, name),
      "support.function.builtin.ode",
      name,
    );
  }
});

test("BOM-free ASCII sample tokenizes without throwing", () => {
  const sample = [
    "# Brusselator",
    "par a=1, b=3",
    "x'=a-(b+1)*x+x^2*y",
    "y'=b*x-x^2*y",
    "init x=1,y=1",
    "@ total=50,dt=.01",
    "done",
  ].join("\n");
  const lines = tokenize(sample);
  assert.equal(lines.length, 7);
  assert.ok(hasScope("par a=1, b=3", "par", "keyword.control.declaration.ode"));
});
