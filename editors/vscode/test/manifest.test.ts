import assert from "node:assert/strict";
import { test } from "node:test";
import * as fs from "node:fs";
import * as path from "node:path";

const root = path.resolve(__dirname, "..", "..");
const pkg = JSON.parse(fs.readFileSync(path.join(root, "package.json"), "utf8"));

test("manifest files exist", () => {
  const g = pkg.contributes.grammars[0];
  assert.ok(fs.existsSync(path.join(root, g.path)), g.path);
  const l = pkg.contributes.languages[0];
  assert.ok(fs.existsSync(path.join(root, l.configuration)), l.configuration);
  assert.equal(g.language, l.id);
});

test("grammar scope name matches the grammar file", () => {
  const g = pkg.contributes.grammars[0];
  const body = JSON.parse(fs.readFileSync(path.join(root, g.path), "utf8"));
  assert.equal(body.scopeName, g.scopeName);
});

test("server path setting is machine scoped", () => {
  const prop = pkg.contributes.configuration.properties["phasecont.server.path"];
  assert.equal(prop.scope, "machine");
  assert.equal(prop.default, "");
});

test("trace setting id matches the client id", () => {
  // LanguageClient("phasecont", ...) reads "phasecont.trace.server".
  assert.ok(pkg.contributes.configuration.properties["phasecont.trace.server"]);
  const ext = fs.readFileSync(path.join(root, "src", "extension.ts"), "utf8");
  assert.match(ext, /new LanguageClient\(\s*"phasecont"/);
});

test("restart command is contributed and registered", () => {
  const cmd = pkg.contributes.commands[0].command;
  assert.equal(cmd, "phasecont.restartServer");
  const ext = fs.readFileSync(path.join(root, "src", "extension.ts"), "utf8");
  assert.ok(ext.includes(`"${cmd}"`));
});

test("only approved runtime dependencies", () => {
  assert.deepEqual(Object.keys(pkg.dependencies), ["vscode-languageclient"]);
  for (const [name, version] of Object.entries<string>({
    ...pkg.dependencies,
    ...pkg.devDependencies,
  })) {
    assert.match(version, /^\d+\.\d+\.\d+$/, `${name} is not pinned exactly`);
  }
});

test("language configuration compiles", () => {
  const cfg = JSON.parse(
    fs.readFileSync(path.join(root, "language-configuration.json"), "utf8"),
  );
  const word = new RegExp(cfg.wordPattern, "g");
  assert.deepEqual("a.b_1 + c".match(word), ["a.b_1", "c"]);
  const start = new RegExp(cfg.folding.markers.start);
  const end = new RegExp(cfg.folding.markers.end);
  assert.ok(start.test("%[1..3]"));
  assert.ok(start.test("  %[j]"));
  assert.ok(end.test("%"));
  assert.ok(end.test("  %  "));
  assert.ok(!end.test("%[1..3]"));
});
