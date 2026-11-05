import assert from "node:assert/strict";
import { test } from "node:test";
import * as path from "node:path";
import {
  BUNDLED_TARGETS,
  bundledRelativePath,
  resolveServer,
  type ServerEnv,
} from "../src/server";

function env(over: Partial<ServerEnv> & { files?: string[] }): ServerEnv {
  const files = new Set(over.files ?? []);
  return {
    extensionPath: path.join(path.sep, "ext"),
    configured: "",
    platform: "linux",
    arch: "x64",
    isFile: (f) => files.has(f),
    ...over,
  };
}

test("bundled path per target", () => {
  assert.equal(
    bundledRelativePath("linux", "x64"),
    "server/linux-x64/pc-lsp",
  );
  assert.equal(
    bundledRelativePath("win32", "x64"),
    "server/win32-x64/pc-lsp.exe",
  );
  assert.equal(
    bundledRelativePath("darwin", "arm64"),
    "server/darwin-arm64/pc-lsp",
  );
  assert.equal(bundledRelativePath("linux", "arm64"), undefined);
  assert.equal(bundledRelativePath("freebsd", "x64"), undefined);
  for (const target of BUNDLED_TARGETS) {
    const [platform, arch] = target.split("-");
    assert.ok(bundledRelativePath(platform!, arch!), target);
  }
});

test("empty setting uses the bundled server when it exists", () => {
  const bundled = path.join(path.sep, "ext", "server", "linux-x64", "pc-lsp");
  const r = resolveServer(env({ files: [bundled] }));
  assert.deepEqual(r, { ok: true, command: bundled, source: "bundled" });
});

test("a blank setting counts as empty", () => {
  const bundled = path.join(path.sep, "ext", "server", "linux-x64", "pc-lsp");
  const r = resolveServer(env({ configured: "   ", files: [bundled] }));
  assert.equal(r.ok && r.source, "bundled");
});

test("missing bundled server is an error that names the path", () => {
  const r = resolveServer(env({}));
  assert.equal(r.ok, false);
  assert.match(!r.ok ? r.message : "", /bundled language server is missing/);
  assert.match(!r.ok ? r.message : "", /pc-lsp/);
});

test("unsupported platform is an error that points at the setting", () => {
  const r = resolveServer(env({ platform: "linux", arch: "arm64" }));
  assert.equal(r.ok, false);
  assert.match(!r.ok ? r.message : "", /linux-arm64/);
  assert.match(!r.ok ? r.message : "", /phasecont\.server\.path/);
});

test("the setting wins over the bundled server", () => {
  const bundled = path.join(path.sep, "ext", "server", "linux-x64", "pc-lsp");
  const mine = "/opt/pc-lsp";
  const r = resolveServer(env({ configured: mine, files: [bundled, mine] }));
  assert.deepEqual(r, { ok: true, command: mine, source: "setting" });
});

test("a setting that points nowhere does not fall back", () => {
  const bundled = path.join(path.sep, "ext", "server", "linux-x64", "pc-lsp");
  const r = resolveServer(
    env({ configured: "/opt/missing/pc-lsp", files: [bundled] }),
  );
  assert.equal(r.ok, false);
  assert.match(!r.ok ? r.message : "", /does not point at a file/);
});

test("a relative setting is rejected", () => {
  const r = resolveServer(
    env({ configured: "target/debug/pc-lsp", files: ["target/debug/pc-lsp"] }),
  );
  assert.equal(r.ok, false);
  assert.match(!r.ok ? r.message : "", /absolute/);
});

test("a Windows drive path counts as absolute on any host", () => {
  const exe = "C:\\tools\\pc-lsp.exe";
  const r = resolveServer(
    env({ platform: "win32", configured: exe, files: [exe] }),
  );
  assert.deepEqual(r, { ok: true, command: exe, source: "setting" });
});
