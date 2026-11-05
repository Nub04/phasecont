// Finds the pc-lsp executable. This file does not import "vscode", so the
// tests run it under plain node.

import * as path from "node:path";

/** Platform ids that have a bundled server: `<process.platform>-<process.arch>`. */
export const BUNDLED_TARGETS = [
  "linux-x64",
  "win32-x64",
  "darwin-arm64",
  "darwin-x64",
] as const;

export interface ServerEnv {
  /** Folder the extension is installed in. */
  extensionPath: string;
  /** Value of the `phasecont.server.path` setting, "" when unset. */
  configured: string;
  platform: string;
  arch: string;
  /** True when `file` is an existing regular file. */
  isFile: (file: string) => boolean;
}

export type Resolution =
  | { ok: true; command: string; source: "setting" | "bundled" }
  | { ok: false; message: string };

/** Relative path of the bundled server for one platform, or undefined. */
export function bundledRelativePath(
  platform: string,
  arch: string,
): string | undefined {
  const target = `${platform}-${arch}`;
  if (!(BUNDLED_TARGETS as readonly string[]).includes(target)) {
    return undefined;
  }
  const exe = platform === "win32" ? "pc-lsp.exe" : "pc-lsp";
  return path.posix.join("server", target, exe);
}

/**
 * Picks the server to launch. A non-empty setting wins and is never
 * second-guessed: if it does not point at a file the result is an error,
 * not a silent switch to the bundled server.
 */
export function resolveServer(env: ServerEnv): Resolution {
  const configured = env.configured.trim();
  if (configured !== "") {
    const abs = path.isAbsolute(configured) || isWindowsAbsolute(configured);
    if (!abs) {
      return {
        ok: false,
        message: `phasecont.server.path must be an absolute path, got "${configured}".`,
      };
    }
    if (!env.isFile(configured)) {
      return {
        ok: false,
        message: `phasecont.server.path does not point at a file: ${configured}`,
      };
    }
    return { ok: true, command: configured, source: "setting" };
  }

  const rel = bundledRelativePath(env.platform, env.arch);
  if (rel === undefined) {
    return {
      ok: false,
      message:
        `This extension has no bundled language server for ${env.platform}-${env.arch}. ` +
        `Build pc-lsp from the PhaseCont repository and set phasecont.server.path.`,
    };
  }
  const full = path.join(env.extensionPath, ...rel.split("/"));
  if (!env.isFile(full)) {
    return {
      ok: false,
      message: `The bundled language server is missing: ${full}`,
    };
  }
  return { ok: true, command: full, source: "bundled" };
}

function isWindowsAbsolute(p: string): boolean {
  return /^[A-Za-z]:[\\/]/.test(p) || p.startsWith("\\\\");
}
