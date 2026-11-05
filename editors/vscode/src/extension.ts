import * as fs from "node:fs";
import * as vscode from "vscode";
import {
  LanguageClient,
  type LanguageClientOptions,
  type ServerOptions,
} from "vscode-languageclient/node";
import { resolveServer } from "./server";

let client: LanguageClient | undefined;
let output: vscode.OutputChannel | undefined;

function isFile(file: string): boolean {
  try {
    return fs.statSync(file).isFile();
  } catch {
    return false;
  }
}

async function startClient(context: vscode.ExtensionContext): Promise<void> {
  const configured = vscode.workspace
    .getConfiguration("phasecont")
    .get<string>("server.path", "");
  const resolution = resolveServer({
    extensionPath: context.extensionPath,
    configured,
    platform: process.platform,
    arch: process.arch,
    isFile,
  });
  if (!resolution.ok) {
    output?.appendLine(resolution.message);
    void vscode.window.showErrorMessage(`PhaseCont: ${resolution.message}`);
    return;
  }
  output?.appendLine(`Starting ${resolution.command} (${resolution.source})`);

  const serverOptions: ServerOptions = {
    command: resolution.command,
    args: [],
  };
  const clientOptions: LanguageClientOptions = {
    // pc-lsp reads file: URIs only, so unsaved untitled buffers are skipped.
    documentSelector: [{ scheme: "file", language: "ode" }],
    outputChannel: output,
  };
  const next = new LanguageClient(
    "phasecont",
    "PhaseCont",
    serverOptions,
    clientOptions,
  );
  client = next;
  try {
    await next.start();
  } catch (err) {
    client = undefined;
    const text = err instanceof Error ? err.message : String(err);
    output?.appendLine(`The language server did not start: ${text}`);
    void vscode.window.showErrorMessage(
      `PhaseCont: the language server did not start. ${text}`,
    );
  }
}

async function stopClient(): Promise<void> {
  const current = client;
  client = undefined;
  if (current !== undefined) {
    try {
      await current.stop();
    } catch {
      // The server already exited. Nothing is left to stop.
    }
  }
}

export async function activate(
  context: vscode.ExtensionContext,
): Promise<void> {
  output = vscode.window.createOutputChannel("PhaseCont");
  context.subscriptions.push(output);

  const restart = async (): Promise<void> => {
    await stopClient();
    await startClient(context);
  };

  context.subscriptions.push(
    vscode.commands.registerCommand("phasecont.restartServer", restart),
    vscode.workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration("phasecont.server.path")) {
        void restart();
      }
    }),
  );

  await startClient(context);
}

export async function deactivate(): Promise<void> {
  await stopClient();
}
