/**
 * Minimal local Pi RPC proxy (Phase 1).
 *
 * Lets the local Python project (app/llm_explanation.py) reach the DeepSeek model Pi
 * already has configured, WITHOUT this process ever reading, printing or copying any
 * provider credential: authentication is performed by Pi itself (ModelRuntime resolves
 * ~/.pi/agent/auth.json). This proxy only relays text.
 *
 * Chain: app/llm_explanation.py -> HTTP 127.0.0.1 -> this proxy -> pi --mode rpc
 *        (stdio JSONL, one isolated child process per request) -> DeepSeek
 *
 * Enforced here:
 * - listens on 127.0.0.1 only; only POST /v1/chat/completions and GET /health
 * - random in-memory proxy token (NOT a provider API key), never written to disk
 * - ONE child process per request => no cross-request session contamination
 * - LF-only JSONL framing (never a generic line reader that splits U+2028/U+2029)
 * - "prompt accepted" != "generation finished"; completion is `agent_settled`
 * - per-request timeout kills the child so it cannot keep spending quota
 * - logs never contain Authorization, the token, request bodies or model output
 *
 * Node standard library only. No dependencies.
 */
import http from "node:http";
import crypto from "node:crypto";
import { StringDecoder } from "node:string_decoder";
import { spawn as nodeSpawn } from "node:child_process";
import { existsSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

export const DEFAULT_MODEL = "deepseek-v4-pro";
export const DEFAULT_PROVIDER = "deepseek";
export const DEFAULT_PORT = 8791; // 8787 is used by the local pi-web-ui server
export const DEFAULT_TIMEOUT_MS = 120_000;
export const MAX_BODY_BYTES = 256 * 1024;
export const MAX_MESSAGES = 20;
export const MAX_SYSTEM_PROMPT_BYTES = 8 * 1024;

const FALLBACK_SYSTEM_PROMPT =
  "你是库存补货决策解释助手，只根据用户给出的结构化字段解释已经算好的决策结果，" +
  "不得重新计算或修改任何数量，不得编造缺失事实；证据不足时明确说明无法判断。";

/**
 * Location of the real CLI entry inside the npm package. On Windows `pi` is a `.cmd`
 * shim, so `child_process.spawn("pi", ...)` fails with ENOENT; the proxy therefore always
 * runs this JavaScript entry through `process.execPath` instead of shelling out.
 */
export const PI_CLI_RELATIVE = "node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js";

/**
 * Resolve the Pi CLI entry to an ABSOLUTE path (never relative to the cwd).
 * Order: PI_CLI_PATH -> %APPDATA%\npm -> ~/.pi/agent/npm.
 * Throws a diagnosable error when nothing is found; never falls back to a shell launch.
 */
export function resolvePiCliPath(options = {}) {
  const { env = process.env, exists = existsSync } = options;
  const explicit = env.PI_CLI_PATH;
  if (explicit) {
    if (!exists(explicit)) {
      throw new Error(`PI_CLI_PATH points to a missing file: ${explicit}`);
    }
    return explicit;
  }
  const home = env.USERPROFILE || env.HOME;
  const candidates = [];
  if (env.APPDATA) candidates.push(join(env.APPDATA, "npm", PI_CLI_RELATIVE));
  if (home) candidates.push(join(home, ".pi", "agent", "npm", PI_CLI_RELATIVE));
  for (const candidate of candidates) {
    if (exists(candidate)) return candidate;
  }
  throw new Error(
    "cannot locate the Pi CLI entry (pi-coding-agent/dist/bundle/cli.js); " +
      "set PI_CLI_PATH to its absolute path",
  );
}

export class TimeoutError extends Error {}
export class CancelledError extends Error {}
export class RpcError extends Error {}

/** Pi CLI args. Verified against this machine's `pi --help`. */
export function piArgs({ provider = DEFAULT_PROVIDER, model = DEFAULT_MODEL, systemPrompt = "" } = {}) {
  return [
    "--mode", "rpc",
    "--provider", provider,
    "--model", model,
    "--system-prompt", systemPrompt || FALLBACK_SYSTEM_PROMPT,
    "--no-session",
    "--no-tools",
    "--no-extensions",
    "--no-skills",
    "--no-prompt-templates",
    "--no-context-files",
  ];
}

/** Split a growing string on LF only. Returns { lines, rest }. */
export function splitLines(text) {
  const parts = text.split("\n");
  const rest = parts.pop() ?? "";
  return { lines: parts.map((line) => (line.endsWith("\r") ? line.slice(0, -1) : line)), rest };
}

/**
 * Incremental LF-framed reader that decodes UTF-8 across chunk boundaries.
 * A naive `chunk.toString("utf8")` turns a multi-byte character split across two chunks
 * into U+FFFD and loses the original bytes, so a StringDecoder is used instead.
 */
export function createLineReader(onLine) {
  const decoder = new StringDecoder("utf8");
  let buffer = "";
  const push = (text) => {
    buffer += text;
    const { lines, rest } = splitLines(buffer);
    buffer = rest;
    for (const line of lines) onLine(line);
  };
  return {
    write(chunk) {
      push(decoder.write(chunk));
    },
    end() {
      push(decoder.end());
      if (buffer) {
        const last = buffer; // a trailing line that never received its LF
        buffer = "";
        onLine(last);
      }
    },
  };
}

function extractAssistantText(message) {
  if (!message || typeof message !== "object") return "";
  const content = message.content;
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .filter((part) => part && part.type === "text" && typeof part.text === "string")
      .map((part) => part.text)
      .join("");
  }
  return "";
}

/** Normalize messages[] into (system text, single prompt text). */
export function buildPrompt(messages) {
  const system = [];
  const rest = [];
  for (const message of messages) {
    const content = typeof message.content === "string" ? message.content : "";
    if (message.role === "system") system.push(content);
    else rest.push(`${message.role ?? "user"}: ${content}`);
  }
  return { systemText: system.join("\n\n"), userText: rest.join("\n\n") };
}

/**
 * Run ONE isolated Pi RPC child and resolve with the assistant text.
 * Rejects with TimeoutError / RpcError. Always kills the child afterwards.
 */
export function runRpcOnce({ spawn, piCliPath, systemText, userText, timeoutMs, signal, onChild, onChildDone }) {
  return new Promise((resolve, reject) => {
    let child;
    try {
      // Always launch through the Node binary: `pi` itself is a Windows .cmd shim that
      // spawn() cannot execute, and a shell launch would re-interpret the system prompt.
      const cliPath = piCliPath ?? resolvePiCliPath();
      child = spawn(process.execPath, [cliPath, ...piArgs({ systemPrompt: systemText })], {
        stdio: ["pipe", "pipe", "pipe"],
      });
    } catch (error) {
      reject(new RpcError(`failed to start pi: ${error.message}`));
      return;
    }
    onChild?.(child);

    let settled = false;
    let deltas = "";
    let lastAssistantText = "";
    let lastStopReason = null;
    let promptAccepted = false;
    let lastError = "";

    let abortHandler = null;
    const finish = (fn, value) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (abortHandler && signal) signal.removeEventListener("abort", abortHandler);
      try {
        child.kill();
      } catch {
        /* already gone */
      }
      onChildDone?.(child);
      fn(value);
    };

    const timer = setTimeout(() => {
      finish(reject, new TimeoutError(`pi rpc timed out after ${timeoutMs} ms`));
    }, timeoutMs);
    if (typeof timer.unref === "function") timer.unref();

    // Client went away: stop the child at once instead of letting it finish (and bill).
    abortHandler = () => finish(reject, new CancelledError("client disconnected before completion"));
    if (signal) {
      signal.addEventListener("abort", abortHandler, { once: true });
      if (signal.aborted) abortHandler();
    }

    const send = (command) => {
      if (settled) return;
      try {
        child.stdin.write(`${JSON.stringify(command)}\n`);
      } catch (error) {
        finish(reject, new RpcError(`failed to write to pi stdin: ${error.message}`));
      }
    };

    const handleLine = (line) => {
      if (!line.trim()) return;
      let event;
      try {
        event = JSON.parse(line);
      } catch {
        return; // ignore non-JSON noise on stdout
      }
      const type = event?.type;

      if (type === "response") {
        if (event.success === false) {
          // Fail fast instead of waiting for the whole RPC timeout.
          finish(reject, new RpcError(`rpc command ${event.command ?? "unknown"} failed`));
          return;
        }
        if (event.id === "new_session") {
          send({ id: "prompt", type: "prompt", message: userText });
        } else if (event.id === "prompt") {
          promptAccepted = true; // accepted only; completion comes from agent_settled
        }
        return;
      }
      if (type === "message_update") {
        const deltaEvent = event.assistantMessageEvent;
        if (deltaEvent?.type === "text_delta" && typeof deltaEvent.delta === "string") deltas += deltaEvent.delta;
        return;
      }
      if (type === "message_end") {
        const text = extractAssistantText(event.message);
        if (text && event.message?.role === "assistant") {
          lastAssistantText = text;
          lastStopReason = typeof event.message.stopReason === "string" ? event.message.stopReason : null;
        }
        return;
      }
      if (type === "agent_settled") {
        // agent_settled only means the agent stopped working; stopReason decides whether the
        // generation actually succeeded. Documented values: stop | length | toolUse | error | aborted.
        if (lastStopReason === "error") {
          finish(reject, new RpcError("pi reported a generation error"));
          return;
        }
        if (lastStopReason === "aborted") {
          // Upstream abort: NOT 499, which is reserved for a client-side cancel.
          finish(reject, new RpcError("pi reported the generation was aborted upstream"));
          return;
        }
        if (lastStopReason === "toolUse") {
          finish(reject, new RpcError("pi stopped for a tool call, which this proxy does not execute"));
          return;
        }
        // message_end.message is the documented authoritative content; deltas are the fallback.
        const text = (lastAssistantText || deltas).trim();
        if (!text) {
          finish(reject, new RpcError("pi rpc returned empty text"));
          return;
        }
        const finishReason = lastStopReason === "length" ? "length" : lastStopReason === "stop" ? "stop" : "unknown";
        finish(resolve, { text, finishReason });
        return;
      }
      if (type === "extension_error") lastError = "pi reported an extension error";
    };

    const reader = createLineReader(handleLine);
    child.stdout?.on("data", (chunk) => reader.write(chunk));
    child.stdout?.on("end", () => reader.end());
    child.stdout?.on("error", () => {});
    child.stderr?.on("data", () => {}); // diagnostics only; never logged
    child.on("error", (error) => finish(reject, new RpcError(`pi rpc process error: ${error.message}`)));
    child.on("exit", (code) => {
      if (settled) return;
      const reason = lastError || (promptAccepted ? "pi rpc exited before completion" : "pi rpc exited early");
      finish(reject, new RpcError(`${reason} (exit code ${code})`));
    });

    // fresh child == fresh session; still announced explicitly for clarity
    send({ id: "new_session", type: "new_session" });
  });
}

function json(res, status, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(body),
  });
  res.end(body);
}

function errorPayload(message, type, code = null) {
  return { error: { message, type, code } };
}

function readBody(req, maxBytes) {
  return new Promise((resolve, reject) => {
    const chunks = [];
    let size = 0;
    let tooLarge = false;
    let failed = false;
    // Keep draining the stream: answering before the client finished uploading can
    // reset the connection, so the 413 is raised on "end" instead.
    req.on("data", (chunk) => {
      size += chunk.length;
      if (size > maxBytes) {
        tooLarge = true;
        chunks.length = 0;
        return;
      }
      chunks.push(chunk);
    });
    req.on("end", () => {
      if (tooLarge) reject(Object.assign(new Error("request body too large"), { status: 413 }));
      else resolve(Buffer.concat(chunks).toString("utf8"));
    });
    req.on("error", (error) => {
      if (failed) return;
      failed = true;
      reject(Object.assign(error, { status: 400 }));
    });
  });
}

/** Validate a chat-completions body; returns { model, messages } or throws { status }. */
export function validateBody(raw) {
  let parsed;
  try {
    parsed = JSON.parse(raw);
  } catch {
    throw Object.assign(new Error("request body is not valid JSON"), { status: 400 });
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
    throw Object.assign(new Error("request body must be a JSON object"), { status: 400 });
  }
  if (parsed.model !== undefined && typeof parsed.model !== "string") {
    throw Object.assign(new Error("model must be a string"), { status: 400 });
  }
  const { messages } = parsed;
  if (!Array.isArray(messages) || messages.length === 0) {
    throw Object.assign(new Error("messages must be a non-empty array"), { status: 400 });
  }
  if (messages.length > MAX_MESSAGES) {
    throw Object.assign(new Error(`messages must contain at most ${MAX_MESSAGES} items`), { status: 400 });
  }
  for (const message of messages) {
    if (!message || typeof message !== "object") {
      throw Object.assign(new Error("each message must be an object"), { status: 400 });
    }
    if (message.role !== "system" && message.role !== "user" && message.role !== "assistant") {
      throw Object.assign(new Error("message.role must be system, user or assistant"), { status: 400 });
    }
    if (typeof message.content !== "string") {
      throw Object.assign(new Error("message.content must be a string"), { status: 400 });
    }
  }
  return { model: parsed.model, messages };
}

export function createProxy(options = {}) {
  const {
    token = crypto.randomBytes(32).toString("hex"),
    spawn = nodeSpawn,
    piCliPath,
    timeoutMs = DEFAULT_TIMEOUT_MS,
    maxBodyBytes = MAX_BODY_BYTES,
    log = (line) => process.stderr.write(`${line}\n`),
  } = options;

  const live = new Set();
  const rpcRunner = options.rpcRunner ?? runRpcOnce;

  const server = http.createServer((req, res) => {
    const started = Date.now();
    const path = (req.url ?? "").split("?")[0];
    // Never log a client-controlled path or method verbatim.
    const route = path === "/v1/chat/completions" ? "chat/completions" : path === "/health" ? "health" : "other";
    const method = /^[A-Z]{3,10}$/.test(req.method ?? "") ? req.method : "OTHER";

    if (req.method === "GET" && path === "/health") {
      json(res, 200, { status: "ok" }); // never touches the model
      return;
    }
    // Authentication runs before the route and method checks: otherwise the 404/405
    // replies tell an unauthenticated caller which routes exist. /health is exempt.
    const header = req.headers["authorization"] ?? "";
    const presented = header.startsWith("Bearer ") ? header.slice(7) : "";
    let authorized = false;
    try {
      authorized =
        presented.length === token.length && crypto.timingSafeEqual(Buffer.from(presented), Buffer.from(token));
    } catch {
      authorized = false;
    }
    if (!authorized) {
      json(res, 401, errorPayload("invalid or missing proxy token", "authentication_error"));
      log(`[proxy] ${method} ${route} status=401 ms=${Date.now() - started}`);
      return;
    }

    if (path !== "/v1/chat/completions") {
      json(res, 404, errorPayload("not found", "invalid_request_error"));
      log(`[proxy] ${method} ${route} status=404 ms=${Date.now() - started}`);
      return;
    }
    if (req.method !== "POST") {
      json(res, 405, errorPayload("method not allowed", "invalid_request_error"));
      log(`[proxy] ${method} ${route} status=405 ms=${Date.now() - started}`);
      return;
    }

    // If the client goes away the Pi child must be stopped, not left generating.
    const controller = new AbortController();
    res.on("close", () => {
      if (!res.writableEnded) controller.abort();
    });

    readBody(req, maxBodyBytes)
      .then((raw) => {
        const { messages } = validateBody(raw);
        const { systemText, userText } = buildPrompt(messages);
        // The system prompt is passed as an argv element, so the OS command-line
        // limit applies: reject it here instead of relying on a spawn failure.
        if (Buffer.byteLength(systemText, "utf8") > MAX_SYSTEM_PROMPT_BYTES) {
          throw Object.assign(new Error(`system prompt exceeds ${MAX_SYSTEM_PROMPT_BYTES} bytes`), {
            status: 400,
          });
        }
        return rpcRunner({
          spawn,
          piCliPath,
          systemText,
          userText,
          timeoutMs,
          signal: controller.signal,
          onChild: (child) => live.add(child),
          onChildDone: (child) => live.delete(child),
        });
      })
      .then(({ text, finishReason }) => {
        if (res.destroyed) {
          log(`[proxy] POST ${route} status=499 ms=${Date.now() - started} reason=cancelled`);
          return;
        }
        try {
          json(res, 200, {
            id: `chatcmpl-${crypto.randomUUID()}`,
            object: "chat.completion",
            created: Math.floor(Date.now() / 1000),
            model: DEFAULT_MODEL,
            choices: [{ index: 0, message: { role: "assistant", content: text }, finish_reason: finishReason }],
          });
        } catch {
          /* socket already gone */
        }
        log(`[proxy] POST ${route} status=200 ms=${Date.now() - started} finish=${finishReason}`);
      })
      .catch((error) => {
        const cancelled = error instanceof CancelledError;
        const timedOut = error instanceof TimeoutError;
        const status = cancelled ? 499 : timedOut ? 504 : error?.status ?? 502;
        const type = cancelled
          ? "cancelled"
          : timedOut
            ? "timeout_error"
            : status >= 500
              ? "api_error"
              : "invalid_request_error";
        if (!cancelled) {
          try {
            json(res, status, errorPayload(error?.message ?? "proxy failure", type));
          } catch {
            /* socket already gone */
          }
        }
        log(
          `[proxy] POST ${route} status=${status} ms=${Date.now() - started} reason=${
            cancelled ? "cancelled" : timedOut ? "timeout" : "error"
          }`,
        );
      });
  });

  function close() {
    for (const child of live) {
      try {
        child.kill();
      } catch {
        /* ignore */
      }
    }
    live.clear();
    return new Promise((resolve) => {
      // Idle keep-alive sockets would otherwise keep server.close() pending forever.
      if (typeof server.closeAllConnections === "function") server.closeAllConnections();
      server.close(() => resolve());
    });
  }

  return { server, token, close, live, rpcRunner };
}

/** Listen on 127.0.0.1; if the requested port is taken, fall back to an ephemeral one. */
export function listenOn(server, port) {
  const attempt = (candidate) =>
    new Promise((resolve, reject) => {
      const onError = (error) => {
        server.removeListener("listening", onListening);
        reject(error);
      };
      const onListening = () => {
        server.removeListener("error", onError);
        resolve();
      };
      server.once("error", onError);
      server.once("listening", onListening);
      server.listen(candidate, "127.0.0.1");
    });
  return attempt(port).catch((error) => {
    if (error?.code !== "EADDRINUSE") throw error;
    return attempt(0);
  });
}

/**
 * Decide the startup token and whether the banner may print it.
 * Exported so the security-relevant parsing is covered by tests.
 */
export function resolveStartupToken(env = process.env) {
  const explicit = env.PI_LLM_PROXY_TOKEN;
  if (explicit !== undefined && explicit.length < 16) {
    throw new Error("PI_LLM_PROXY_TOKEN must be at least 16 characters");
  }
  const showToken = !["0", "false", "no"].includes(
    String(env.PI_LLM_PROXY_SHOW_TOKEN ?? "1").trim().toLowerCase(),
  );
  return { token: explicit || undefined, showToken };
}

async function main() {
  const requested = Number(process.env.PI_LLM_PROXY_PORT ?? DEFAULT_PORT);
  let piCliPath;
  try {
    piCliPath = resolvePiCliPath(); // fail fast, with a diagnosable message
  } catch (error) {
    process.stderr.write("[proxy] " + error.message + "\n");
    process.exit(1);
  }
  let startup;
  try {
    startup = resolveStartupToken();
  } catch (error) {
    process.stderr.write("[proxy] " + error.message + "\n");
    process.exit(1);
  }
  const showToken = startup.showToken;
  const { server, token, close } = createProxy({
    piCliPath,
    ...(startup.token ? { token: startup.token } : {}),
  });
  await listenOn(server, requested);
  const address = server.address();
  process.stderr.write(
    [
      "",
      "Pi RPC LLM proxy (local only)",
      `  endpoint : http://127.0.0.1:${address.port}/v1${
        address.port === requested ? "" : `  (requested ${requested} was busy; fell back to an ephemeral port)`
      }`,
      `  model    : ${DEFAULT_PROVIDER}/${DEFAULT_MODEL} (via Pi; credentials resolved by Pi)`,
      `  pi entry : ${piCliPath}`,
      showToken ? `  token    : ${token}` : "  token    : hidden (PI_LLM_PROXY_SHOW_TOKEN=0)",
      "",
      showToken
        ? "  SENSITIVE CREDENTIAL: shown once, in-memory only. Do NOT redirect stderr"
        : "  Token display is disabled; clients need the value passed via PI_LLM_PROXY_TOKEN.",
      showToken ? "  to a persisted or uploaded file." : "  Never write it to a log, issue or repository.",
      "  It is NOT a DeepSeek API key: Pi keeps its own provider credentials.",
      "  On the Python side set OPENAI_API_KEY to this token and:",
      `  SCOS_LLM_BASE_URL=http://127.0.0.1:${address.port}/v1, SCOS_LLM_MODEL=${DEFAULT_MODEL}.`,
      "  Ctrl+C to stop.",
      "",
    ].join("\n"),
  );

  const shutdown = () => {
    close().finally(() => process.exit(0));
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

const isMain = process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href;
if (isMain) {
  main().catch((error) => {
    process.stderr.write(`[proxy] failed to start: ${error.message}\n`);
    process.exit(1);
  });
}
