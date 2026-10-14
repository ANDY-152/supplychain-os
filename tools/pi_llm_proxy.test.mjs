/**
 * Phase 1 tests for tools/pi_llm_proxy.mjs.
 *
 * Everything here uses a FAKE RPC child process: no `pi` process is spawned, no
 * DeepSeek request is made, no quota is consumed.
 *
 * Requests go through node:http directly (not global fetch) so that a developer
 * machine with HTTP_PROXY set does not route 127.0.0.1 traffic through a proxy.
 */
import { test } from "node:test";
import assert from "node:assert/strict";
import http from "node:http";
import { EventEmitter } from "node:events";
import { spawn as nodeSpawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import {
  DEFAULT_TIMEOUT_MS,
  MAX_SYSTEM_PROMPT_BYTES,
  PI_CLI_RELATIVE,
  createLineReader,
  createProxy,
  listenOn,
  piArgs,
  resolvePiCliPath,
  resolveStartupToken,
  splitLines,
} from "./pi_llm_proxy.mjs";

const TOKEN = "a".repeat(64);
const BODY = {
  model: "deepseek-v4-pro",
  messages: [
    { role: "system", content: "SYSTEM-PROMPT-TEXT" },
    { role: "user", content: "SKU001 facts here" },
  ],
  temperature: 0,
};

/** A fake child that behaves like `pi --mode rpc` over LF-framed JSONL. */
function makeFakeChild({ onCommand }) {
  const stdout = new EventEmitter();
  const stderr = new EventEmitter();
  const child = new EventEmitter();
  child.stdout = stdout;
  child.stderr = stderr;
  child.killed = false;
  child.kill = () => {
    if (child.killed) return true;
    child.killed = true;
    setImmediate(() => child.emit("exit", null, "SIGTERM")); // mirror a real process
    return true;
  };
  const emit = (obj) => stdout.emit("data", Buffer.from(`${JSON.stringify(obj)}\n`));
  child.emitLine = emit; // lets tests inject extra protocol events
  child.sent = [];
  child.stdin = {
    write(line) {
      const command = JSON.parse(line);
      child.sent.push(command.type);
      onCommand?.(command, { emit, child, raw: line });
    },
  };
  return child;
}

/** Fake spawn returning a fresh child per call (mirrors one-child-per-request). */
function fakeSpawn(makeChild) {
  const children = [];
  const calls = [];
  const spawn = (command, args, options) => {
    calls.push({ command, args, options });
    const child = makeChild(children.length);
    children.push(child);
    return child;
  };
  return { spawn, children, calls };
}

function okBehavior() {
  return () =>
    makeFakeChild({
      onCommand(command, { emit }) {
        if (command.type === "new_session") {
          emit({ id: command.id, type: "response", command: "new_session", success: true });
        } else if (command.type === "prompt") {
          emit({ id: command.id, type: "response", command: "prompt", success: true });
          emit({
            type: "message_update",
            assistantMessageEvent: { type: "text_delta", contentIndex: 0, delta: `ECHO[${command.message}]` },
          });
          emit({ type: "agent_settled" });
        }
      },
    });
}

const FAKE_CLI = "C:/fake/pi-cli.js";

async function startProxy(options = {}) {
  const logs = [];
  // piCliPath is always injected here: tests must never resolve or spawn a real Pi process
  const proxy = createProxy({ token: TOKEN, piCliPath: FAKE_CLI, log: (line) => logs.push(line), ...options });
  await new Promise((resolve) => proxy.server.listen(0, "127.0.0.1", resolve));
  const { port } = proxy.server.address();
  return {
    proxy,
    logs,
    port,
    url: `http://127.0.0.1:${port}`,
    stop: () => proxy.close(),
  };
}

/** Direct HTTP call (bypasses any system HTTP proxy). */
function httpRequest(url, { method = "GET", token, body } = {}) {
  return new Promise((resolve, reject) => {
    const target = new URL(url);
    const payload = body === undefined ? null : typeof body === "string" ? body : JSON.stringify(body);
    const headers = {};
    if (payload !== null) {
      headers["content-type"] = "application/json";
      headers["content-length"] = Buffer.byteLength(payload);
    }
    if (token) headers.authorization = `Bearer ${token}`;
    const req = http.request(
      {
        host: target.hostname,
        port: target.port,
        path: target.pathname + target.search,
        method,
        headers,
        // Explicit agent: this machine sets NODE_USE_ENV_PROXY=1, which would
        // otherwise route 127.0.0.1 traffic through the local proxy and break tests.
        agent: new http.Agent({ keepAlive: false }),
      },
      (res) => {
        const chunks = [];
        res.on("data", (chunk) => chunks.push(chunk));
        res.on("end", () => {
          const text = Buffer.concat(chunks).toString("utf8");
          let parsed = null;
          try {
            parsed = JSON.parse(text);
          } catch {
            /* non-JSON body */
          }
          resolve({ status: res.statusCode, text, json: parsed });
        });
      },
    );
    req.on("error", reject);
    if (payload !== null) req.write(payload);
    req.end();
  });
}

const post = (ctx, token = TOKEN, body = BODY) =>
  httpRequest(`${ctx.url}/v1/chat/completions`, { method: "POST", token, body });

test("1. valid token + request returns the simulated explanation text", async () => {
  const ctx = await startProxy(fakeSpawn(okBehavior()));
  try {
    const res = await post(ctx);
    assert.equal(res.status, 200);
    assert.equal(res.json.object, "chat.completion");
    assert.equal(res.json.choices[0].message.role, "assistant");
    assert.equal(res.json.choices[0].message.content, "ECHO[user: SKU001 facts here]");
    assert.equal(res.json.model, "deepseek-v4-pro");
  } finally {
    await ctx.stop();
  }
});

test("2. missing or invalid token returns 401 and never spawns", async () => {
  const fake = fakeSpawn(okBehavior());
  const ctx = await startProxy(fake);
  try {
    const missing = await httpRequest(`${ctx.url}/v1/chat/completions`, { method: "POST", body: BODY });
    assert.equal(missing.status, 401);
    const wrong = await post(ctx, "b".repeat(64));
    assert.equal(wrong.status, 401);
    assert.equal(wrong.json.error.type, "authentication_error");
    const short = await post(ctx, "short");
    assert.equal(short.status, 401);
    assert.equal(fake.children.length, 0, "no RPC child may start for unauthorized calls");
  } finally {
    await ctx.stop();
  }
});

test("3. malformed bodies return 4xx without spawning a child", async () => {
  const fake = fakeSpawn(okBehavior());
  const ctx = await startProxy(fake);
  try {
    assert.equal((await post(ctx, TOKEN, "{not json")).status, 400);
    assert.equal((await post(ctx, TOKEN, { messages: [] })).status, 400);
    assert.equal((await post(ctx, TOKEN, { messages: "nope" })).status, 400);
    assert.equal((await post(ctx, TOKEN, { messages: [{ role: "user", content: 42 }] })).status, 400);
    assert.equal((await post(ctx, TOKEN, { messages: [{ role: "robot", content: "x" }] })).status, 400);
    assert.equal((await post(ctx, TOKEN, { model: 7, messages: [{ role: "user", content: "x" }] })).status, 400);
    const many = { messages: Array.from({ length: 21 }, () => ({ role: "user", content: "x" })) };
    assert.equal((await post(ctx, TOKEN, many)).status, 400);
    const huge = { messages: [{ role: "user", content: "x".repeat(300 * 1024) }] };
    assert.equal((await post(ctx, TOKEN, huge)).status, 413);
    assert.equal(fake.children.length, 0);
  } finally {
    await ctx.stop();
  }
});

test("4. RPC timeout returns 504 and kills the child", async () => {
  const fake = fakeSpawn(() => makeFakeChild({ onCommand() {} })); // never answers
  const ctx = await startProxy({ ...fake, timeoutMs: 120 });
  try {
    const res = await post(ctx);
    assert.equal(res.status, 504);
    assert.equal(res.json.error.type, "timeout_error");
    assert.equal(fake.children.length, 1);
    assert.equal(fake.children[0].killed, true, "timed-out child must be killed");
    assert.equal(ctx.proxy.live.size, 0, "no child may stay registered after timeout");
  } finally {
    await ctx.stop();
  }
});

test("5. abnormal exit, bad JSON noise and empty text degrade to 502", async () => {
  const exitFake = fakeSpawn(() =>
    makeFakeChild({
      onCommand(command, { child }) {
        if (command.type === "new_session") child.emit("exit", 1);
      },
    }),
  );
  const ctxA = await startProxy(exitFake);
  try {
    assert.equal((await post(ctxA)).status, 502);
  } finally {
    await ctxA.stop();
  }

  const noiseFake = fakeSpawn(() =>
    makeFakeChild({
      onCommand(command, { emit, child }) {
        if (command.type === "new_session") {
          emit({ id: command.id, type: "response", command: "new_session", success: true });
        }
        if (command.type === "prompt") {
          child.stdout.emit("data", Buffer.from("<html>not json</html>\n"));
          emit({ type: "agent_settled" });
        }
      },
    }),
  );
  const ctxB = await startProxy(noiseFake);
  try {
    assert.equal((await post(ctxB)).status, 502);
  } finally {
    await ctxB.stop();
  }

  const emptyFake = fakeSpawn(() =>
    makeFakeChild({
      onCommand(command, { emit }) {
        if (command.type === "new_session") {
          emit({ id: command.id, type: "response", command: "new_session", success: true });
        }
        if (command.type === "prompt") {
          emit({ id: command.id, type: "response", command: "prompt", success: true });
          emit({ type: "agent_settled" });
        }
      },
    }),
  );
  const ctxC = await startProxy(emptyFake);
  try {
    const res = await post(ctxC);
    assert.equal(res.status, 502);
    assert.equal(res.json.error.type, "api_error");
  } finally {
    await ctxC.stop();
  }
});

test("6. concurrent requests keep separate sessions and responses", async () => {
  const fake = fakeSpawn(okBehavior());
  const ctx = await startProxy(fake);
  try {
    const bodies = ["alpha", "beta", "gamma"].map((tag) => ({
      messages: [{ role: "user", content: `TAG-${tag}` }],
    }));
    const results = await Promise.all(bodies.map((b) => post(ctx, TOKEN, b)));
    const texts = results.map((r) => r.json.choices[0].message.content);
    assert.deepEqual(texts, ["ECHO[user: TAG-alpha]", "ECHO[user: TAG-beta]", "ECHO[user: TAG-gamma]"]);
    assert.equal(fake.children.length, 3, "one isolated child per request");
  } finally {
    await ctx.stop();
  }
});

test("7. logs never contain the token, Authorization header or request body", async () => {
  const ctx = await startProxy(fakeSpawn(okBehavior()));
  try {
    await post(ctx);
    const joined = ctx.logs.join("\n");
    assert.ok(!joined.includes(TOKEN), "token must not appear in logs");
    assert.ok(!/bearer/i.test(joined), "Authorization must not appear in logs");
    assert.ok(!joined.includes("SKU001 facts here"), "request body must not appear in logs");
    assert.ok(!joined.includes("SYSTEM-PROMPT-TEXT"), "system prompt must not appear in logs");
    assert.ok(!joined.includes("ECHO["), "model output must not appear in logs");
    assert.ok(ctx.logs.some((line) => line.includes("status=200")), "a status line should be logged");
  } finally {
    await ctx.stop();
  }
});

test("8. the server binds 127.0.0.1 only", async () => {
  const ctx = await startProxy(fakeSpawn(okBehavior()));
  try {
    assert.equal(ctx.proxy.server.address().address, "127.0.0.1");
    assert.equal(ctx.proxy.server.address().family, "IPv4");
  } finally {
    await ctx.stop();
  }
});

test("9. GET /health answers without spawning any RPC child", async () => {
  const fake = fakeSpawn(okBehavior());
  const ctx = await startProxy(fake);
  try {
    const res = await httpRequest(`${ctx.url}/health`);
    assert.equal(res.status, 200);
    assert.deepEqual(res.json, { status: "ok" });
    assert.equal(fake.children.length, 0);
  } finally {
    await ctx.stop();
  }
});

test("10. stopping the proxy kills live child processes", async () => {
  const fake = fakeSpawn(() => makeFakeChild({ onCommand() {} })); // hangs
  const ctx = await startProxy({ ...fake, timeoutMs: 30_000 });
  const pending = post(ctx).catch(() => null);
  await new Promise((resolve) => setTimeout(resolve, 60));
  const stopped = ctx.proxy.close();
  assert.equal(fake.children[0].killed, true, "close() must kill the in-flight child");
  assert.equal(ctx.proxy.live.size, 0);
  await pending;
  await stopped;
});

test("11. U+2028 / U+2029 inside a JSON string do not split frames", async () => {
  const weird = "line\u2028sep\u2029end";
  const fake = fakeSpawn(() =>
    makeFakeChild({
      onCommand(command, { emit }) {
        if (command.type === "new_session") {
          emit({ id: command.id, type: "response", command: "new_session", success: true });
        }
        if (command.type === "prompt") {
          emit({ id: command.id, type: "response", command: "prompt", success: true });
          emit({ type: "message_update", assistantMessageEvent: { type: "text_delta", delta: weird } });
          emit({ type: "agent_settled" });
        }
      },
    }),
  );
  const ctx = await startProxy(fake);
  try {
    const res = await post(ctx);
    assert.equal(res.json.choices[0].message.content, weird);
  } finally {
    await ctx.stop();
  }
});

test("12. unauthenticated probes get 401; authorized unknown routes get 404/405", async () => {
  const fake = fakeSpawn(okBehavior());
  const ctx = await startProxy(fake);
  try {
    // authentication precedes routing, so an unauthenticated caller cannot probe routes
    assert.equal((await httpRequest(`${ctx.url}/v1/chat/completions`)).status, 401);
    assert.equal((await httpRequest(`${ctx.url}/v1/models`)).status, 401);
    // with a valid token the real status codes apply again
    assert.equal((await httpRequest(`${ctx.url}/v1/chat/completions`, { token: TOKEN })).status, 405);
    assert.equal((await httpRequest(`${ctx.url}/v1/models`, { token: TOKEN })).status, 404);
    assert.equal(fake.children.length, 0);
  } finally {
    await ctx.stop();
  }
});

test("13. listenOn falls back to an ephemeral port when the requested port is busy", async () => {
  const a = createProxy({ token: TOKEN, spawn: fakeSpawn(okBehavior()).spawn, log: () => {} });
  const b = createProxy({ token: TOKEN, spawn: fakeSpawn(okBehavior()).spawn, log: () => {} });
  try {
    await listenOn(a.server, 0);
    const taken = a.server.address().port;
    await listenOn(b.server, taken);
    assert.notEqual(b.server.address().port, taken);
    assert.equal(b.server.address().address, "127.0.0.1");
  } finally {
    await a.close();
    await b.close();
  }
});

// --- CLI entry resolution (Windows .cmd shim fix) -------------------------

test("14. resolvePiCliPath prefers PI_CLI_PATH when the file exists", () => {
  const explicit = "C:/custom/pi-cli.js";
  const resolved = resolvePiCliPath({ env: { PI_CLI_PATH: explicit }, exists: (p) => p === explicit });
  assert.equal(resolved, explicit);
});

test("15. resolvePiCliPath rejects a missing PI_CLI_PATH with a diagnosable error", () => {
  assert.throws(
    () => resolvePiCliPath({ env: { PI_CLI_PATH: "C:/nope/cli.js" }, exists: () => false }),
    /PI_CLI_PATH points to a missing file/,
  );
});

test("16. resolvePiCliPath falls back to absolute install locations (cwd independent)", () => {
  const appData = "C:/Users/x/AppData/Roaming";
  const viaAppData = join(appData, "npm", PI_CLI_RELATIVE);
  assert.equal(
    resolvePiCliPath({ env: { APPDATA: appData }, exists: (p) => p === viaAppData }),
    viaAppData,
  );

  const home = "C:/Users/x";
  const viaHome = join(home, ".pi", "agent", "npm", PI_CLI_RELATIVE);
  assert.equal(
    resolvePiCliPath({ env: { USERPROFILE: home }, exists: (p) => p === viaHome }),
    viaHome,
  );
  assert.ok(viaHome.includes("pi-coding-agent"), "resolved path points at the pi package");
  assert.ok(!viaHome.startsWith("."), "resolved path is absolute, not cwd relative");
});

test("17. resolvePiCliPath reports a clear error when nothing is found", () => {
  assert.throws(() => resolvePiCliPath({ env: {}, exists: () => false }), /PI_CLI_PATH/);
});

test("18. the proxy launches the CLI via process.execPath and never uses a shell", async () => {
  const fake = fakeSpawn(okBehavior());
  const ctx = await startProxy(fake);
  try {
    const res = await post(ctx);
    assert.equal(res.status, 200);
    assert.equal(fake.calls.length, 1);
    const call = fake.calls[0];
    assert.equal(call.command, process.execPath);
    assert.equal(call.args[0], FAKE_CLI);
    assert.ok(call.args.includes("--mode") && call.args.includes("rpc"));
    assert.ok(call.args.includes("--no-tools"));
    assert.ok(!call.options || !call.options.shell, "must not use shell: true");
  } finally {
    await ctx.stop();
  }
});

// --- opt-in real Pi RPC handshake (no prompt, no model call) --------------

test("19. real pi RPC handshake sends only new_session", { skip: process.env.PI_RPC_SMOKE !== "1" }, async () => {
  const cliPath = resolvePiCliPath();
  assert.ok(existsSync(cliPath), `resolved CLI entry must exist: ${cliPath}`);

  const child = nodeSpawn(process.execPath, [cliPath, ...piArgs({ systemPrompt: "PRECHECK" })], {
    stdio: ["pipe", "pipe", "pipe"],
    env: { ...process.env, PI_OFFLINE: "1" },
  });
  const seen = [];
  let buf = "";
  let settled = false;

  const outcome = await new Promise((resolve) => {
    const timer = setTimeout(() => {
      if (!settled) {
        settled = true;
        try {
          child.kill();
        } catch {
          /* gone */
        }
        resolve({ timeout: true });
      }
    }, 20000);
    child.stdout.on("data", (chunk) => {
      buf += chunk.toString("utf8");
      const { lines, rest } = splitLines(buf);
      buf = rest;
      for (const line of lines) {
        if (!line.trim()) continue;
        let event;
        try {
          event = JSON.parse(line);
        } catch {
          seen.push({ type: "NON_JSON" });
          continue;
        }
        seen.push({ type: event.type, command: event.command, success: event.success, id: event.id ?? null });
        if (event.type === "response" && event.command === "new_session") {
          child.stdin.end(); // closing stdin must lead to a clean exit
        }
      }
    });
    child.stderr.on("data", () => {}); // diagnostics only
    child.on("exit", (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve({ code });
    });
    child.on("error", (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      resolve({ spawnError: error.message });
    });
    child.stdin.write(`${JSON.stringify({ id: "new_session", type: "new_session" })}\n`);
  });

  assert.ok(!outcome.timeout, "real handshake must not time out");
  assert.ok(!outcome.spawnError, `spawn must succeed: ${outcome.spawnError}`);
  assert.equal(outcome.code, 0, "pi must exit cleanly after stdin closes");
  assert.deepEqual(seen, [{ type: "response", command: "new_session", success: true, id: "new_session" }]);
  // Only one frame, and no agent/turn/message events => no inference was triggered.
});

// --- phase 2.1: frozen configuration and edge cases -----------------------

test("20. proxy default RPC timeout is 120s and stays above the Python default", () => {
  assert.equal(DEFAULT_TIMEOUT_MS, 120_000);
  const py = readFileSync(new URL("../app/llm_explanation.py", import.meta.url), "utf8");
  const match = py.match(/timeout:\s*float\s*=\s*(\d+)/);
  assert.ok(match, "Python client must declare a numeric default timeout");
  const pyDefault = Number(match[1]) * 1000;
  assert.equal(pyDefault, 90_000, "Python default request timeout must be 90s");
  assert.ok(pyDefault < DEFAULT_TIMEOUT_MS, "client must give up before the proxy does");
});

test("21. a Chinese multi-line system prompt is one argv element (no shell)", async () => {
  const fake = fakeSpawn(okBehavior());
  const ctx = await startProxy(fake);
  const NL = String.fromCharCode(10);
  const systemText = ["你是库存补货决策解释助手。", "只根据输入字段解释，不得修改数量。", "证据不足时说明无法判断。"].join(NL);
  try {
    const res = await httpRequest(`${ctx.url}/v1/chat/completions`, {
      method: "POST",
      token: TOKEN,
      body: { model: "deepseek-v4-pro", messages: [{ role: "system", content: systemText }, { role: "user", content: "x" }] },
    });
    assert.equal(res.status, 200);
    const call = fake.calls[0];
    assert.equal(call.command, process.execPath);
    const idx = call.args.indexOf("--system-prompt");
    assert.ok(idx >= 0, "--system-prompt must be present");
    assert.equal(call.args[idx + 1], systemText, "system prompt must survive byte-for-byte as one argv element");
    assert.equal(call.args.filter((a) => a === systemText).length, 1);
    assert.ok(!call.options || !call.options.shell, "must not go through a shell");
  } finally {
    await ctx.stop();
  }
});

test("22. non-ASCII model output is decoded as UTF-8", async () => {
  const chinese = "当前库存偏低，建议按系统给出的数量补货；数据质量需人工确认。✅";
  const fake = fakeSpawn(() =>
    makeFakeChild({
      onCommand(command, { emit }) {
        if (command.type === "new_session") {
          emit({ id: command.id, type: "response", command: "new_session", success: true });
        }
        if (command.type === "prompt") {
          emit({ id: command.id, type: "response", command: "prompt", success: true });
          emit({ type: "message_update", assistantMessageEvent: { type: "text_delta", delta: chinese } });
          emit({ type: "agent_settled" });
        }
      },
    }),
  );
  const ctx = await startProxy(fake);
  try {
    const res = await post(ctx);
    assert.equal(res.status, 200);
    assert.equal(res.json.choices[0].message.content, chinese);
  } finally {
    await ctx.stop();
  }
});

test("23. client disconnect aborts the request and kills the child", async () => {
  const fake = fakeSpawn(() => makeFakeChild({ onCommand() {} })); // never answers
  const ctx = await startProxy({ ...fake, timeoutMs: 30_000 });
  try {
    const target = new URL(`${ctx.url}/v1/chat/completions`);
    const payload = JSON.stringify({ messages: [{ role: "user", content: "x" }] });
    const req = http.request(
      {
        host: target.hostname,
        port: target.port,
        path: target.pathname,
        method: "POST",
        headers: {
          "content-type": "application/json",
          "content-length": Buffer.byteLength(payload),
          authorization: `Bearer ${TOKEN}`,
        },
        agent: new http.Agent({ keepAlive: false }),
      },
    );
    req.on("error", () => {}); // the socket is destroyed on purpose
    req.write(payload);
    req.end();
    await new Promise((resolve) => setTimeout(resolve, 80)); // let the proxy spawn the child
    assert.equal(fake.children.length, 1);
    assert.equal(fake.children[0].killed, false, "child is running while the request is in flight");

    req.destroy(); // client goes away
    await new Promise((resolve) => setTimeout(resolve, 150));

    assert.equal(fake.children[0].killed, true, "child must be killed when the client disconnects");
    assert.equal(ctx.proxy.live.size, 0, "no child may stay registered after a disconnect");
  } finally {
    await ctx.stop();
  }
});

test("24. the proxy reads no provider credentials (env access is allowlisted)", () => {
  const src = readFileSync(new URL("./pi_llm_proxy.mjs", import.meta.url), "utf8");
  const envKeys = [...src.matchAll(/(?<![.A-Za-z0-9_])env\.([A-Z_]{2,})/g)].map((m) => m[1]);
  assert.deepEqual([...new Set(envKeys)].sort(), [
    "APPDATA",
    "HOME",
    "PI_CLI_PATH",
    "PI_LLM_PROXY_SHOW_TOKEN",
    "PI_LLM_PROXY_TOKEN",
    "USERPROFILE",
  ]);
  const procEnv = [...src.matchAll(/process\.env\.([A-Z_]{2,})/g)].map((m) => m[1]);
  assert.deepEqual(
    [...new Set(procEnv)].sort(),
    ["PI_LLM_PROXY_PORT"],
    "only proxy settings may be read straight off process.env",
  );
  assert.ok(!/readFileSync|readFile\(/.test(src), "the proxy must not read credential files");
  assert.ok(!/auth\.json/.test(src.replace(/^\s*\*.*$/gm, "")), "auth.json may only be mentioned in comments");
});

// --- phase 2.3: fixes for the reviewed findings ---------------------------

const NL = String.fromCharCode(10);
const REPLACEMENT = String.fromCharCode(0xfffd);

test("25. (M1) createLineReader survives multi-byte characters split across chunks", () => {
  const payload = {
    type: "message_end",
    message: { role: "assistant", content: [{ type: "text", text: "中文解释：库存位置与再订货点合计 20。" }] },
  };
  const bytes = Buffer.from(JSON.stringify(payload) + NL, "utf8");

  // prove this input really breaks naive per-chunk decoding
  const naive = Array.from(bytes)
    .map((b) => Buffer.from([b]).toString("utf8"))
    .join("");
  assert.ok(naive.includes(REPLACEMENT), "test input must be able to corrupt naive decoding");

  const lines = [];
  const reader = createLineReader((line) => lines.push(line));
  for (const byte of bytes) reader.write(Buffer.from([byte])); // one byte per chunk
  reader.end();

  assert.equal(lines.length, 1);
  assert.ok(!lines[0].includes(REPLACEMENT), "no U+FFFD may appear");
  const parsed = JSON.parse(lines[0]);
  assert.equal(parsed.message.content[0].text, "中文解释：库存位置与再订货点合计 20。");
});

test("26. (M1) createLineReader flushes a trailing line that has no final LF", () => {
  const lines = [];
  const reader = createLineReader((line) => lines.push(line));
  reader.write(Buffer.from('{"type":"agent_settled"}', "utf8"));
  assert.equal(lines.length, 0, "an unterminated line is held back until end()");
  reader.end();
  assert.deepEqual(lines, ['{"type":"agent_settled"}']);
});

test("27. (M2) a failed new_session fails fast and never sends prompt", async () => {
  const fake = fakeSpawn(() =>
    makeFakeChild({
      onCommand(command, { emit }) {
        if (command.type === "new_session") {
          emit({ id: command.id, type: "response", command: "new_session", success: false });
        }
      },
    }),
  );
  const ctx = await startProxy({ ...fake, timeoutMs: 30_000 });
  try {
    const t0 = Date.now();
    const res = await post(ctx);
    const elapsed = Date.now() - t0;
    assert.equal(res.status, 502);
    assert.ok(elapsed < 5000, `must fail fast instead of waiting for the timeout (took ${elapsed}ms)`);
    assert.deepEqual(fake.children[0].sent, ["new_session"], "prompt must never be sent");
    assert.equal(fake.children[0].killed, true, "child must be terminated");
    assert.equal(ctx.proxy.live.size, 0, "child must be removed from live");
  } finally {
    await ctx.stop();
  }
});

test("28. (L1) message_end is authoritative and thinking blocks are ignored", async () => {
  const full = "完整的中文解释文本，包含全部要素。";
  const fake = fakeSpawn(() =>
    makeFakeChild({
      onCommand(command, { emit }) {
        if (command.type === "new_session") {
          emit({ id: command.id, type: "response", command: "new_session", success: true });
        }
        if (command.type === "prompt") {
          emit({ id: command.id, type: "response", command: "prompt", success: true });
          emit({ type: "message_update", assistantMessageEvent: { type: "text_delta", delta: "完整的中文" } });
          emit({
            type: "message_end",
            message: {
              role: "assistant",
              content: [
                { type: "thinking", thinking: "内部推理不得外泄" },
                { type: "text", text: full },
              ],
            },
          });
          emit({ type: "agent_settled" });
        }
      },
    }),
  );
  const ctx = await startProxy(fake);
  try {
    const res = await post(ctx);
    assert.equal(res.status, 200);
    const content = res.json.choices[0].message.content;
    assert.equal(content, full, "the authoritative final message must win over partial deltas");
    assert.ok(!content.includes("内部推理"), "thinking blocks must not be exposed");
  } finally {
    await ctx.stop();
  }
});

test("29. (L2) close() resolves even when an idle keep-alive connection exists", async () => {
  const ctx = await startProxy(fakeSpawn(okBehavior()));
  const agent = new http.Agent({ keepAlive: true, maxSockets: 1 });
  try {
    await new Promise((resolve, reject) => {
      const req = http.request({ host: "127.0.0.1", port: ctx.port, path: "/health", method: "GET", agent }, (res) => {
        res.resume();
        res.on("end", resolve);
      });
      req.on("error", reject);
      req.end();
    });
    const t0 = Date.now();
    await ctx.proxy.close();
    const elapsed = Date.now() - t0;
    assert.ok(elapsed < 3000, `close() must not hang on an idle keep-alive socket (took ${elapsed}ms)`);
  } finally {
    agent.destroy();
  }
});

test("30. (L3) an oversized system prompt is rejected before any child is spawned", async () => {
  const fake = fakeSpawn(okBehavior());
  const ctx = await startProxy(fake);
  try {
    const tooLong = "中".repeat(3000); // 9000 bytes > 8 KiB
    assert.ok(Buffer.byteLength(tooLong, "utf8") > MAX_SYSTEM_PROMPT_BYTES);
    const rejected = await post(ctx, TOKEN, {
      messages: [{ role: "system", content: tooLong }, { role: "user", content: "x" }],
    });
    assert.equal(rejected.status, 400);
    assert.equal(rejected.json.error.type, "invalid_request_error");
    assert.equal(fake.children.length, 0, "no RPC child may be spawned for an oversized prompt");

    const okText = "中".repeat(100); // 300 bytes, well within the limit
    const accepted = await post(ctx, TOKEN, {
      messages: [{ role: "system", content: okText }, { role: "user", content: "x" }],
    });
    assert.equal(accepted.status, 200);
    assert.equal(fake.children.length, 1);
  } finally {
    await ctx.stop();
  }
});

test("31. (L5) client-controlled paths and tokens never reach the logs", async () => {
  const ctx = await startProxy(fakeSpawn(okBehavior()));
  try {
    await httpRequest(`${ctx.url}/v1/chat/completions/${TOKEN}?x=${encodeURIComponent("秘密")}`, {
      method: "POST",
      token: TOKEN,
      body: BODY,
    });
    await httpRequest(`${ctx.url}/etc/passwd`, { method: "GET" }); // unauthenticated => 401
    const joined = ctx.logs.join(NL);
    assert.ok(!joined.includes(TOKEN), "the token must never be logged");
    assert.ok(!joined.includes("秘密"), "path content must not be logged");
    assert.ok(!joined.includes("passwd"), "the raw path must not be logged");
    assert.ok(joined.includes("other"), "unmatched routes are logged with a fixed label");
    assert.ok(joined.includes("status=404"), "the authenticated unknown route is logged");
    assert.ok(joined.includes("status=401"), "the unauthenticated probe is logged");
  } finally {
    await ctx.stop();
  }
});

// --- phase 2.5: stopReason semantics --------------------------------------

/** Fake child that finishes with a given stopReason (undefined => no stopReason). */
function stopReasonBehavior(stopReason, { deltas = "", finalText = "" } = {}) {
  return () =>
    makeFakeChild({
      onCommand(command, { emit }) {
        if (command.type === "new_session") {
          emit({ id: command.id, type: "response", command: "new_session", success: true });
        }
        if (command.type === "prompt") {
          emit({ id: command.id, type: "response", command: "prompt", success: true });
          if (deltas) {
            emit({ type: "message_update", assistantMessageEvent: { type: "text_delta", delta: deltas } });
          }
          if (stopReason !== undefined || finalText) {
            const message = { role: "assistant", content: [{ type: "text", text: finalText || deltas }] };
            if (stopReason !== undefined) message.stopReason = stopReason;
            emit({ type: "message_end", message });
          }
          emit({ type: "agent_settled" });
        }
      },
    });
}

test("32. (F1) stopReason=error returns 502 and never leaks partial text", async () => {
  const partial = "不应泄露的部分模型文本";
  const fake = fakeSpawn(stopReasonBehavior("error", { deltas: partial }));
  const ctx = await startProxy(fake);
  try {
    const res = await post(ctx);
    assert.equal(res.status, 502);
    assert.equal(res.json.error.type, "api_error");
    assert.ok(!res.text.includes(partial), "partial model text must not be returned");
    assert.equal(fake.children[0].killed, true);
    assert.equal(ctx.proxy.live.size, 0);
  } finally {
    await ctx.stop();
  }
});

test("33. (F1) stopReason=aborted is an upstream error, not a 499", async () => {
  const fake = fakeSpawn(stopReasonBehavior("aborted", { deltas: "部分内容" }));
  const ctx = await startProxy(fake);
  try {
    const res = await post(ctx);
    assert.notEqual(res.status, 499, "499 is reserved for client disconnects");
    assert.equal(res.status, 502);
    assert.ok(!res.text.includes("部分内容"));
    assert.equal(fake.children[0].killed, true);
    assert.equal(ctx.proxy.live.size, 0);
  } finally {
    await ctx.stop();
  }
});

test("34. (F1) stopReason=toolUse is not passed off as a finished explanation", async () => {
  const fake = fakeSpawn(stopReasonBehavior("toolUse", { deltas: "准备调用工具" }));
  const ctx = await startProxy(fake);
  try {
    const res = await post(ctx);
    assert.equal(res.status, 502);
    assert.ok(!res.text.includes("准备调用工具"));
    assert.equal(fake.children[0].killed, true);
    assert.equal(ctx.proxy.live.size, 0);
  } finally {
    await ctx.stop();
  }
});

test("35. (F1) stopReason=length is returned but flagged as truncated", async () => {
  const truncated = "被截断的解释文本";
  const fake = fakeSpawn(stopReasonBehavior("length", { deltas: truncated }));
  const ctx = await startProxy(fake);
  try {
    const res = await post(ctx);
    assert.equal(res.status, 200);
    assert.equal(res.json.choices[0].message.content, truncated);
    assert.equal(res.json.choices[0].finish_reason, "length", "truncation must be signalled");
    assert.ok(ctx.logs.some((l) => l.includes("finish=length")), "logs use a fixed reason label");
    assert.ok(!ctx.logs.join(NL).includes(truncated), "logs must not contain model text");
  } finally {
    await ctx.stop();
  }
});

test("36. (F1) stopReason=stop returns the full text and a single response", async () => {
  const full = "完整且正常的解释文本。";
  const fake = fakeSpawn(stopReasonBehavior("stop", { deltas: full }));
  const ctx = await startProxy(fake);
  try {
    const res = await post(ctx);
    assert.equal(res.status, 200);
    assert.equal(res.json.choices[0].message.content, full);
    assert.equal(res.json.choices[0].finish_reason, "stop");
    // a duplicated agent_settled must not produce a second response or throw
    fake.children[0].emitLine({ type: "agent_settled" });
    await new Promise((r) => setTimeout(r, 30));
    assert.equal(ctx.logs.filter((l) => l.includes("status=200")).length, 1, "exactly one response");
    assert.equal(ctx.proxy.live.size, 0);
  } finally {
    await ctx.stop();
  }
});

test("37. (F1) a missing stopReason keeps the compatible path without inventing success", async () => {
  const text = "无 stopReason 时的兼容文本";
  const fake = fakeSpawn(stopReasonBehavior(undefined, { deltas: text }));
  const ctx = await startProxy(fake);
  try {
    const res = await post(ctx);
    assert.equal(res.status, 200, "compatibility: still returns the text");
    assert.equal(res.json.choices[0].message.content, text);
    assert.equal(res.json.choices[0].finish_reason, "unknown", "no success reason may be invented");
    assert.equal(fake.children[0].killed, true);
    assert.equal(ctx.proxy.live.size, 0);
  } finally {
    await ctx.stop();
  }
});

test("38. (F3) resolveStartupToken handles the token display switch and explicit tokens", () => {
  assert.deepEqual(resolveStartupToken({}), { token: undefined, showToken: true });
  assert.equal(resolveStartupToken({ PI_LLM_PROXY_SHOW_TOKEN: "1" }).showToken, true);
  assert.equal(resolveStartupToken({ PI_LLM_PROXY_SHOW_TOKEN: "0" }).showToken, false);
  assert.equal(resolveStartupToken({ PI_LLM_PROXY_SHOW_TOKEN: "FALSE" }).showToken, false);
  assert.equal(resolveStartupToken({ PI_LLM_PROXY_SHOW_TOKEN: " no " }).showToken, false);
  const supplied = "x".repeat(32);
  assert.deepEqual(resolveStartupToken({ PI_LLM_PROXY_TOKEN: supplied }), {
    token: supplied,
    showToken: true,
  });
  assert.equal(
    resolveStartupToken({ PI_LLM_PROXY_TOKEN: supplied, PI_LLM_PROXY_SHOW_TOKEN: "0" }).showToken,
    false,
    "hiding must not disable the supplied token",
  );
  assert.throws(() => resolveStartupToken({ PI_LLM_PROXY_TOKEN: "short" }), /at least 16 characters/);
});
