/**
 * EchoZero Models Distribution Worker
 *
 * Cloudflare Worker that serves ML model files from R2.
 * Access is gated by APP_SECRET (X-App-Token header) which proves the
 * request comes from a legitimate EchoZero app instance.  Per-user
 * subscription validation happens client-side via the license lease
 * (valid until billing cycle end) -- no Memberstack round-trips here.
 *
 * R2 bucket layout:
 *   manifest.json              -- model registry
 *   models/<filename>.pth      -- model files
 *
 * Endpoints:
 *   GET /manifest              -- returns manifest.json
 *   GET /download/<filename>   -- streams model file with Content-Length
 *
 * Secrets (set via wrangler secret put):
 *   APP_SECRET  -- shared secret the desktop app sends as X-App-Token
 */

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, X-App-Token",
};

export default {
  async fetch(request, env) {
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    if (request.method !== "GET") {
      return jsonResponse({ error: "Method not allowed" }, 405);
    }

    // Validate APP_SECRET
    if (env.APP_SECRET) {
      const appToken = request.headers.get("X-App-Token");
      if (appToken !== env.APP_SECRET) {
        return jsonResponse({ error: "Unauthorized" }, 401);
      }
    }

    const url = new URL(request.url);

    if (url.pathname === "/manifest") {
      return handleManifest(env);
    }

    const downloadMatch = url.pathname.match(/^\/download\/(.+)$/);
    if (downloadMatch) {
      return handleDownload(env, downloadMatch[1]);
    }

    return jsonResponse({ error: "Not found" }, 404);
  },
};

async function handleManifest(env) {
  const object = await env.MODELS_BUCKET.get("manifest.json");
  if (!object) {
    return jsonResponse({ error: "Manifest not found" }, 404);
  }

  const manifest = await object.text();
  return new Response(manifest, {
    status: 200,
    headers: {
      ...CORS_HEADERS,
      "Content-Type": "application/json",
      "Cache-Control": "no-cache",
    },
  });
}

async function handleDownload(env, filename) {
  const safeFilename = filename.replace(/[^a-zA-Z0-9._-]/g, "");
  if (!safeFilename || safeFilename !== filename) {
    return jsonResponse({ error: "Invalid filename" }, 400);
  }

  const object = await env.MODELS_BUCKET.get(`models/${safeFilename}`);
  if (!object) {
    return jsonResponse({ error: "Model not found" }, 404);
  }

  return new Response(object.body, {
    status: 200,
    headers: {
      ...CORS_HEADERS,
      "Content-Type": "application/octet-stream",
      "Content-Length": object.size.toString(),
      "Content-Disposition": `attachment; filename="${safeFilename}"`,
    },
  });
}

function jsonResponse(body, status) {
  return Response.json(body, { status, headers: CORS_HEADERS });
}
