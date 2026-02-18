/**
 * EchoZero Auth Verification Worker
 *
 * Cloudflare Worker that acts as a proxy between the EchoZero desktop app
 * and the Memberstack Admin API. The Admin API key stays server-side --
 * the desktop app never sees it.
 *
 * Authorization logic:
 *   A member is verified ONLY if they have at least one active plan connection
 *   in Memberstack. A free signup with no plan is rejected.
 *
 *   To restrict to specific plans, set the ALLOWED_PLAN_IDS env var
 *   (comma-separated list of Memberstack plan IDs). If not set, any
 *   active plan grants access.
 *
 * Endpoints:
 *   POST /verify  { "member_id": "mem_...", "token": "..." }
 *     ->  { "verified": true, "email": "...", "billing_period_end": "...", ... }
 *
 *   POST /link  { "code": "...", "token": "...", "member_id": "..." }  (no X-App-Token; from browser)
 *     ->  { "ok": true }  (stores in KV for polling; browser-agnostic flow)
 *   GET /link?code=X  (requires X-App-Token; from desktop app)
 *     ->  { "linked": true, "token": "...", "member_id": "...", "member_info": {...} }
 *
 * Secrets (set via `wrangler secret put`):
 *   MEMBERSTACK_ADMIN_KEY  - Memberstack Admin API key (sk_...)
 *   APP_SECRET             - Shared secret the desktop app sends in X-App-Token header
 *
 * Optional env vars (set in wrangler.toml [vars] or via `wrangler secret put`):
 *   ALLOWED_PLAN_IDS       - Comma-separated plan IDs that grant access (e.g., "pln_abc123,pln_def456")
 *                            If not set, any active plan connection grants access.
 */

const CORS_HEADERS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, X-App-Token",
};

const LINK_TTL = 300; // 5 minutes

export default {
  async fetch(request, env) {
    // Handle CORS preflight
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    const url = new URL(request.url);

    // --- /link: server-mediated flow (browser-agnostic) ---
    if (url.pathname === "/link") {
      if (request.method === "POST") {
        return handleLinkPost(request, env, CORS_HEADERS);
      }
      if (request.method === "GET") {
        return handleLinkGet(request, env, CORS_HEADERS);
      }
      return Response.json(
        { error: "Method not allowed" },
        { status: 405, headers: CORS_HEADERS }
      );
    }

    // --- /verify: original desktop verification ---
    if (url.pathname !== "/verify") {
      return Response.json(
        { verified: false, error: "Not found" },
        { status: 404, headers: CORS_HEADERS }
      );
    }

    if (request.method !== "POST") {
      return Response.json(
        { verified: false, error: "Method not allowed" },
        { status: 405, headers: CORS_HEADERS }
      );
    }

    // Validate app token (prevents unauthorized callers)
    if (env.APP_SECRET) {
      const appToken = request.headers.get("X-App-Token");
      if (appToken !== env.APP_SECRET) {
        return Response.json(
          { verified: false, error: "Invalid app token" },
          { status: 401, headers: CORS_HEADERS }
        );
      }
    }

    // Parse request body
    let body;
    try {
      body = await request.json();
    } catch {
      return Response.json(
        { verified: false, error: "Invalid JSON body" },
        { status: 400, headers: CORS_HEADERS }
      );
    }

    const memberId = body.member_id;
    const token = body.token;
    if (!memberId || typeof memberId !== "string") {
      return Response.json(
        { verified: false, error: "Missing or invalid member_id" },
        { status: 400, headers: CORS_HEADERS }
      );
    }
    if (!token || typeof token !== "string") {
      return Response.json(
        { verified: false, error: "Missing or invalid token" },
        { status: 400, headers: CORS_HEADERS }
      );
    }

    try {
      const result = await verifyMemberAndGetInfo(env, memberId, token);
      if (!result.success) {
        return Response.json(result.body, { status: result.status, headers: CORS_HEADERS });
      }
      return Response.json(result.body, { status: 200, headers: CORS_HEADERS });
    } catch (err) {
      return Response.json(
        { verified: false, error: "Verification service unavailable" },
        { status: 502, headers: CORS_HEADERS }
      );
    }
  },
};

async function handleLinkPost(request, env, CORS_HEADERS) {
  let body;
  try {
    body = await request.json();
  } catch {
    return Response.json(
      { ok: false, error: "Invalid JSON body" },
      { status: 400, headers: CORS_HEADERS }
    );
  }
  const code = body.code;
  const token = body.token;
  const memberId = body.member_id;
  if (!code || typeof code !== "string" || code.length < 8 || code.length > 64) {
    return Response.json(
      { ok: false, error: "Invalid or missing code" },
      { status: 400, headers: CORS_HEADERS }
    );
  }
  if (!memberId || typeof memberId !== "string") {
    return Response.json(
      { ok: false, error: "Missing member_id" },
      { status: 400, headers: CORS_HEADERS }
    );
  }
  if (!token || typeof token !== "string") {
    return Response.json(
      { ok: false, error: "Missing token" },
      { status: 400, headers: CORS_HEADERS }
    );
  }

  const result = await verifyMemberAndGetInfo(env, memberId, token);
  if (!result.success) {
    const errBody = result.body && result.body.error ? result.body : { error: "Verification failed" };
    return Response.json(errBody, { status: result.status || 401, headers: CORS_HEADERS });
  }

  const kvKey = "link:" + code;
  const payload = JSON.stringify({
    token,
    member_id: memberId,
    member_info: result.body,
  });
  try {
    await env.LINK_STORE.put(kvKey, payload, { expirationTtl: LINK_TTL });
  } catch (e) {
    return Response.json(
      { ok: false, error: "Storage error" },
      { status: 502, headers: CORS_HEADERS }
    );
  }
  return Response.json({ ok: true }, { status: 200, headers: CORS_HEADERS });
}

async function handleLinkGet(request, env, CORS_HEADERS) {
  if (env.APP_SECRET) {
    const appToken = request.headers.get("X-App-Token");
    if (appToken !== env.APP_SECRET) {
      return Response.json(
        { error: "Unauthorized" },
        { status: 401, headers: CORS_HEADERS }
      );
    }
  }
  const url = new URL(request.url);
  const code = url.searchParams.get("code");
  if (!code || code.length < 8) {
    return Response.json(
      { error: "Missing or invalid code" },
      { status: 400, headers: CORS_HEADERS }
    );
  }
  const kvKey = "link:" + code;
  const value = await env.LINK_STORE.get(kvKey);
  if (!value) {
    return Response.json(
      { error: "Not found", linked: false },
      { status: 404, headers: CORS_HEADERS }
    );
  }
  await env.LINK_STORE.delete(kvKey);
  let data;
  try {
    data = JSON.parse(value);
  } catch {
    return Response.json(
      { error: "Invalid stored data" },
      { status: 500, headers: CORS_HEADERS }
    );
  }
  return Response.json(
    { linked: true, token: data.token, member_id: data.member_id, member_info: data.member_info },
    { status: 200, headers: CORS_HEADERS }
  );
}

async function verifyMemberAndGetInfo(env, memberId, token) {
  const verifyResponse = await fetch(
    "https://admin.memberstack.com/members/verify-token",
    {
      method: "POST",
      headers: {
        "x-api-key": env.MEMBERSTACK_ADMIN_KEY,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ token }),
    }
  );

  if (!verifyResponse.ok) {
    return { success: false, status: 401, body: { verified: false, error: "Token invalid or expired" } };
  }

  const tokenData = await verifyResponse.json();
  const decoded = tokenData.data || tokenData;
  const nowSeconds = Math.floor(Date.now() / 1000);
  if (!decoded || decoded.type !== "member" || !decoded.id || decoded.exp < nowSeconds) {
    return { success: false, status: 401, body: { verified: false, error: "Token invalid or expired" } };
  }
  if (decoded.id !== memberId) {
    return { success: false, status: 401, body: { verified: false, error: "Token/member mismatch" } };
  }

  const msResponse = await fetch(
    `https://admin.memberstack.com/members/${encodeURIComponent(memberId)}`,
    {
      headers: {
        "x-api-key": env.MEMBERSTACK_ADMIN_KEY,
        "Content-Type": "application/json",
      },
    }
  );

  if (msResponse.status === 404) {
    return { success: false, status: 404, body: { verified: false, error: "Member not found" } };
  }
  if (!msResponse.ok) {
    return { success: false, status: 502, body: { verified: false, error: `Memberstack API error: ${msResponse.status}` } };
  }

  const data = await msResponse.json();
  const member = data.data || data;
  const planConnections = member.planConnections || [];
  const activePlans = planConnections.filter((pc) => pc.status === "ACTIVE");

  if (activePlans.length === 0) {
    return {
      success: false,
      status: 403,
      body: {
        verified: false,
        error: "No active plan. Please subscribe to access EchoZero.",
        id: member.id,
        email: (member.auth && member.auth.email) || "",
      },
    };
  }

  if (env.ALLOWED_PLAN_IDS) {
    const allowedIds = env.ALLOWED_PLAN_IDS.split(",").map((id) => id.trim());
    const hasAllowedPlan = activePlans.some((pc) => allowedIds.includes(pc.planId));
    if (!hasAllowedPlan) {
      return {
        success: false,
        status: 403,
        body: {
          verified: false,
          error: "Your current plan does not include desktop app access.",
          id: member.id,
          email: (member.auth && member.auth.email) || "",
        },
      };
    }
  }

  const billingPeriodEnd = getBillingPeriodEnd(activePlans);
  return {
    success: true,
    status: 200,
    body: {
      verified: true,
      id: member.id,
      email: (member.auth && member.auth.email) || "",
      plan_connections: activePlans.map((pc) => ({
        plan_id: pc.planId,
        plan_name: pc.planName || "",
        status: pc.status,
        billing_period_end: extractPeriodEnd(pc),
      })),
      billing_period_end: billingPeriodEnd || "",
      created_at: member.createdAt || "",
    },
  };
}

function getBillingPeriodEnd(activePlans) {
  let latest = null;
  for (const pc of activePlans) {
    const candidate = extractPeriodEnd(pc);
    if (!candidate) continue;
    if (!latest || candidate > latest) {
      latest = candidate;
    }
  }
  return latest;
}

function extractPeriodEnd(planConnection) {
  if (!planConnection || typeof planConnection !== "object") {
    return "";
  }

  const candidates = [
    planConnection.currentPeriodEnd,
    planConnection.current_period_end,
    planConnection.periodEnd,
    planConnection.period_end,
    planConnection.endsAt,
    planConnection.endAt,
    planConnection.expiresAt,
    planConnection.expireAt,
    planConnection.payment && planConnection.payment.currentPeriodEnd,
    planConnection.payment && planConnection.payment.current_period_end,
    planConnection.payment && planConnection.payment.periodEnd,
    planConnection.payment && planConnection.payment.period_end,
    planConnection.payment && planConnection.payment.endsAt,
    planConnection.payment && planConnection.payment.endAt,
  ];

  let latest = null;
  for (const value of candidates) {
    const parsed = parseDate(value);
    if (!parsed) continue;
    if (!latest || parsed > latest) {
      latest = parsed;
    }
  }

  return latest ? latest.toISOString() : "";
}

function parseDate(value) {
  if (value === null || value === undefined || value === "") {
    return null;
  }

  if (typeof value === "number") {
    // Accept both seconds and milliseconds timestamps.
    const ms = value > 1000000000000 ? value : value * 1000;
    const date = new Date(ms);
    if (Number.isNaN(date.getTime())) return null;
    return date;
  }

  if (typeof value === "string") {
    // Handle numeric strings as timestamps.
    if (/^\d+$/.test(value)) {
      return parseDate(Number(value));
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return null;
    return date;
  }

  return null;
}
