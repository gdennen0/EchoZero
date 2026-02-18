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
  "Access-Control-Allow-Methods": "POST, OPTIONS",
  "Access-Control-Allow-Headers": "Content-Type, X-App-Token",
};

export default {
  async fetch(request, env) {
    // Handle CORS preflight
    if (request.method === "OPTIONS") {
      return new Response(null, { status: 204, headers: CORS_HEADERS });
    }

    // Only POST allowed
    if (request.method !== "POST") {
      return Response.json(
        { verified: false, error: "Method not allowed" },
        { status: 405, headers: CORS_HEADERS }
      );
    }

    // Route check
    const url = new URL(request.url);
    if (url.pathname !== "/verify") {
      return Response.json(
        { verified: false, error: "Not found" },
        { status: 404, headers: CORS_HEADERS }
      );
    }

    // Validate app token (prevents unauthorized callers)
    if (env.APP_SECRET) {
      const token = request.headers.get("X-App-Token");
      if (token !== env.APP_SECRET) {
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

    // Verify the member token first, then ensure it matches member_id.
    try {
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
        return Response.json(
          { verified: false, error: "Token invalid or expired" },
          { status: 401, headers: CORS_HEADERS }
        );
      }

      const tokenData = await verifyResponse.json();
      const decoded = tokenData.data || tokenData;
      const nowSeconds = Math.floor(Date.now() / 1000);
      if (!decoded || decoded.type !== "member" || !decoded.id || decoded.exp < nowSeconds) {
        return Response.json(
          { verified: false, error: "Token invalid or expired" },
          { status: 401, headers: CORS_HEADERS }
        );
      }

      if (decoded.id !== memberId) {
        return Response.json(
          { verified: false, error: "Token/member mismatch" },
          { status: 401, headers: CORS_HEADERS }
        );
      }

      // Call Memberstack Admin API for current subscription state.
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
        return Response.json(
          { verified: false, error: "Member not found" },
          { status: 404, headers: CORS_HEADERS }
        );
      }

      if (!msResponse.ok) {
        return Response.json(
          { verified: false, error: `Memberstack API error: ${msResponse.status}` },
          { status: 502, headers: CORS_HEADERS }
        );
      }

      const data = await msResponse.json();
      const member = data.data || data;
      const planConnections = member.planConnections || [];

      // --- AUTHORIZATION CHECK ---
      // Member must have at least one active plan connection.
      // "active" means the plan connection status is "ACTIVE".
      const activePlans = planConnections.filter(
        (pc) => pc.status === "ACTIVE"
      );

      if (activePlans.length === 0) {
        return Response.json(
          {
            verified: false,
            error: "No active plan. Please subscribe to access EchoZero.",
            id: member.id,
            email: (member.auth && member.auth.email) || "",
          },
          { status: 403, headers: CORS_HEADERS }
        );
      }

      // Optional: restrict to specific plan IDs
      // Set ALLOWED_PLAN_IDS as a comma-separated string in wrangler.toml or secrets
      if (env.ALLOWED_PLAN_IDS) {
        const allowedIds = env.ALLOWED_PLAN_IDS.split(",").map((id) =>
          id.trim()
        );
        const hasAllowedPlan = activePlans.some((pc) =>
          allowedIds.includes(pc.planId)
        );

        if (!hasAllowedPlan) {
          return Response.json(
            {
              verified: false,
              error:
                "Your current plan does not include desktop app access.",
              id: member.id,
              email: (member.auth && member.auth.email) || "",
            },
            { status: 403, headers: CORS_HEADERS }
          );
        }
      }

      const billingPeriodEnd = getBillingPeriodEnd(activePlans);

      // Member exists and has an active (allowed) plan -- verified
      return Response.json(
        {
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
        { status: 200, headers: CORS_HEADERS }
      );
    } catch (err) {
      return Response.json(
        { verified: false, error: "Verification service unavailable" },
        { status: 502, headers: CORS_HEADERS }
      );
    }
  },
};

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
