# EchoZero Web Portal Implementation Plan

Step-by-step plan for Webflow + MemberStack member portal, gated content, and desktop app telemetry.

---

## Phase 1: MemberStack Setup and Page Gating

### Step 1.1: Verify MemberStack Integration

1. In Webflow: ensure MemberStack is installed (Project Settings > Integrations > MemberStack).
2. Add the MemberStack embed code to your site (if not already present).
3. Confirm your OpenBeta plan exists in MemberStack dashboard.
4. Verify the plan is set to $0 and allows signup.

### Step 1.2: Create Member-Only Pages Structure

1. In Webflow Pages panel, ensure these pages exist under "MemberPortal":
   - Home (dashboard/hub)
   - Downloads
   - Docs
   - Chat
   - Profile settings
   - Bug Submit (create if missing)
   - Plans/billing (create if missing)

### Step 1.3: Apply MemberStack Protection to Each Page

1. Open each page: Downloads, Docs, Chat, Profile settings, Bug Submit, Plans/billing.
2. Add MemberStack "Member Only" protection:
   - Use MemberStack's page protection (e.g., "Protected Pages" in MemberStack dashboard, or visibility rules in Webflow).
   - Require plan: OpenBeta for each protected page.
3. Configure redirect: when non-members hit a protected page, redirect to login or signup.
4. Test: log out, visit each page, confirm redirect. Log in, confirm access.

### Step 1.4: Set Up Login/Signup Flow

1. Ensure login and signup forms are on your site (or linked from MemberStack).
2. Add a "Login" and "Sign Up" link in your nav (visible when logged out).
3. Add "Member Portal" or "Dashboard" link (visible when logged in).
4. Test full flow: signup, login, logout, access to protected pages.

---

## Phase 2: Portal Page Content

### Step 2.1: Downloads Page

1. Add download links (installers, release notes).
2. Use Webflow CMS or static links.
3. If you have multiple file types or versions, consider a simple table or list.
4. Gate the entire page with MemberStack (Step 1.3).

### Step 2.2: Docs Page

1. Decide: embed docs in Webflow, or link to external docs (e.g., GitBook, Notion).
2. If in Webflow: create a Docs collection, add articles, render on the Docs page.
3. If external: add a link/card that goes to your docs URL.
4. Ensure the page (or the external link) is only reachable when logged in.

### Step 2.3: Chat Page

1. Integrate your chat provider (Intercom, Crisp, Discord embed, etc.).
2. Add the chat widget or embed to the Chat page.
3. Ensure the Chat page is gated (Step 1.3).
4. If using Discord: embed the Discord widget or provide an invite link.

### Step 2.4: Profile Settings Page

1. Add MemberStack profile component if available (email, name, password).
2. Or create a simple layout with links to MemberStack account management.
3. Ensure profile page is gated.

### Step 2.5: Plans/Billing Page

1. Add MemberStack billing/plan component or embed.
2. Show current plan (OpenBeta) and upgrade options for future paid plans.
3. Allow users to view and manage subscription (cancel, change plan).
4. Gate the page.

### Step 2.6: Bug Submit Page

1. Create a Webflow form with fields:
   - Title (required)
   - Description (required)
   - Category (e.g., Bug, Feature Request)
   - Steps to reproduce (optional)
   - Attachments (if Webflow supports; otherwise skip or use external tool)
2. Configure form submission handler (see Step 2.7).
3. Gate the page so only logged-in members can submit.

### Step 2.7: Bug Submit Form Backend

Choose one:

**Option A: Zapier/Make (no code)**
1. Connect Webflow form to Zapier or Make.
2. Trigger: form submitted.
3. Action: send to email, Slack, Notion, or Airtable.
4. Optionally use MemberStack to pass user email into the form (hidden field).

**Option B: Custom API**
1. Create a serverless function (Vercel, Netlify, Cloudflare Workers).
2. Endpoint: POST /api/bugs with { title, description, category, user_email }.
3. Store in database (Supabase, Airtable, etc.).
4. Point Webflow form to this endpoint via custom code or third-party form service.

**Option C: Third-party form service**
1. Use Typeform, Tally, or Jotform with MemberStack gating.
2. Embed or link from Bug Submit page.
3. Configure to capture member email via MemberStack.

---

## Phase 3: Navigation and UX

### Step 3.1: Portal Navigation

1. Create consistent nav for MemberPortal pages:
   - Home | Downloads | Docs | Chat | Profile | Bug Submit | Plans
2. Use MemberStack visibility: show nav only when logged in.
3. Ensure active page is highlighted.

### Step 3.2: Redirects and Empty States

1. When non-member visits /member-portal/*: redirect to login.
2. Add a friendly empty state or CTA on Home for new members.
3. Add breadcrumbs if helpful for Docs.

### Step 3.3: Mobile

1. Ensure nav collapses for mobile.
2. Test all forms and links on mobile.

---

## Phase 4: Desktop App Telemetry

### Step 4.1: Define What to Collect

1. List events you care about, e.g.:
   - App launched
   - Block executed (type, success/failure)
   - Project saved/loaded
   - Export completed
   - Session duration
   - Version, OS, machine ID (hashed)
2. Avoid: PII, file contents, project data.
3. Decide: opt-in, opt-out, or always-on (with disclosure).

### Step 4.2: Create Telemetry Backend

1. Choose hosting: Vercel + Supabase, Railway, AWS Lambda + DynamoDB, etc.
2. Create API:
   - POST /api/telemetry or /api/events
   - Auth: API key or JWT (from MemberStack or your auth).
   - Body: { event, timestamp, properties, user_id?, session_id? }
3. Store in database with schema: event, timestamp, properties (JSON), user_id.
4. Add basic validation and rate limiting.

### Step 4.3: Add Telemetry to EchoZero Desktop App

1. Add a telemetry module (e.g., `src/infrastructure/telemetry/`).
2. Implement: `track(event, properties)` that POSTs to your API.
3. Hook into app lifecycle (startup, shutdown) and key actions.
4. Use async/batch sends to avoid blocking UI.
5. Respect user preference (opt-in/out) if you support it.

### Step 4.4: Link Telemetry to Members

1. When user logs in via MemberStack in the desktop app, store member ID or email (hashed).
2. Include this ID in telemetry payloads so you can associate usage with accounts.
3. Ensure your auth flow (e.g., webflow_desktop_login) provides a stable identifier.

### Step 4.5: Analytics and Dashboards

1. Query your telemetry DB for usage reports.
2. Use Metabase, Grafana, or simple scripts.
3. Track: DAU, events per user, block usage, version adoption.

---

## Phase 5: Testing and Launch

### Step 5.1: MemberStack Gating Test

1. Log out, visit each protected page; confirm redirect.
2. Log in as OpenBeta member; confirm access.
3. Test from incognito and different browsers.

### Step 5.2: Form and Flows Test

1. Submit a test bug report; verify it arrives.
2. Test profile update (if applicable).
3. Test Plans/billing flow (view plan, cancel if you have that).

### Step 5.3: Desktop Telemetry Test

1. Run desktop app; trigger key events.
2. Check backend logs and DB for events.
3. Verify no PII is sent and rate limits work.

### Step 5.4: Documentation and Privacy

1. Add a Privacy Policy covering: what you collect, how you use it, retention.
2. Add Terms of Service if needed.
3. Add a simple "About data collection" in the app if telemetry is always-on.

---

## Checklist Summary

| Phase | Task | Status |
|-------|------|--------|
| 1.1 | Verify MemberStack integration | [ ] |
| 1.2 | Create all portal pages | [ ] |
| 1.3 | Apply protection to each page | [ ] |
| 1.4 | Set up login/signup flow | [ ] |
| 2.1 | Build Downloads page content | [ ] |
| 2.2 | Build Docs page content | [ ] |
| 2.3 | Build Chat page content | [ ] |
| 2.4 | Build Profile settings page | [ ] |
| 2.5 | Build Plans/billing page | [ ] |
| 2.6 | Build Bug Submit form | [ ] |
| 2.7 | Configure Bug Submit backend | [ ] |
| 3.1 | Portal navigation | [ ] |
| 3.2 | Redirects and empty states | [ ] |
| 3.3 | Mobile responsiveness | [ ] |
| 4.1 | Define telemetry events | [ ] |
| 4.2 | Create telemetry API | [ ] |
| 4.3 | Add telemetry to desktop app | [ ] |
| 4.4 | Link telemetry to member IDs | [ ] |
| 4.5 | Set up analytics/dashboards | [ ] |
| 5.1 | Gating tests | [ ] |
| 5.2 | Form and flow tests | [ ] |
| 5.3 | Telemetry tests | [ ] |
| 5.4 | Privacy/Terms docs | [ ] |

---

## Recommended Order

1. Phase 1 (MemberStack + gating) – foundation.
2. Phase 2.1–2.5 (page content) – value for members.
3. Phase 2.6–2.7 (Bug Submit) – feedback loop.
4. Phase 3 (nav and UX) – polish.
5. Phase 4 (telemetry) – after portal is stable.
6. Phase 5 (testing and docs) – before and after launch.

---

## References

- MemberStack + Webflow: https://memberstack.com
- Webflow forms: https://university.webflow.com/lesson/forms
- Your existing auth: `src/infrastructure/auth/webflow_desktop_login.html`, `license_lease.py`
