// Deployment config for the static frontend (served from Vercel).
// After you deploy the backend to Render, paste its URL below and commit.
// This intentionally stays a plain JS file (no build step) so Vercel can
// serve this folder as-is with zero configuration.
window.__API_BASE_URL__ = "https://YOUR-BACKEND-NAME.onrender.com";

// Only needed if you set APP_API_KEY on the Render backend. Leave blank to
// run without auth (fine for a personal demo, not for a public link you'll
// share widely - see SECURITY.md).
window.__API_KEY__ = "";
