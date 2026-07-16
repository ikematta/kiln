// Fully static SPA: one prerendered shell, everything else client-side
// against the gateway's /admin API (same origin).
export const prerender = true;
export const ssr = false;
