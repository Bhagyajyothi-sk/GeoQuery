/**
 * GeoQuery AI — Centralized API Client
 *
 * All backend communication goes through this module.
 * Set VITE_API_BASE in a .env file to override the default.
 */

// ── Base URL ──────────────────────────────────────────────────────────────────
export const API_BASE =
  import.meta.env?.VITE_API_BASE || "http://127.0.0.1:8000";

// ── Internal helper ───────────────────────────────────────────────────────────

async function _post(path, body) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const data = await response.json().catch(() => ({}));

  if (!response.ok) {
    const message =
      data?.detail ||
      data?.message ||
      `Request to ${path} failed (HTTP ${response.status}).`;
    throw new Error(message);
  }

  return data;
}

// ── Public API ────────────────────────────────────────────────────────────────

/** Run the full NL pipeline. POST /api/query { query } */
export async function runQuery(queryText) {
  return _post("/api/query", { query: queryText.trim() });
}

/** Generate NDVI map PNG. POST /api/maps/ndvi */
export async function generateNDVIMap({
  latitude, longitude, startDate, endDate, maxCloudCover = 30, buffer = 0.05,
}) {
  const data = await _post("/api/maps/ndvi", {
    latitude, longitude,
    start_date: startDate, end_date: endDate,
    max_cloud_cover: maxCloudCover, buffer,
  });
  if (!data.map_url) throw new Error("NDVI computation succeeded but no map URL returned.");
  const mapUrl = data.map_url.startsWith("http") ? data.map_url : `${API_BASE}${data.map_url}`;
  return { ...data, map_url: mapUrl };
}

/** Generate NDWI map PNG. POST /api/maps/ndwi */
export async function generateNDWIMap({
  latitude, longitude, startDate, endDate, maxCloudCover = 30, buffer = 0.05,
}) {
  const data = await _post("/api/maps/ndwi", {
    latitude, longitude,
    start_date: startDate, end_date: endDate,
    max_cloud_cover: maxCloudCover, buffer,
  });
  if (!data.map_url) throw new Error("NDWI computation succeeded but no map URL returned.");
  const mapUrl = data.map_url.startsWith("http") ? data.map_url : `${API_BASE}${data.map_url}`;
  return { ...data, map_url: mapUrl };
}

/** Two-period NDVI change detection. POST /api/maps/change-detection */
export async function runChangeDetection({
  latitude, longitude,
  firstStartDate, firstEndDate, secondStartDate, secondEndDate,
  maxCloudCover = 30,
}) {
  return _post("/api/maps/change-detection", {
    latitude, longitude,
    first_start_date: firstStartDate, first_end_date: firstEndDate,
    second_start_date: secondStartDate, second_end_date: secondEndDate,
    max_cloud_cover: maxCloudCover,
  });
}

/** Semantic satellite-tile search. POST /api/search */
export async function semanticSearch({ visualQuery, location = null, topK = 5 }) {
  return _post("/api/search", { visual_query: visualQuery, location, top_k: topK });
}

/** Backend liveness probe. GET /health */
export async function checkHealth() {
  const response = await fetch(`${API_BASE}/health`);
  if (!response.ok) throw new Error("Backend health check failed.");
  return response.json();
}

// ── Date helpers ──────────────────────────────────────────────────────────────

/** Split a date range into two equal halves for change detection. */
export function splitDateRange(startDate, endDate) {
  const start = new Date(startDate);
  const end = new Date(endDate);
  const midpoint = new Date(start.getTime() + (end.getTime() - start.getTime()) / 2);
  const firstEnd = midpoint.toISOString().slice(0, 10);
  const secondStart = new Date(midpoint.getTime() + 86400000).toISOString().slice(0, 10);
  return { firstStart: startDate, firstEnd, secondStart, secondEnd: endDate };
}

/** Default 12-month window ending today. */
export function defaultDateWindow() {
  const today = new Date();
  const lastYear = new Date(today);
  lastYear.setFullYear(lastYear.getFullYear() - 1);
  return { startDate: lastYear.toISOString().slice(0, 10), endDate: today.toISOString().slice(0, 10) };
}

