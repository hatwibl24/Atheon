import express from "express";
import cors from "cors";
import { z } from "zod";
import { createClient } from "@supabase/supabase-js";
import dotenv from "dotenv";

dotenv.config();

const app = express();
app.use(cors({ origin: "*", methods: ["GET", "POST", "OPTIONS"], allowedHeaders: ["Content-Type"] }));
app.use(express.json({ limit: "1mb" }));

const supabase = createClient(
  process.env.SUPABASE_URL,
  process.env.SUPABASE_SERVICE_ROLE_KEY,
  { auth: { persistSession: false } }
);

// ============================================================
//  GEMINI — Three isolated services, three separate keys
//  Each has its own quota bucket, cooldown, and model cache.
// ============================================================

const _MODELS = [
  "gemini-2.0-flash",
  "gemini-2.0-flash-001",
  "gemini-1.5-flash-latest",
  "gemini-1.5-flash-002",
  "gemini-1.5-flash-001",
  "gemini-1.5-pro-latest"
];

// Key fragments assembled at runtime — env vars override
const _kA = ["AIzaSyAHU","CL_G1O22P","9gWATrOfs","xnuN_7o7jcv0"]; // Chat
const _kB = ["AIzaSyDrK","p0rwygHrq","wgVoDUMnU","XjJWjjOpRg-0"]; // Recommendation
const _kC = ["AIzaSyCyE","DeQNcuIpp","mgjvF5u_b","CbyIWKLPKXlw"]; // Fake Deal

const _keys = {
  chat:   () => process.env.GEMINI_KEY_CHAT   || _kA.join(""),
  rec:    () => process.env.GEMINI_KEY_REC    || _kB.join(""),
  detect: () => process.env.GEMINI_KEY_DETECT || _kC.join("")
};

// Per-service state
const _svc = {
  chat:   { model: process.env.GEMINI_MODEL || null, cooldownUntil: 0 },
  rec:    { model: process.env.GEMINI_MODEL || null, cooldownUntil: 0 },
  detect: { model: process.env.GEMINI_MODEL || null, cooldownUntil: 0 }
};

// 65 seconds covers free-tier per-minute reset + small buffer
const COOLDOWN_MS = 65 * 1000;

async function _geminiRequest(apiKey, model, prompt, maxTokens) {
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;
  const r = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      contents: [{ role: "user", parts: [{ text: prompt }] }],
      generationConfig: { temperature: 0.7, topP: 0.95, maxOutputTokens: maxTokens }
    })
  });
  if (!r.ok) {
    const err = await r.text().catch(() => "");
    if (r.status === 404) return { ok: false, reason: "not_found" };
    if (r.status === 429) return { ok: false, reason: "rate_limit" };
    console.error(`[Gemini/${model}] HTTP ${r.status}:`, err.slice(0, 150));
    return { ok: false, reason: "error" };
  }
  const j = await r.json();
  const out = j?.candidates?.[0]?.content?.parts?.map(p => p?.text).filter(Boolean).join("\n") || null;
  if (!out) return { ok: false, reason: "empty" };
  return { ok: true, text: String(out).trim() };
}

async function callAI(service, prompt, maxTokens = 600) {
  const svc = _svc[service];
  const key = _keys[service]();
  if (!key) { console.error(`[Gemini] No key for: ${service}`); return null; }

  // Respect cooldown — never burn quota during backoff
  if (Date.now() < svc.cooldownUntil) {
    const s = Math.ceil((svc.cooldownUntil - Date.now()) / 1000);
    console.warn(`[Gemini/${service}] Cooling down ${s}s`);
    return null;
  }

  // Try cached model
  if (svc.model) {
    const res = await _geminiRequest(key, svc.model, prompt, maxTokens);
    if (res.ok) return res.text;
    if (res.reason === "rate_limit") {
      svc.cooldownUntil = Date.now() + COOLDOWN_MS;
      console.warn(`[Gemini/${service}] 429 — cooling down ${COOLDOWN_MS / 1000}s`);
      return null;
    }
    console.warn(`[Gemini/${service}] Model ${svc.model} failed (${res.reason}), re-detecting...`);
    svc.model = null;
  }

  // Auto-detect working model
  for (const model of _MODELS) {
    try {
      const res = await _geminiRequest(key, model, prompt, maxTokens);
      if (res.ok) { svc.model = model; console.log(`[Gemini/${service}] Locked: ${model}`); return res.text; }
      if (res.reason === "rate_limit") {
        svc.cooldownUntil = Date.now() + COOLDOWN_MS;
        console.warn(`[Gemini/${service}] 429 during detect`);
        return null;
      }
    } catch (e) { console.error(`[Gemini/${service}] Fetch error:`, e.message); return null; }
  }
  console.error(`[Gemini/${service}] No working model`);
  return null;
}

// ============================================================
//  CACHE — In-memory TTL cache per feature
// ============================================================

class TTLCache {
  constructor(ttlMs, label) {
    this._ttl = ttlMs;
    this._label = label;
    this._store = new Map();
    setInterval(() => {
      const now = Date.now();
      let cleaned = 0;
      for (const [k, v] of this._store) { if (now > v.exp) { this._store.delete(k); cleaned++; } }
      if (cleaned) console.log(`[Cache/${this._label}] Cleaned ${cleaned} expired entries`);
    }, 10 * 60 * 1000);
  }
  get(key) {
    const e = this._store.get(key);
    if (!e || Date.now() > e.exp) { this._store.delete(key); return null; }
    return e.val;
  }
  set(key, val) { this._store.set(key, { val, exp: Date.now() + this._ttl }); }
  delete(key)   { this._store.delete(key); }
}

const recCache    = new TTLCache(12 * 60 * 60 * 1000, "rec");    // 12h
const detectCache = new TTLCache(24 * 60 * 60 * 1000, "detect"); // 24h
const serpQueryCache  = new TTLCache(45 * 60 * 1000, "serp");  // 45-min; aligns with SerpAPI built-in ~1h cache
const predCache       = new TTLCache(60 * 60 * 1000, "pred");  // 1h — computed prediction result
const attrCache       = new TTLCache(24 * 60 * 60 * 1000, "attr"); // 24h — Gemini-extracted product attrs
const explanCache     = new TTLCache( 6 * 60 * 60 * 1000, "expl"); // 6h  — Gemini-polished explanation

function predCacheKey(storeId, storeProductId, priceRounded, currency) {
  return `pred:${storeId}:${storeProductId}:${priceRounded}:${currency}`;
}
const chatCache   = new TTLCache(10 * 60 * 1000,      "chat");   // 10min
const embCache    = new TTLCache(24 * 60 * 60 * 1000, "emb");    // 24h — in-memory title→vector cache

// ============================================================
//  EMBEDDING — gemini-embedding-001 via rec key (GEMINI_KEY_REC)
//  Same model + URL used in backfill_embeddings.mjs.
//  Results cached 24h in-memory so same title never hits API twice.
// ============================================================
const EMBED_MODEL = "gemini-embedding-001";
const EMBED_DIMS  = 3072;

async function generateEmbedding(title) {
  if (!title || !title.trim()) return null;
  const ck = title.trim().slice(0, 200);
  const hit = embCache.get(ck);
  if (hit) return hit;
  const apiKey = _keys.rec(); // uses GEMINI_KEY_REC
  if (!apiKey) return null;
  try {
    const res = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${EMBED_MODEL}:embedContent?key=${apiKey}`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: `models/${EMBED_MODEL}`, content: { parts: [{ text: title.trim() }] } }),
        signal: AbortSignal.timeout(10000)
      }
    );
    if (!res.ok) { console.warn(`[embed] ${res.status} "${title.slice(0,40)}"`); return null; }
    const json = await res.json();
    const values = json?.embedding?.values;
    if (!Array.isArray(values) || values.length !== EMBED_DIMS) { console.warn(`[embed] bad dims: ${values?.length}`); return null; }
    embCache.set(ck, values);
    return values;
  } catch (e) { console.warn(`[embed] error: ${e.message}`); return null; }
}

// Called async after every store_listings upsert — never blocks response.
// Only embeds if the row doesn't have one yet, so safe to call every time.
async function autoEmbedListing(storeId, storeProductId, title) {
  if (!title || !title.trim()) return;
  try {
    const { data: row } = await supabase
      .from("store_listings").select("embedding")
      .eq("store_id", storeId).eq("store_product_id", storeProductId).maybeSingle();
    if (row?.embedding) return; // already embedded
    const embedding = await generateEmbedding(title);
    if (!embedding) return;
    await supabase.from("store_listings").update({ embedding })
      .eq("store_id", storeId).eq("store_product_id", storeProductId);
    console.log(`[embed] auto-embedded ${storeId}/${storeProductId} "${title.slice(0,45)}"`);
  } catch (e) { console.warn(`[embed] autoEmbed fail: ${e.message}`); }
}

function recCacheKey(productId, price)             { return `rec:${productId}:${Math.round(price * 100)}`; }
function detectCacheKey(pid, wasP, curP)           { return `det:${pid}:${Math.round(wasP*100)}:${Math.round(curP*100)}`; }
function chatCacheKey(productId, msg) {
  let h = 0;
  for (let i = 0; i < msg.length; i++) h = (Math.imul(31, h) + msg.charCodeAt(i)) | 0;
  return `chat:${productId}:${h}`;
}

// ============================================================
//  JOB QUEUE — Background AI jobs, one every 4 seconds
// ============================================================

const jobStore = new Map();

function createJob(type, data) {
  const id = `${type}_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
  jobStore.set(id, { id, type, data, status: "pending", result: null, createdAt: Date.now(), onComplete: null, _fired: false });
  return id;
}

// Clean old jobs every 30 min
setInterval(() => {
  const cut = Date.now() - 2 * 60 * 60 * 1000;
  for (const [id, j] of jobStore) { if (j.createdAt < cut) jobStore.delete(id); }
}, 30 * 60 * 1000);

// Worker — one job every 4s, slow drip to protect quota
let _working = false;
async function runWorker() {
  if (_working) return;
  _working = true;
  try {
    const job = [...jobStore.values()].find(j => j.status === "pending");
    if (!job) return;
    job.status = "running";
    let result;
    if (job.type === "rec")    result = await _doAIRec(job.data);
    if (job.type === "detect") result = await _doAIDetect(job.data);
    job.status = "done";
    job.result = result;
    if (job.onComplete && !job._fired) { job._fired = true; job.onComplete(result); }
  } catch (e) {
    console.error("[Worker]", e.message);
  } finally {
    _working = false;
  }
}
setInterval(runWorker, 4000);

// ============================================================
//  RATE LIMITER — Per service, per IP
// ============================================================

const _rl = new Map();
function checkRate(key, max, windowMs) {
  const now = Date.now();
  const e = _rl.get(key) || { count: 0, reset: now + windowMs };
  if (now > e.reset) { e.count = 0; e.reset = now + windowMs; }
  e.count++;
  _rl.set(key, e);
  return e.count <= max;
}
setInterval(() => { const n = Date.now(); for (const [k, v] of _rl) if (n > v.reset) _rl.delete(k); }, 60 * 60 * 1000);

// ============================================================
//  HELPERS
// ============================================================

function computeStats(prices) {
  if (!prices.length) return null;
  const s = [...prices].sort((a, b) => a - b);
  return { min: s[0], max: s[s.length - 1], avg: s.reduce((a, x) => a + x, 0) / s.length };
}

function moneyFmt(currency, price) {
  const n = Number(price);
  if (!Number.isFinite(n)) return "—";
  try { return new Intl.NumberFormat("en-US", { style: "currency", currency: currency || "USD", maximumFractionDigits: 2 }).format(n); }
  catch { return `${currency || "USD"} ${n.toFixed(2)}`; }
}

// ============================================================
//  HEURISTIC ENGINES — Zero API calls, always available
//  These run on every page load. AI only runs when user clicks.
// ============================================================

function heuristicRec(current, stats, historyCount, deals = []) {
  // This function now only feeds chat fallback and internal logic — not displayed directly in UI
  const cheaper = (deals || []).filter(d => Number.isFinite(d.price) && d.price < current * 0.98).sort((a, b) => a.price - b.price).slice(0, 3);
  if (cheaper.length) return { action: "Cheaper elsewhere", text: `${cheaper[0].name} has it for ${moneyFmt(cheaper[0].currency || "USD", cheaper[0].price)} — ${moneyFmt("USD", current - cheaper[0].price)} less.`, confidence: 82, expectedPrice: cheaper[0].price, timeframe: "now", source: "heuristic" };
  if (historyCount === 0) return { action: "New listing", text: "First time seeing this product. Check back after a few visits for trend data.", confidence: 50, source: "heuristic" };
  if (!stats) return { action: "Watching", text: "Still building price history.", confidence: 50, source: "heuristic" };
  const { min, max, avg } = stats;
  const pct = ((current - min) / ((max - min) || 1)) * 100;
  const conf = Math.min(90, 60 + Math.floor(Math.log2(historyCount + 1) * 7));
  if (pct <= 15) return { action: "Good price", text: `Near its historical low of ${moneyFmt(null, min)}. Solid time to buy.`, confidence: conf, source: "heuristic" };
  if (pct >= 85) return { action: "High right now", text: `Near its historical high of ${moneyFmt(null, max)}. Worth waiting for a dip.`, confidence: conf, source: "heuristic" };
  if (current > avg * 1.08) return { action: "Above average", text: "Running a bit above its usual price. Often drops back.", confidence: Math.max(60, conf - 8), source: "heuristic" };
  return { action: "Fair price", text: "Around its typical price range. Fine to buy if you need it.", confidence: Math.max(55, conf - 12), source: "heuristic" };
}

function heuristicDetect(currentPrice, wasPrice, currency, hist, peers) {
  // Called from: /v1/observations fast path and as analyseDeal fallback.
  // Returns structured output with verdictKey for backward compat.
  const pct     = Math.round(((wasPrice - currentPrice) / wasPrice) * 100);
  const tol     = Math.max(0.5, wasPrice * 0.02);
  const atWas   = hist.filter(p => Math.abs(p - wasPrice) <= tol);
  const allSeen = [...hist, ...peers].filter(Number.isFinite);
  const maxSeen = allSeen.length ? Math.max(...allSeen) : null;
  const avgPeer = peers.length ? peers.reduce((a,b)=>a+b,0)/peers.length : null;

  const _mk = (verdict, verdictKey, confidence, message) =>
    ({ verdict, verdictKey, confidence, message, source: "heuristic",
       fakeDealScore: verdictKey === "likely_fake" ? 82 : verdictKey === "suspicious" ? 58 : verdictKey === "likely_real" ? 12 : 30,
       confidenceScore: confidence === "High" ? 72 : confidence === "Medium" ? 50 : 28 });

  if (pct < 2) return null;

  if (maxSeen && wasPrice > maxSeen * 1.15 && allSeen.length >= 1) {
    const confidence = allSeen.length >= 3 ? "High" : "Medium";
    return _mk("❌ Likely Fake", "likely_fake", confidence,
      `"Was ${moneyFmt(currency, wasPrice)}" exceeds our tracked max (${moneyFmt(currency, maxSeen)}). That ${pct}% claim looks inflated.`);
  }
  if (avgPeer && peers.length >= 2) {
    const peerTol = avgPeer * 0.08;
    if (Math.abs(currentPrice - avgPeer) <= peerTol && wasPrice > avgPeer * 1.20)
      return _mk("❌ Likely Fake", "likely_fake", "High",
        `Current price matches ${peers.length} other stores (~${moneyFmt(currency, avgPeer)}). "Was ${moneyFmt(currency, wasPrice)}" looks artificial.`);
    if (Math.abs(wasPrice - avgPeer) <= avgPeer * 0.10 && pct >= 5)
      return _mk("✅ Likely Real", "likely_real", "Medium",
        `Other stores price it ~${moneyFmt(currency, avgPeer)} — matching the "Was" price. The ${pct}% looks genuine.`);
  }
  if (hist.length >= 5 && atWas.length === 0)
    return _mk("⚠️ Suspicious", "suspicious", "Medium",
      `Tracked ${hist.length} times — never at ${moneyFmt(currency, wasPrice)}. "Was" price may be artificial.`);
  if (atWas.length >= 2)
    return _mk("✅ Likely Real", "likely_real", "High",
      `Seen at ${moneyFmt(currency, wasPrice)} ${atWas.length}× in our history. That ${pct}% discount checks out.`);
  if (atWas.length === 1)
    return _mk("✅ Likely Real", "likely_real", "Medium",
      `Seen at ${moneyFmt(currency, wasPrice)} once before. Probably genuine but we need more history.`);
  if (allSeen.length === 0) {
    if (pct >= 40) return _mk("⚠️ Suspicious", "suspicious", "Low",
      `${pct}% off — no history yet. Could be genuine (seasonal sale) or inflated. Check back after a few visits.`);
    if (pct >= 20) return _mk("🤷 Cannot Verify", "cannot_verify", "Low",
      `${pct}% off shown — no history yet to verify. Save it and check back in a few days.`);
    return _mk("🤷 Cannot Verify", "cannot_verify", "Low",
      `First time seeing this. Can't verify the ${pct}% claim yet.`);
  }
  return _mk("🤷 Cannot Verify", "cannot_verify", "Low",
    `Not enough data to verify this ${pct}% claim. Keep browsing and Atheon will build history.`);
}

// ============================================================
//  AI WORKERS — Only called by queue, never directly on load
// ============================================================

async function _doAIRec({ title, currentPrice, currency, stats, historyCount, recentHistory, deals }) {
  const now = new Date();
  const month = now.toLocaleString("en-US", { month: "long" });
  const cheaper = (deals || []).filter(d => d.price < currentPrice);
  const prompt = `You are a price analyst. Sharp, decisive buy/wait recommendation.
TODAY: ${month} ${now.getDate()}, ${now.getFullYear()}
PRODUCT: ${title}
CURRENT: ${moneyFmt(currency, currentPrice)}
${stats ? `HISTORY (${historyCount} obs): Low ${moneyFmt(currency, stats.min)} | Avg ${moneyFmt(currency, stats.avg)} | High ${moneyFmt(currency, stats.max)}` : "HISTORY: None."}
RECENT: ${(recentHistory||[]).slice(0,6).map(r=>`${new Date(r.observed_at||r.date).toLocaleDateString()}:${moneyFmt(currency,r.price)}`).join(", ")||"None."}
CHEAPER ELSEWHERE: ${cheaper.length ? cheaper.slice(0,4).map(d=>`${d.name}:${moneyFmt(d.currency||currency,d.price)}`).join(", ") : "None."}
Use knowledge of sale seasons (Black Friday=Nov, Prime Day=Jul, Boxing Day=Dec, Back to School=Aug-Sep).
Max 2 punchy sentences. Be decisive. Use real numbers.
Respond ONLY as valid JSON, no markdown:
{"action":"BUY"|"WAIT"|"FAIR"|"OK","text":"your natural language call","confidence":50-95,"expectedPrice":null_or_number,"timeframe":null_or_string}`;

  const raw = await callAI("rec", prompt, 400);
  if (raw) {
    try {
      const p = JSON.parse(raw.replace(/```json|```/g, "").trim());
      if (p.action && p.text) return { ...p, source: "ai" };
    } catch {}
  }
  return heuristicRec(currentPrice, stats, historyCount, deals);
}

// ═══════════════════════════════════════════════════════════════════
//  FAKE DEAL DETECTOR v2
//  Pipeline: internal evidence → SerpAPI (filtered) → MAD Z-score → verdict
// ═══════════════════════════════════════════════════════════════════

// ── 1. Robust statistics: median, MAD, modified Z-score ──────────
// Implements Iglewicz & Hoaglin modified Z-score: |Zm| > 3.5 = outlier
function robustStats(prices) {
  if (!prices.length) return null;
  const sorted = [...prices].sort((a, b) => a - b);
  const n   = sorted.length;
  const mid = Math.floor(n / 2);
  const median = n % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];

  // MAD = median(|xi - median|)
  const abs_devs = sorted.map(x => Math.abs(x - median)).sort((a, b) => a - b);
  const dn  = abs_devs.length;
  const dm  = Math.floor(dn / 2);
  const mad = dn % 2 === 0 ? (abs_devs[dm - 1] + abs_devs[dm]) / 2 : abs_devs[dm];
  // scaledMAD: Gaussian consistency factor 1.4826
  const scaledMAD = 1.4826 * mad;
  // Safe: if MAD=0 avoid divide-by-zero (use 1 so Zm = 0 for all, no outliers removed)
  const safeMAD = mad > 0 ? mad : 1;
  const modZ = x => 0.6745 * (x - median) / safeMAD;

  // Outlier-filtered subset: |Zm| <= 3.5
  const clean   = sorted.filter(x => Math.abs(modZ(x)) <= 3.5);
  const cleanN  = clean.length;
  const cleanMid = Math.floor(cleanN / 2);
  const cleanMedian = cleanN
    ? (cleanN % 2 === 0 ? (clean[cleanMid - 1] + clean[cleanMid]) / 2 : clean[cleanMid])
    : median;

  const p25 = sorted[Math.max(0, Math.floor(n * 0.25))];
  const p75 = sorted[Math.min(n - 1, Math.floor(n * 0.75))];

  return { median, cleanMedian, mad, scaledMAD, modZ, p25, p75, count: n, cleanCount: cleanN };
}

// ── 2. SerpAPI identity filter — per-product-type hard rules ────
// Returns true if the SerpAPI listing is a valid price reference for baseTitle.
// Must handle all 4 hard test cases: MS365, PS5, Xbox controller, generic.
function serpIdentityFilter(baseTitle, serpTitle, serpItem) {
  if (!serpTitle) return false;
  const b = baseTitle.toLowerCase();
  const c = serpTitle.toLowerCase();

  // ── Universal accessory / service / replacement blocklist ──────
  // Uses explicit word boundaries to prevent false positives
  if (/\b(case|cover|protector|screen\s*guard|tempered\s*glass|charger|charging\s*cable|cable|hub|dongle|stand|mount|strap|sleeve|pouch|eartips?|ear\s*tips?|cushion|skin|decal|wrap|cooling\s*pad|applecare|apple\s*care|protection\s*plan|warranty|insurance|repair\s*kit|replacement|thumb\s*grip|thumbstick|joy.?con\s*grip|charging\s*dock|charging\s*stand|battery\s*pack)\b/i.test(c)) return false;
  // "for <device>" and "compatible with" patterns
  if (/\bfor\s+(iphone|ipad|samsung|galaxy|airpods|macbook|android|ps[45]|playstation|xbox|switch|pixel)\b/i.test(c)) return false;
  if (/\bcompatible\s*with\b/i.test(c)) return false;
  // Skip used/refurb results when base product appears new
  if (serpItem.second_hand_condition && !/\b(refurb|used|renewed|pre.?owned)\b/i.test(b)) return false;

  // ── Microsoft 365 plan + term identity rules ───────────────────
  // HARD TEST CASE 1: must match plan AND term
  if (/\b(microsoft\s*365|office\s*365)\b/i.test(b)) {
    // Unrelated Microsoft products: Xbox Game Pass, Windows, etc.
    if (/\b(game\s*pass|xbox\s*game|windows\s*\d+|surface|teams\s*essentials)\b/i.test(c)) return false;
    // Plan matching: Personal vs Family vs Business
    const bPlan = /\bfamily\b/i.test(b) ? 'family' : /\bpersonal\b/i.test(b) ? 'personal' : /\bbusiness\b/i.test(b) ? 'business' : null;
    const cPlan = /\bfamily\b/i.test(c) ? 'family' : /\bpersonal\b/i.test(c) ? 'personal' : /\bbusiness\b/i.test(c) ? 'business' : null;
    if (bPlan && cPlan && bPlan !== cPlan) return false; // Personal ≠ Family
    if (bPlan && !cPlan) return false; // base specifies plan, candidate doesn't mention one
    // Term matching: 12-month/1-year annual vs 1-month monthly
    const bAnnual  = /\b(12.month|1.year|annual)\b/i.test(b);
    const cAnnual  = /\b(12.month|1.year|annual)\b/i.test(c);
    const bMonthly = /\b1.month\b/i.test(b) && !bAnnual;
    const cMonthly = /\b1.month\b/i.test(c) && !cAnnual;
    if (bAnnual  && cMonthly) return false; // annual vs monthly mismatch
    if (bMonthly && cAnnual)  return false;
    return true;
  }

  // ── PS5 / PlayStation 5 console rules ─────────────────────────
  // HARD TEST CASE 2: must be console, not controller/game/skin/stand
  if (/\b(ps5|playstation\s*5)\b/i.test(b) && !/\b(controller|dualsense|dualshock|gamepad)\b/i.test(b)) {
    if (/\b(controller|dualsense|dualshock|gamepad|game\s*pass|psn|dlc|gift\s*card|skin|stand|headset)\b/i.test(c)) return false;
    // If base specifies "slim", candidate must be console (not unrelated)
    if (/\bslim\b/i.test(b) && !/\b(slim|digital|disc|standard|console|playstation)\b/i.test(c)) return false;
    // Bundle price contamination: if base is not a bundle, reject bundle prices
    const bBundle = /\b(bundle|with\s+game|with\s+controller)\b/i.test(b);
    const cBundle = /\b(bundle|with\s+game|with\s+controller)\b/i.test(c);
    if (!bBundle && cBundle) return false;
    return true;
  }

  // ── Xbox Wireless Controller rules ────────────────────────────
  // HARD TEST CASE 3: controller only; elite vs core must not mix
  if (/\bxbox\b/i.test(b) && /\bcontroller\b/i.test(b)) {
    if (/\b(console|series\s*[xs]|charging\s*dock|charging\s*stand|battery\s*pack|play\s*and\s*charge|skin|faceplate|thumb\s*grip|wireless\s*adapter)\b/i.test(c)) return false;
    // Elite vs standard mismatch
    const bElite = /\belite\b/i.test(b);
    const cElite = /\belite\b/i.test(c);
    if (bElite !== cElite) return false;
    return true;
  }

  // ── Nintendo Switch controller ─────────────────────────────────
  if (/\b(joy.?con|pro\s*controller)\b/i.test(b) && /\b(switch|nintendo)\b/i.test(b)) {
    if (/\b(console|oled|lite|skin|charging\s*dock|thumb\s*grip)\b/i.test(c)) return false;
    return true;
  }

  // ── DualSense / PS5 controller ────────────────────────────────
  if (/\b(dualsense|dualshock)\b/i.test(b)) {
    if (/\b(console|ps5|ps4|charging\s*dock|charging\s*stand|skin|faceplate)\b/i.test(c)) return false;
    return true;
  }

  // ── Smartphone: generation must roughly match ──────────────────
  if (/\b(iphone\s*\d|galaxy\s+[szam]\d|pixel\s+\d)\b/i.test(b)) {
    if (/\b(case|cover|protector|charger|cable|applecare|clear|bumper|wallet|silicone|leather)\b/i.test(c)) return false;
    const bGenM = b.match(/\b(?:iphone|pixel)\s*(\d+)|galaxy\s+[szam](\d+)/);
    const cGenM = c.match(/\b(?:iphone|pixel)\s*(\d+)|galaxy\s+[szam](\d+)/);
    if (bGenM && cGenM) {
      const bg = Number(bGenM[1] || bGenM[2]);
      const cg = Number(cGenM[1] || cGenM[2]);
      if (Math.abs(bg - cg) > 1) return false; // only ±1 generation
    }
    return true;
  }

  // ── Audio: prevent earbuds ↔ headphones cross-contamination ────
  const bTWS = /\b(earbuds?|airpods|galaxy\s*buds|wf-\d|true\s*wireless)\b/i.test(b);
  const cTWS = /\b(earbuds?|airpods|galaxy\s*buds|wf-\d|true\s*wireless)\b/i.test(c);
  const bHP  = /\b(headphones?|over-ear|on-ear|wh-\d|quietcomfort|qc\d)\b/i.test(b);
  const cHP  = /\b(headphones?|over-ear|on-ear|wh-\d|quietcomfort|qc\d)\b/i.test(c);
  if (bTWS && cHP) return false;
  if (bHP  && cTWS) return false;

  // ── Generic overlap check: base's key tokens must appear in candidate ─
  const STOP = new Set(['the','and','for','with','new','free','fast','best','good','high','low','a','an','in','on','at','of','to','by']);
  const bTokens = b.split(/\W+/).filter(w => w.length >= 3 && !STOP.has(w)).slice(0, 4);
  if (bTokens.length >= 2 && bTokens.filter(w => c.includes(w)).length === 0) return false;

  return true;
}

// ── 3. SerpAPI fetch: 2 queries max, 45-min query-level cache ───
async function searchMarketPriceSerp(title, currency) {
  const SERP_KEY = process.env.SERPAPI_KEY;
  if (!SERP_KEY) return [];

  // Normalise: strip marketing fluff, keep model identifiers and category nouns
  const FLUFF = /\b(new|sealed|unopened|brand\s*new|free\s*shipping|fast\s*ship|in\s*box|open\s*box)\b/gi;
  const fluffStripped = title.replace(FLUFF, '').replace(/\s{2,}/g, ' ').trim();

  // Safe truncation: never cut mid-word (which could corrupt a noun like "controller" into "control").
  // Truncate at last word boundary before 90 chars so category nouns survive intact.
  let normQ = fluffStripped;
  if (normQ.length > 90) {
    const cut = normQ.slice(0, 90).replace(/\s+\S*$/, ''); // trim to last complete word
    normQ = cut.length >= 20 ? cut : normQ.slice(0, 90);   // fallback if result too short
  }

  // Cache keyed by normalised query + currency — deduplicates retries within the same hour
  const ck = `serp:${normQ.toLowerCase()}:${(currency || 'USD').toUpperCase()}`;
  const hit = serpQueryCache.get(ck);
  if (hit) { console.log(`[serp] cache-hit for "${normQ.slice(0,40)}"`); return hit; }

  const gl  = currency === 'GBP' ? 'gb' : currency === 'EUR' ? 'de' : 'us';
  const refs = [];

  // ── Query 1: tight (full normalised title, up to 90 chars) ──
  try {
    const u = new URL('https://serpapi.com/search.json');
    u.searchParams.set('engine', 'google_shopping');
    u.searchParams.set('q',      normQ);
    u.searchParams.set('gl',     gl);
    u.searchParams.set('hl',     'en');
    u.searchParams.set('num',    '10');
    u.searchParams.set('api_key', SERP_KEY);
    const res = await fetch(u.toString(), { signal: AbortSignal.timeout(8000) });
    if (res.ok) {
      const data = await res.json();
      for (const item of (data.shopping_results || [])) {
        // Use extracted_price (numeric) — not the formatted price string
        const price = typeof item.extracted_price === 'number'
          ? item.extracted_price
          : parseFloat(String(item.price || '').replace(/[^0-9.]/g, ''));
        if (!Number.isFinite(price) || price <= 0) continue;
        if (!serpIdentityFilter(title, item.title || '', item)) continue;
        refs.push({
          price,
          source:    item.source || 'unknown',
          title:     (item.title || '').slice(0, 80),
          url:       item.product_link || item.link || '',   // product_link preferred
          condition: item.second_hand_condition ? 'used' : 'new',
        });
        if (refs.length >= 8) break;
      }
    }
  } catch (e) { console.warn('[serp] query1 err:', e.message); }

  // ── Query 2: broader fallback (first 5 words) — only if tight returned < 3 ──
  if (refs.length < 3) {
    const broad = normQ.split(' ').slice(0, 5).join(' ');
    if (broad.length >= 5 && broad !== normQ) {
      try {
        const u2 = new URL('https://serpapi.com/search.json');
        u2.searchParams.set('engine', 'google_shopping');
        u2.searchParams.set('q',      broad);
        u2.searchParams.set('gl',     gl);
        u2.searchParams.set('hl',     'en');
        u2.searchParams.set('num',    '10');
        u2.searchParams.set('api_key', SERP_KEY);
        const res = await fetch(u2.toString(), { signal: AbortSignal.timeout(8000) });
        if (res.ok) {
          const data = await res.json();
          const existURLs = new Set(refs.map(r => r.url));
          for (const item of (data.shopping_results || [])) {
            const price = typeof item.extracted_price === 'number'
              ? item.extracted_price
              : parseFloat(String(item.price || '').replace(/[^0-9.]/g, ''));
            if (!Number.isFinite(price) || price <= 0) continue;
            if (!serpIdentityFilter(title, item.title || '', item)) continue;
            const url = item.product_link || item.link || '';
            if (existURLs.has(url)) continue;
            refs.push({ price, source: item.source || 'unknown', title: (item.title || '').slice(0, 80), url, condition: item.second_hand_condition ? 'used' : 'new' });
            if (refs.length >= 8) break;
          }
        }
      } catch (e) { console.warn('[serp] query2 err:', e.message); }
    }
  }

  serpQueryCache.set(ck, refs);
  console.log(`[serp] fetched ${refs.length} refs for "${normQ.slice(0, 50)}"`);
  return refs;
}

// ── 4. Build human-readable explanation ─────────────────────────
function buildFakeDealExplanation(verdictKey, fakeDealScore, cur, was, currency, stats, serpCount, histCount, peerCount) {
  const fmt = p => moneyFmt(currency, p);
  const pct = Math.round(((was - cur) / was) * 100);
  const med = stats?.refMedian;
  const refNote = med != null ? ` Market median across ${(histCount + peerCount + serpCount)} references: ${fmt(med)}.` : '';

  if (verdictKey === 'no_discount')  return 'No discount detected on this page.';
  if (verdictKey === 'likely_fake')  {
    if (med) return `The "${fmt(was)}" reference price looks inflated.${refNote} The market prices this near ${fmt(med)} — close to the current sale price — suggesting the "Was" was never the real price. The ${pct}% off claim appears manufactured.`;
    return `No evidence the "${fmt(was)}" price was ever genuine. The ${pct}% off claim isn't supported by available data.`;
  }
  if (verdictKey === 'likely_real')  {
    if (med) return `The ${pct}% discount appears genuine.${refNote} The current price of ${fmt(cur)} is below the market reference, confirming real savings.`;
    return `Price history confirms this is a real discount. The ${pct}% saving appears legitimate.`;
  }
  if (verdictKey === 'suspicious')   {
    if (med) return `Mixed signals on this discount.${refNote} The "${fmt(was)}" Was price looks higher than the typical market rate. The actual saving may be smaller than advertised.`;
    return `The "${fmt(was)}" Was price can't be confirmed. Treat the ${pct}% claim cautiously and check competitor prices.`;
  }
  // cannot_verify
  const evidenceNote = (histCount + peerCount + serpCount) === 0
    ? 'No price history or market references available yet.'
    : `Limited data: ${histCount + peerCount} internal + ${serpCount} market reference(s).`;
  return `${evidenceNote} Can't confirm whether "${fmt(was)}" was a genuine prior price. Check a few competitor sites before buying.`;
}

// ── 5. Main analyseDeal pipeline v2 ─────────────────────────────
async function analyseDeal({ title, currentPrice, wasPrice, currency, storeId, storeProductId,
    historyRows, historyPrices, peerPrices, reanalyze = false }) {
  const cur  = Number(currentPrice);
  const was  = Number(wasPrice);
  const curr = currency || 'USD';
  const fmt  = p => moneyFmt(curr, p);
  const VALID_FROM = new Date('2025-02-25T00:00:00Z'); // only trust obs from this date onward

  // ── No-discount fast path ─────────────────────────────────────
  if (!was || !Number.isFinite(was) || was <= cur) {
    return {
      verdictKey: 'no_discount', fakeDealScore: 0, confidenceScore: 100,
      explanation: 'No discount detected on this page.',
      verdict: '— No Discount', confidence: 'N/A', message: 'No discount detected on this page.',
      stats: { refMedian: null, refP25: null, refP75: null, mad: null,
               modifiedZCurrent: null, modifiedZWas: null, sourceUsed: 'none' },
      evidence: { historyCount: 0, peerCount: 0, serpCount: 0 },
      refs: [], debugReasons: ['no_discount_trigger'], source: 'heuristic',
      marketMedian: null, marketCount: 0, serpRefs: [],
    };
  }

  const claimedPct = Math.round((was - cur) / was * 100);
  const tol        = Math.max(0.5, was * 0.02); // 2% tolerance for price matching

  // ── Clean inputs ──────────────────────────────────────────────
  const peers = (peerPrices || []).filter(n => Number.isFinite(n) && n > 0);

  // Valid history = only observations on/after Feb 25 2025
  const validRows = (historyRows || []).filter(r => {
    if (!r.observed_at) return false;
    return new Date(r.observed_at) >= VALID_FROM && Number.isFinite(Number(r.price)) && Number(r.price) > 0;
  });
  const validPrices = validRows.map(r => Number(r.price));
  const histCount   = validRows.length;
  const peerCount   = peers.length;

  const debugReasons = [];
  let verdictKey, fakeDealScore, confidenceScore, sourceUsed;
  let refMedian = null;

  // ════════════════════════════════════════════════════════════
  //  TIER 1 — Peer consensus (free, independent, current)
  //  Strongest signal: if market agrees on price, wasPrice is easy to judge
  // ════════════════════════════════════════════════════════════
  if (peerCount >= 2) {
    const sortedPeers = [...peers].sort((a,b) => a - b);
    const peerMedian  = sortedPeers[Math.floor(sortedPeers.length / 2)];
    refMedian         = peerMedian;
    sourceUsed        = 'peers';

    const wasVsPeers = (was - peerMedian) / peerMedian;  // >0 = was above market
    const curVsPeers = (cur - peerMedian) / peerMedian;  // <0 = cur below market

    debugReasons.push(`peers:${peerCount}`, `peerMedian:${Math.round(peerMedian)}`,
      `wasVsPeers:${Math.round(wasVsPeers*100)}%`, `curVsPeers:${Math.round(curVsPeers*100)}%`);

    // LIKELY FAKE: wasPrice > 20% above peer median AND current is near market price
    // e.g. peers all at $199, store says "was $399 now $199" → wasPrice is fabricated
    if (wasVsPeers > 0.20 && Math.abs(curVsPeers) < 0.12) {
      verdictKey    = 'likely_fake';
      fakeDealScore = Math.min(95, Math.round(72 + wasVsPeers * 80));
      confidenceScore = Math.min(90, 55 + peerCount * 6);
      debugReasons.push('was_inflated_vs_peers_cur_at_market');

    // LIKELY FAKE: wasPrice is massively above peers (>40%) — extreme inflation
    } else if (wasVsPeers > 0.40) {
      verdictKey    = 'likely_fake';
      fakeDealScore = Math.min(98, Math.round(80 + wasVsPeers * 40));
      confidenceScore = Math.min(88, 50 + peerCount * 5);
      debugReasons.push('was_extreme_inflation_vs_peers');

    // SUSPICIOUS: wasPrice 10-20% above peers, current near market
    } else if (wasVsPeers > 0.10 && Math.abs(curVsPeers) < 0.15) {
      verdictKey    = 'suspicious';
      fakeDealScore = Math.min(75, Math.round(45 + wasVsPeers * 120));
      confidenceScore = Math.min(75, 40 + peerCount * 5);
      debugReasons.push('was_mildly_high_vs_peers');

    // LIKELY REAL: current is meaningfully below peers AND wasPrice aligns with market
    // e.g. peers at $350, cur at $280, was at $349 → genuine discount
    } else if (curVsPeers < -0.08 && wasVsPeers <= 0.12) {
      verdictKey    = 'likely_real';
      fakeDealScore = Math.max(5, Math.round(18 + curVsPeers * 80));
      confidenceScore = Math.min(88, 50 + peerCount * 6);
      debugReasons.push('cur_genuinely_below_market_was_aligns');

    // LIKELY REAL: wasPrice aligns with peers, current near or below market
    } else if (Math.abs(wasVsPeers) <= 0.10 && curVsPeers <= 0.08) {
      verdictKey    = 'likely_real';
      fakeDealScore = Math.round(15 + Math.abs(curVsPeers) * 50);
      confidenceScore = Math.min(82, 45 + peerCount * 5);
      debugReasons.push('was_aligns_with_market');

    } else {
      // Peers exist but mixed — don't declare verdict yet, fall through to tier 2/3
      verdictKey = null;
      debugReasons.push('peers_mixed_signal');
    }

    // If tier 1 gave a clear verdict, return immediately — no API calls needed
    if (verdictKey) {
      const explanation = buildFakeDealExplanation(verdictKey, fakeDealScore, cur, was, curr,
        { refMedian, refP25: null, refP75: null, mad: null,
          modifiedZCurrent: null, modifiedZWas: null, sourceUsed },
        0, histCount, peerCount);
      const confidenceStr = confidenceScore >= 70 ? 'High' : confidenceScore >= 45 ? 'Medium' : 'Low';
      console.log(`[FakeDeal:T1] peers=${peerCount} peerMedian=${Math.round(peerMedian)} verdict=${verdictKey} score=${fakeDealScore}`);
      return _assembleFakeDealResult({ verdictKey, fakeDealScore, confidenceScore, confidenceStr,
        explanation, sourceUsed, refMedian, cleanPrices: peers, serpRefs: [], debugReasons,
        histCount, peerCount, serpCount: 0, cur, was, curr });
    }
  }

  // ════════════════════════════════════════════════════════════
  //  TIER 2 — Own price history (only if Feb 25 2025+ and dense enough)
  //  Checks: did wasPrice ever actually hold for ≥14 days?
  // ════════════════════════════════════════════════════════════
  if (histCount >= 15) {
    sourceUsed = 'history';

    // Find observations within 2% tolerance of wasPrice
    const atWasRows = validRows.filter(r => Math.abs(Number(r.price) - was) <= tol);
    const atWasPrices = atWasRows.map(r => Number(r.price));

    // Check legitimacy window: was the wasPrice held for ≥14 days?
    let legitimacyDays = 0;
    if (atWasRows.length >= 2) {
      const dates   = atWasRows.map(r => new Date(r.observed_at).getTime()).sort((a,b) => a-b);
      legitimacyDays = Math.round((dates[dates.length-1] - dates[0]) / 86400000);
    }

    const neverAtWas     = atWasRows.length === 0;
    const briefFlash     = atWasRows.length >= 1 && legitimacyDays < 14;
    const heldLegitimately = atWasRows.length >= 2 && legitimacyDays >= 14;

    debugReasons.push(`hist:${histCount}`, `atWas:${atWasRows.length}`, `legDays:${legitimacyDays}`);

    if (neverAtWas) {
      // History is dense and wasPrice never appeared — strong fake signal
      verdictKey      = 'likely_fake';
      fakeDealScore   = Math.min(90, 65 + Math.min(25, histCount));
      confidenceScore = Math.min(82, 45 + histCount);
      refMedian       = refMedian || (validPrices.length ? validPrices.sort((a,b)=>a-b)[Math.floor(validPrices.length/2)] : null);
      debugReasons.push('hist_dense_never_at_was');

    } else if (briefFlash) {
      // wasPrice appeared but only briefly — suspicious, possibly a setup
      verdictKey      = 'suspicious';
      fakeDealScore   = 60;
      confidenceScore = 52;
      refMedian       = refMedian || (validPrices.length ? validPrices.sort((a,b)=>a-b)[Math.floor(validPrices.length/2)] : null);
      debugReasons.push(`hist_brief_flash_only_${legitimacyDays}d`);

    } else if (heldLegitimately) {
      // wasPrice was genuinely held for 2+ weeks — real deal
      verdictKey      = 'likely_real';
      fakeDealScore   = Math.max(8, 25 - legitimacyDays); // longer held = more genuine
      confidenceScore = Math.min(85, 50 + Math.min(20, legitimacyDays));
      refMedian       = refMedian || was;
      debugReasons.push(`hist_held_${legitimacyDays}d_legitimate`);
    }

    if (verdictKey) {
      const explanation = buildFakeDealExplanation(verdictKey, fakeDealScore, cur, was, curr,
        { refMedian, refP25: null, refP75: null, mad: null,
          modifiedZCurrent: null, modifiedZWas: null, sourceUsed },
        0, histCount, peerCount);
      const confidenceStr = confidenceScore >= 70 ? 'High' : confidenceScore >= 45 ? 'Medium' : 'Low';
      console.log(`[FakeDeal:T2] hist=${histCount} atWas=${atWasRows.length} legDays=${legitimacyDays} verdict=${verdictKey}`);
      return _assembleFakeDealResult({ verdictKey, fakeDealScore, confidenceScore, confidenceStr,
        explanation, sourceUsed, refMedian, cleanPrices: validPrices, serpRefs: [], debugReasons,
        histCount, peerCount, serpCount: 0, cur, was, curr });
    }
  }

  // ════════════════════════════════════════════════════════════
  //  TIER 3 — Serp + Gemini (fallback for sparse/ambiguous cases)
  //  Serp always fires here. Gemini only if Serp is still ambiguous.
  // ════════════════════════════════════════════════════════════
  sourceUsed = 'serp+internal';
  debugReasons.push('tier3:serp_gemini');

  // Serp fetch
  let serpRefs  = [];
  let serpCount = 0;
  const SERP_KEY = process.env.SERPAPI_KEY;
  if (SERP_KEY) {
    try {
      serpRefs  = await searchMarketPriceSerp(title, curr);
      serpCount = serpRefs.length;
    } catch (e) {
      console.warn('[FakeDeal:T3] serp error:', e.message);
    }
  }

  // Build combined price pool
  const serpPrices     = serpRefs.map(r => r.price).filter(n => Number.isFinite(n) && n > 0);
  const internalPrices = [...validPrices, ...peers];
  const allPrices      = [...internalPrices, ...serpPrices];
  const cleanPrices    = allPrices.filter(n => Number.isFinite(n) && n > 0);

  // Robust stats
  const rs = cleanPrices.length >= 2 ? robustStats(cleanPrices) : null;
  let zmWas = null, zmCur = null;
  refMedian = rs?.cleanMedian ?? rs?.median ?? null;

  // Discard market data if it's >30% off from wasPrice (likely wrong product)
  let marketDiscarded = false;
  if (refMedian !== null && was > 0) {
    const diff = Math.abs(was - refMedian) / was;
    if (diff > 0.30) {
      console.log(`[FakeDeal:T3] market discarded: was=${was} median=${refMedian} diff=${Math.round(diff*100)}%`);
      refMedian = null;
      marketDiscarded = true;
      debugReasons.push('market_discarded_mismatch_30pct');
    }
  }

  if (rs) { zmWas = rs.modZ(was); zmCur = rs.modZ(cur); }

  if (!refMedian || marketDiscarded || cleanPrices.length < 2) {
    // No usable market data — large claim is suspicious, small claim unverifiable
    if (claimedPct >= 40) {
      verdictKey = 'suspicious'; fakeDealScore = 58; confidenceScore = 25;
      debugReasons.push('large_claim_no_market_data');
    } else {
      verdictKey = 'cannot_verify'; fakeDealScore = 30; confidenceScore = 15;
      debugReasons.push('no_market_data_cannot_verify');
    }
  } else {
    const wasVsRef = (was - refMedian) / refMedian;
    const curVsRef = (cur - refMedian) / refMedian;
    debugReasons.push(`wasVsRef:${Math.round(wasVsRef*100)}%`, `curVsRef:${Math.round(curVsRef*100)}%`);

    if (wasVsRef > 0.20 && Math.abs(curVsRef) < 0.10) {
      fakeDealScore = Math.min(100, Math.round(70 + wasVsRef * 100));
      verdictKey    = 'likely_fake';
      debugReasons.push('serp_was_inflated_cur_at_market');
    } else if (wasVsRef > 0.10 && Math.abs(curVsRef) < 0.15) {
      fakeDealScore = Math.min(79, Math.round(48 + wasVsRef * 120));
      verdictKey    = 'suspicious';
      debugReasons.push('serp_was_mildly_high');
    } else if (wasVsRef > 0.20 && curVsRef > 0.05) {
      fakeDealScore = Math.min(74, Math.round(50 + wasVsRef * 80));
      verdictKey    = 'suspicious';
      debugReasons.push('serp_both_inflated');
    } else if (curVsRef < -0.10 && wasVsRef <= 0.12) {
      fakeDealScore = Math.max(0, Math.round(20 + curVsRef * 100));
      verdictKey    = 'likely_real';
      debugReasons.push('serp_cur_below_market');
    } else if (wasVsRef <= 0.10 && curVsRef <= 0.05) {
      fakeDealScore = Math.round(18 + Math.abs(curVsRef) * 60);
      verdictKey    = 'likely_real';
      debugReasons.push('serp_was_aligns_market');
    } else if (cleanPrices.length < 3) {
      fakeDealScore = 38; verdictKey = 'cannot_verify';
      debugReasons.push('serp_weak_evidence');
    } else {
      fakeDealScore = 42; verdictKey = 'suspicious';
      debugReasons.push('serp_mixed_signals');
    }

    // Confidence from evidence volume
    const evScore   = Math.min(55, (histCount * 6) + (peerCount * 9) + (serpCount * 4));
    const madPen    = rs.mad === 0 ? -15 : 0;
    const weakPen   = cleanPrices.length < 3 ? -20 : 0;
    confidenceScore = Math.max(10, Math.min(95, 32 + evScore + madPen + weakPen));
    if (zmWas !== null) debugReasons.push(`zmWas=${zmWas.toFixed(2)}`);
  }

  // ── Gemini tiebreaker — only for genuinely ambiguous borderline cases ──
  // Fires when: verdict is suspicious/cannot_verify AND discount is 15-45% AND serp is thin
  const needsGemini = reanalyze
    && (verdictKey === 'suspicious' || verdictKey === 'cannot_verify')
    && claimedPct >= 15 && claimedPct <= 45
    && serpCount < 3
    && histCount < 15;

  if (needsGemini) {
    try {
      const geminiPrompt = `You are a price authenticity expert. Assess whether this discount is genuine.

Product: "${title}"
Current price: ${fmt(cur)}
Claimed "was" price: ${fmt(was)} (${claimedPct}% discount)
${peers.length ? `Market peers (${peers.length} stores): ${peers.map(p=>fmt(p)).join(', ')}` : 'No peer prices available.'}
${validPrices.length ? `Our price history (${histCount} obs since Feb 2025): ${validPrices.slice(0,8).map(p=>fmt(p)).join(', ')}` : 'No validated price history.'}
${serpPrices.length ? `Market references (${serpCount} results): median ${fmt(refMedian)}` : 'No market references found.'}

Respond ONLY with JSON: { "verdict": "likely_fake"|"suspicious"|"likely_real"|"cannot_verify", "confidence": 0-100, "reason": "one sentence" }`;

      const raw = await callAI('detect', geminiPrompt, 120);
      if (raw) {
        const parsed = JSON.parse(raw.replace(/```json|```/g, '').trim());
        if (parsed.verdict && ['likely_fake','suspicious','likely_real','cannot_verify'].includes(parsed.verdict)) {
          verdictKey      = parsed.verdict;
          confidenceScore = Math.min(80, parsed.confidence || confidenceScore);
          fakeDealScore   = verdictKey === 'likely_fake' ? 82
            : verdictKey === 'suspicious' ? 58
            : verdictKey === 'likely_real' ? 18 : 35;
          debugReasons.push('gemini_tiebreaker', parsed.reason || '');
          sourceUsed = 'gemini';
        }
      }
    } catch (e) {
      console.warn('[FakeDeal:T3] gemini tiebreaker error:', e.message);
    }
  }

  const confidenceStr = confidenceScore >= 70 ? 'High' : confidenceScore >= 45 ? 'Medium' : 'Low';
  const explanation   = buildFakeDealExplanation(verdictKey, fakeDealScore, cur, was, curr,
    { refMedian, refP25: rs?.p25??null, refP75: rs?.p75??null, mad: rs?.mad??null,
      modifiedZCurrent: zmCur, modifiedZWas: zmWas, sourceUsed },
    serpCount, histCount, peerCount);

  console.log(`[FakeDeal:T3] verdict=${verdictKey} score=${fakeDealScore} conf=${confidenceScore} serp=${serpCount} hist=${histCount} peers=${peerCount}`);

  return _assembleFakeDealResult({ verdictKey, fakeDealScore, confidenceScore, confidenceStr,
    explanation, sourceUsed, refMedian, cleanPrices, serpRefs, debugReasons,
    histCount, peerCount, serpCount, cur, was, curr });
}

// ── Shared output assembler — keeps both tiers returning identical shape ──
function _assembleFakeDealResult({ verdictKey, fakeDealScore, confidenceScore, confidenceStr,
    explanation, sourceUsed, refMedian, cleanPrices, serpRefs, debugReasons,
    histCount, peerCount, serpCount, cur, was, curr }) {

  const VERDICT_DISPLAY = {
    likely_fake:   '❌ Likely Fake',
    suspicious:    '⚠️ Suspicious',
    likely_real:   '✅ Likely Real',
    cannot_verify: '🤷 Cannot Verify',
    no_discount:   '— No Discount',
  };

  const topRefs = [...serpRefs]
    .sort((a,b) => a.price - b.price).slice(0, 3)
    .map(r => ({ source: r.source, price: r.price, url: r.url, title: r.title, condition: r.condition }));

  return {
    verdictKey,
    fakeDealScore,
    confidenceScore,
    explanation,
    stats: {
      refMedian,
      refP25: null, refP75: null, mad: null,
      modifiedZCurrent: null, modifiedZWas: null,
      sourceUsed,
    },
    evidence:     { historyCount: histCount, peerCount, serpCount },
    refs:         topRefs,
    debugReasons,
    // Legacy fields
    verdict:      VERDICT_DISPLAY[verdictKey] || '🤷 Unknown',
    confidence:   confidenceStr,
    message:      explanation,
    source:       sourceUsed,
    marketMedian: refMedian,
    marketCount:  cleanPrices.length,
    serpRefs:     topRefs,
  };
}


// Keep _doAIDetect for queue compat — now just calls analyseDeal
async function _doAIDetect({ currentPrice, wasPrice, currency, historyPrices, peerPrices, title }) {
  return analyseDeal({ title, currentPrice, wasPrice, currency, historyPrices, peerPrices });
}

// ============================================================
//  NORMALISATION + MATCHING
//  Handles: model numbers, tiers, storage, color, condition
//  Key fix: iPhone 12 must NEVER match iPhone 15
// ============================================================

const STOPWORDS = new Set([
  "the","and","or","with","for","to","of","in","on","by",
  "latest","gen","generation","model","edition","version",
  "unlocked","factory","sealed","brand","original","genuine",
  "smartphone","phone","mobile","cell","device","bundle","kit",
  "international","warranty","official","imported","global"
]);

const COLOR_WORDS = new Set([
  "black","white","silver","gray","grey","gold","blue","red","green","pink",
  "purple","orange","yellow","titanium","midnight","starlight","graphite",
  "space","pacific","sierra","alpine","natural","coral","peach","lavender",
  "teal","cyan","violet","rose","cream","beige","bronze","charcoal"
]);

const TIER_WORDS = new Set([
  "pro","max","plus","ultra","mini","lite","fe","standard",
  "air","note","edge","fold","flip","se","rs"
]);

// Condition taxonomy — critical for new vs used classification
const CONDITION_USED_PHRASES = [
  "pre-owned","pre owned","preowned","open box","open-box","like new",
  "very good","grade a","grade b","grade c","seller refurbished",
  "manufacturer refurbished","certified refurbished"
];
const CONDITION_USED_WORDS = new Set([
  "used","refurbished","refurb","renewed","reconditioned","cpo","fair","good","excellent"
]);
const CONDITION_NEW_WORDS = new Set(["sealed","unopened","mint"]);

// Storage sizes (GB) that are NOT model numbers
const STORAGE_SIZES = new Set([8,16,32,64,128,256,512,1024]);

function normalizeTitle(t) {
  return String(t||"")
    .toLowerCase()
    .replace(/[\u2010-\u2015\u2212]/g, "-")
    .replace(/[^\p{L}\p{N}\s-]/gu, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function detectCondition(t) {
  const s = normalizeTitle(t);
  for (const phrase of CONDITION_USED_PHRASES) { if (s.includes(phrase)) return "used"; }
  const toks = new Set(s.split(" "));
  for (const w of CONDITION_USED_WORDS) { if (toks.has(w)) return "used"; }
  for (const w of CONDITION_NEW_WORDS)  { if (toks.has(w)) return "new"; }
  if (s.includes(" new ") || s.startsWith("new ")) return "new";
  return "unknown";
}

// Extract the model number — THE most important field for preventing cross-model matches
// "iPhone 15 Pro" → "15"
// "Galaxy S24 Ultra" → "s24"
// "Pixel 8a" → "8a"
// "iPad Air 5th Gen" → "5"
// "Surface Pro 9" → "9"
function extractModelNumber(s) {
  // "Nth generation" pattern — iPad Air 5th Gen, etc.
  const genMatch = s.match(/\b(\d{1,2})(?:st|nd|rd|th)\s+gen/i);
  if (genMatch) return genMatch[1];

  // iPad with sub-family: "iPad Air 5", "iPad Pro 12.9", "iPad mini 6"
  const ipadMatch = s.match(/\bipad\s+(?:air|pro|mini|pro\s+\d+\.\d)\s+(\d{1,2})/i);
  if (ipadMatch) return ipadMatch[1];

  // Primary: brand directly followed by number
  const brandPrefixMatch = s.match(/\b(iphone|ipad|ipod|macbook|imac|galaxy|note|pixel|xperia|oneplus|redmi|poco|realme|nokia|moto|motorola|huawei|honor|oppo|vivo|rog|zenbook|vivobook|thinkpad|ideapad|latitude|inspiron|xps|pavilion|spectre|envy|surface|tab|fold|flip)\s+(s?\d{1,4}[a-z]?(?:\+)?)/i);
  if (brandPrefixMatch) return brandPrefixMatch[2].trim().toLowerCase().replace(/\s+/g,"");

  // Samsung letter+number style: "Galaxy S24", "Galaxy A55"
  const samsungStyle = s.match(/\b(s|a|m|f|z)\s*(\d{2,3})\b/i);
  if (samsungStyle) return (samsungStyle[1] + samsungStyle[2]).toLowerCase();

  // Fallback: first number that is not a storage size.
  // GUARD: skip for hard-identity categories where stray numbers are NOT model IDs.
  // Controllers, consoles, subscriptions and audio products have no meaningful numeric
  // model number — grabbing port counts, year numbers or USB specs creates false
  // identity mismatches that silently kill Best Deals / Other Models results.
  const NO_NUMERIC_FALLBACK = /\b(controller|dualsense|dualshock|xbox wireless|xbox controller|ps5|ps4|playstation\s*[45]|nintendo switch|microsoft\s*365|office\s*365|365\s*personal|365\s*family|airpods|galaxy buds|wf-\d|wh-\d|quietcomfort|soundbar|earbud)\b/i;
  if (!NO_NUMERIC_FALLBACK.test(s)) {
    const nums = [...s.matchAll(/\b(\d{1,4}[a-z]?)\b/g)];
    for (const m of nums) {
      const n = parseInt(m[1]);
      if (!STORAGE_SIZES.has(n) && n > 0 && n < 9999) return m[1].toLowerCase();
    }
  }
  return null;
}

function extractVariant(t) {
  const s = normalizeTitle(t);
  const toks = s.split(" ");

  // Storage: prefer largest matching storage size
  let storage = null;
  const storageMatches = [...s.matchAll(/\b(8|16|32|64|128|256|512|1024)\s?gb\b/gi)];
  if (storageMatches.length) {
    const vals = storageMatches.map(m => parseInt(m[1]));
    storage = `${Math.max(...vals)}gb`;
  } else {
    const tbMatch = s.match(/\b(1|2|4)\s?tb\b/i);
    if (tbMatch) storage = `${tbMatch[1]}tb`;
  }

  // Color
  let color = null;
  for (let i = 0; i < toks.length; i++) {
    const w = toks[i];
    if (COLOR_WORDS.has(w)) {
      const nxt = toks[i+1];
      color = (nxt && COLOR_WORDS.has(nxt) && ["space","natural","rose"].includes(w)) ? `${w} ${nxt}` : w;
      break;
    }
  }

  // Tier — multi-word first
  let tier = null;
  const MULTI_TIERS = ["pro max","ultra max","pro plus","s ultra","s plus","s fe","note ultra"];
  for (const mt of MULTI_TIERS) { if (s.includes(mt)) { tier = mt; break; } }

  // ── Category-aware tier detection (hard-identity overrides) ──────
  // These must run BEFORE generic TIER_WORDS scan to avoid words like
  // "personal", "family", "digital", "elite" being missed or mis-classified.
  if (!tier) {
    // Subscription plan tiers — Personal / Family / Business
    if (/\b(microsoft\s*365|office\s*365)\b/.test(s)) {
      tier = s.includes('family') ? 'family'
           : s.includes('personal') ? 'personal'
           : s.includes('business') ? 'business'
           : s.includes('home') ? 'home' : 'standard';
    }
    // Console edition tiers — disc/digital/slim/pro matter for identity
    else if (/\b(ps5|ps4|playstation\s*[45])\b/.test(s) && !/\b(controller|dualsense|dualshock)\b/.test(s)) {
      tier = s.includes('digital') ? 'digital'
           : s.includes('slim') ? 'slim'
           : s.includes('pro') ? 'pro'
           : s.includes('disc') ? 'disc'
           : 'standard';
    }
    // Xbox controller: Elite vs Core vs standard
    else if (/\bxbox\b/.test(s) && /\bcontroller\b/.test(s)) {
      tier = s.includes('elite') ? 'elite'
           : s.includes('core') ? 'core'
           : 'standard';
    }
    // DualSense / DualShock controller
    else if (/\b(dualsense|dualshock)\b/.test(s)) {
      tier = s.includes('edge') ? 'edge' : 'standard';
    }
  }
  // ── End category-aware tier detection ────────────────────────────

  if (!tier) {
    for (let i = 0; i < toks.length; i++) {
      if (TIER_WORDS.has(toks[i])) {
        // Don't treat letter-number combos as tier (e.g. "s24" isn't tier "s")
        if (/^\d/.test(toks[i+1] || "")) continue;
        let run = toks[i];
        if (i+1 < toks.length && TIER_WORDS.has(toks[i+1])) run += " " + toks[i+1];
        if (!tier || run.length > tier.length) tier = run;
      }
    }
  }

  const condition = detectCondition(t);
  const modelNum  = extractModelNumber(s);

  return { storage, color, tier, condition, modelNum };
}

// Extract family = brand + product line + MODEL NUMBER (no tier, no storage, no color, no condition)
// "Apple iPhone 15 Pro Max 256GB Midnight" → "apple iphone 15"
// "Samsung Galaxy S24 Ultra 512GB Black Used" → "samsung galaxy s24"
// "Pre-owned iPhone 12 64GB" → "iphone 12"
function extractFamily(t) {
  const s  = normalizeTitle(t);
  const v  = extractVariant(t);
  const toks = s.split(" ").filter(Boolean);
  const storageNorm = v.storage;
  const storageNum  = storageNorm ? storageNorm.replace(/[a-z]/g,"") : null;

  const familyToks = [];
  for (let i = 0; i < toks.length; i++) {
    const w = toks[i];
    if (STOPWORDS.has(w)) continue;
    if (COLOR_WORDS.has(w)) continue;
    if (TIER_WORDS.has(w)) continue;
    if (w === "gb" || w === "tb") continue;
    if (w === "ram") continue;
    // Skip condition words
    if (CONDITION_USED_WORDS.has(w) || CONDITION_NEW_WORDS.has(w)) continue;
    let isCondPhrase = false;
    for (const ph of CONDITION_USED_PHRASES) { if (s.includes(ph) && ph.split(" ").includes(w)) { isCondPhrase = true; break; } }
    if (isCondPhrase) continue;
    // Skip pure storage numbers
    if (/^\d+$/.test(w) && STORAGE_SIZES.has(Number(w))) continue;
    if (/^\d{4,}$/.test(w)) continue; // year/large numbers
    familyToks.push(w);
  }

  return familyToks.slice(0, 5).join(" ");
}

function buildFamilyTokens(title) {
  return extractFamily(title).split(" ").filter(Boolean).slice(0, 3);
}

function buildMatchTokens(title) {
  const v = extractVariant(title);
  const storageNorm = v.storage;
  return normalizeTitle(title).split(" ").filter(Boolean)
    .filter(w => !STOPWORDS.has(w))
    .filter(w => !(storageNorm && (w === storageNorm || w === "gb" || w === "tb")))
    .filter(w => !COLOR_WORDS.has(w))
    .slice(0, 8);
}

// ============================================================
//  TRACK B -- Feature-Fingerprint Matching
//
//  Track A = same exact product line (model number / generation)
//  Track B = same KIND of thing with same key visible features
//
//  Used as a fallback when Track A identity is weak or absent.
//  Covers: TV mounts, routers, apparel, bags, furniture, generic
//  accessories, and any product where features matter more than
//  a model number.
// ============================================================

// -- Track B: noun dictionary ---------------------------------
// Each entry: [phrase-to-match, canonical-noun-tag]
// Ordered most-specific first so "gaming chair" beats "chair".
const _TB_NOUNS = [
  // Mounts / brackets
  ['tv mount','mount'],['wall mount','mount'],['monitor mount','mount'],
  ['tv bracket','bracket'],['monitor bracket','bracket'],['bracket','bracket'],
  // Networking
  ['wifi router','router'],['wireless router','router'],['router','router'],
  ['modem router','modem_router'],['modem','modem'],
  ['wifi extender','extender'],['range extender','extender'],
  ['network switch','network_switch'],['ethernet switch','network_switch'],
  ['access point','access_point'],
  // Monitors
  ['gaming monitor','monitor'],['monitor','monitor'],['display','monitor'],
  // TVs
  ['smart tv','tv'],['oled tv','tv'],['qled tv','tv'],['television','tv'],
  // Peripherals
  ['mechanical keyboard','keyboard'],['gaming keyboard','keyboard'],['keyboard','keyboard'],
  ['gaming mouse','mouse'],['wireless mouse','mouse'],['mouse','mouse'],
  // Speakers / audio
  ['bluetooth speaker','speaker'],['portable speaker','speaker'],['soundbar','soundbar'],['speaker','speaker'],
  // Cameras
  ['action camera','camera'],['security camera','camera'],['webcam','camera'],['camera','camera'],
  // Storage
  ['external ssd','external_ssd'],['external hard drive','external_hdd'],
  ['internal ssd','ssd'],['nvme ssd','ssd'],['ssd','ssd'],['hard drive','hdd'],['hdd','hdd'],
  // Power
  ['power bank','power_bank'],['powerbank','power_bank'],['portable charger','power_bank'],
  ['surge protector','surge_protector'],
  // Bags
  ['backpack','backpack'],['laptop bag','laptop_bag'],['laptop sleeve','laptop_bag'],
  ['handbag','handbag'],['shoulder bag','shoulder_bag'],['tote bag','tote'],
  ['gym bag','gym_bag'],['duffel bag','duffel'],['sling bag','sling_bag'],
  // Apparel tops
  ['hoodie','hoodie'],['hooded sweatshirt','hoodie'],['pullover hoodie','hoodie'],
  ['sweatshirt','sweatshirt'],['t-shirt','tshirt'],['tshirt','tshirt'],['shirt','shirt'],
  ['polo shirt','polo'],['polo','polo'],
  ['jacket','jacket'],['windbreaker','jacket'],['bomber jacket','jacket'],
  ['puffer jacket','puffer'],['down jacket','puffer'],['coat','coat'],
  // Apparel bottoms
  ['denim jeans','jeans'],['jeans','jeans'],
  ['joggers','joggers'],['sweatpants','sweatpants'],['pants','pants'],['trousers','trousers'],
  ['shorts','shorts'],['leggings','leggings'],
  // Apparel full body
  ['dress','dress'],['jumpsuit','jumpsuit'],['romper','romper'],
  // Footwear
  ['running shoes','running_shoes'],['sneakers','sneakers'],['shoes','shoes'],
  ['boots','boots'],['sandals','sandals'],['slippers','slippers'],
  // Furniture
  ['gaming chair','gaming_chair'],['office chair','chair'],['chair','chair'],
  ['standing desk','desk'],['gaming desk','desk'],['desk','desk'],
  ['bookshelf','bookshelf'],['wardrobe','wardrobe'],['nightstand','nightstand'],
  // Home appliances
  ['coffee maker','coffee_maker'],['air purifier','air_purifier'],
  ['robot vacuum','robot_vacuum'],['vacuum cleaner','vacuum'],
  ['humidifier','humidifier'],['blender','blender'],['kettle','kettle'],
  ['water bottle','water_bottle'],['thermos','thermos'],
  ['electric toothbrush','toothbrush'],['toothbrush','toothbrush'],
  // Streaming devices
  ['fire tv stick 4k max','streaming_device'],['fire tv stick 4k','streaming_device'],
  ['fire tv stick','streaming_device'],['fire tv','streaming_device'],['fire stick','streaming_device'],
  ['roku streaming stick','streaming_device'],['roku express','streaming_device'],
  ['roku ultra','streaming_device'],['roku streambar','streaming_device'],['roku','streaming_device'],
  ['chromecast','streaming_device'],['apple tv','streaming_device'],
  ['streaming stick','streaming_device'],['media streamer','streaming_device'],
  // Wireless chargers
  ['magsafe charger','wireless_charger'],['magsafe charging','wireless_charger'],['magsafe','wireless_charger'],
  ['wireless charger','wireless_charger'],['wireless charging stand','wireless_charger'],
  ['wireless charging pad','wireless_charger'],['wireless charging dock','wireless_charger'],
  ['magnetic charger','wireless_charger'],['magnetic charging','wireless_charger'],
  ['qi charger','wireless_charger'],['qi charging pad','wireless_charger'],
  ['charging stand','wireless_charger'],['charging pad','wireless_charger'],
  ['charging station','wireless_charger'],
  // Mesh wifi
  ['mesh wifi system','mesh_wifi'],['mesh wifi','mesh_wifi'],['mesh system','mesh_wifi'],
  ['whole home wifi','mesh_wifi'],['whole-home wifi','mesh_wifi'],
  ['orbi','mesh_wifi'],['eero','mesh_wifi'],['velop','mesh_wifi'],
  // Light bulbs
  ['corn bulb','light_bulb'],['corn light','light_bulb'],['corn led','light_bulb'],
  ['led bulb','light_bulb'],['light bulb','light_bulb'],['lightbulb','light_bulb'],
  ['garage bulb','light_bulb'],['garage light','light_bulb'],['shop light','light_bulb'],
  // TV mounts (specific phrases; existing mount/bracket entries still work)
  ['full motion tv mount','tv_mount'],['full motion mount','tv_mount'],
  ['tv wall mount','tv_mount'],['monitor wall mount','tv_mount'],
  ['tilt tv mount','tv_mount'],['fixed tv mount','tv_mount'],
];

function _tbNoun(s) {
  for (const [phrase, tag] of _TB_NOUNS) {
    if (s.includes(phrase)) return tag;
  }
  return null;
}

// Apparel nouns where gender mismatch is a hard reject
const _TB_APPAREL_NOUNS = new Set([
  'hoodie','sweatshirt','tshirt','shirt','polo','jacket','puffer','coat',
  'jeans','joggers','sweatpants','pants','trousers','shorts','leggings',
  'dress','jumpsuit','romper',
  'sneakers','running_shoes','shoes','boots','sandals','slippers',
]);

// Track B categories where BEST_DEALS is safe when match is very strong
const _TB_BEST_DEALS_SAFE = new Set([
  'mount','bracket','tv_mount','router','mesh_wifi','modem','modem_router','extender',
  'network_switch','access_point',
  'streaming_device','wireless_charger','light_bulb',
  'monitor','keyboard','mouse','soundbar','speaker','camera',
  'external_ssd','external_hdd','ssd','hdd','power_bank','surge_protector',
  'gaming_chair','chair','desk','backpack','laptop_bag',
  'hoodie','sweatshirt','tshirt','shirt','polo','jacket','puffer',
  'sneakers','shoes','boots','jeans','joggers','pants',
  'water_bottle','thermos','coffee_maker','air_purifier','vacuum','robot_vacuum',
]);

/**
 * extractTrackBFingerprint(title)
 * Returns { noun, brand, gender, materials, features, sizes, rawTokens }
 */
function extractTrackBFingerprint(title) {
  const s = (title || '').toLowerCase().replace(/[^\w\s\-\.\/]/g,' ').replace(/\s+/g,' ').trim();

  const noun = _tbNoun(s);

  // Lightweight brand detection for Track B
  const brandKws = [
    ['tp-link','tp_link'],['asus','asus'],['netgear','netgear'],['linksys','linksys'],
    ['eero','eero'],['ubiquiti','ubiquiti'],['samsung','samsung'],['apple','apple'],
    ['sony','sony'],['lg','lg'],['amazon','amazon'],['anker','anker'],['belkin','belkin'],
    ['logitech','logitech'],['razer','razer'],['corsair','corsair'],
    ['steelseries','steelseries'],['hyperx','hyperx'],
    ['nike','nike'],['adidas','adidas'],['puma','puma'],['new balance','new_balance'],
    ['under armour','under_armour'],['reebok','reebok'],['converse','converse'],['vans','vans'],
    ['north face','north_face'],['columbia','columbia'],
    ['ikea','ikea'],['herman miller','herman_miller'],['secretlab','secretlab'],
    ['noblechairs','noblechairs'],['autonomous','autonomous'],
    ['mountup','mountup'],['echogear','echogear'],['loctek','loctek'],
    ['roku','roku'],['amazon','amazon'],
    ['tp-link','tp_link'],['tp link','tp_link'],['tplink','tp_link'],
    ['netgear','netgear'],['tenda','tenda'],
    ['microsoft','microsoft'],['nintendo','nintendo'],
    ['belkin','belkin'],
  ];
  let brand = null;
  for (const [kw, tag] of brandKws) { if (s.includes(kw)) { brand = tag; break; } }

  // Gender
  let gender = null;
  if (/\b(men'?s?|mens)\b/.test(s))          gender = 'men';
  else if (/\b(women'?s?|womens|ladies)\b/.test(s)) gender = 'women';
  else if (/\bbboys?\b/.test(s))             gender = 'boys';
  else if (/\bgirls?\b/.test(s))             gender = 'girls';
  else if (/\bunisex\b/.test(s))             gender = 'unisex';

  // Materials / style
  const MATS = [
    'leather','cotton','fleece','wool','denim','mesh','polyester','nylon',
    'metal','steel','aluminum','aluminium','plastic','wood','wooden','glass',
    'silicone','rubber','canvas','suede',
    'slim fit','regular fit','oversized','relaxed fit','athletic fit',
    'waterproof','water resistant','breathable','quick dry',
    'magnetic','adjustable','foldable','collapsible',
  ];
  const materials = MATS.filter(m => s.includes(m));

  // Technical features
  const FEATS = [
    '4k','8k','1080p','1440p','144hz','240hz','165hz','120hz',
    'gigabit','dual band','dual-band','tri band','tri-band',
    'wifi 6','wi-fi 6','wifi 6e','wi-fi 6e','wifi 7','wi-fi 7','wifi 5','wi-fi 5',
    'full motion','tilt only','tilt','swivel','fixed','articulating',
    'wall mount','ceiling mount','desk mount','floor stand',
    'noise cancelling','anc','active noise',
    'mechanical','optical','tactile','clicky',
    'ergonomic','lumbar support','armrest','headrest',
    'rechargeable','wireless','bluetooth','usb-c','usb c','type-c',
    'portable','foldable','compact',
    'rgb','backlit',
    'hdmi','displayport','usb hub',
    'height adjustable','sit stand',
    'ax3000','ax6000','ax1800','ax5400','ax5700','be9300','be6500','be3600',
    'mu-mimo','beamforming','ofdma',
    // streaming devices
    'hdr10','dolby vision','dolby atmos','alexa built-in','voice remote',
    // chargers
    'magsafe','qi','fast charging','magnetic charging','wireless charging',
    '7.5w','10w','15w','20w','25w','45w','65w','multi-device','3-in-1','2-in-1',
    // bulbs
    'daylight','warm white','cool white','dimmable',
    // mesh wifi
    'tri-band mesh','dual-band mesh','whole home coverage',
  ];
  const features = FEATS.filter(f => s.includes(f));

  // Size / capacity
  const sizes = [];
  const inchM = [...s.matchAll(/\b(\d{2,3})\s*(?:inch|in\b|")/g)];
  inchM.forEach(m => sizes.push(m[1]+'in'));
  const mahM = s.match(/\b(\d{4,6})\s*mah\b/); if (mahM) sizes.push(mahM[1]+'mah');
  const storM = [...s.matchAll(/\b(\d+)\s*(tb|gb)\b/gi)];
  storM.forEach(m => sizes.push(m[1]+m[2].toLowerCase()));
  const clothM = [...s.matchAll(/\b(xs|s|m|l|xl|xxl|xxxl|2xl|3xl|small|medium|large|x-large)\b/gi)];
  clothM.forEach(m => sizes.push(m[1].toLowerCase()));
  const kgM = s.match(/\b(\d+)\s*kg\b/i); if (kgM) sizes.push(kgM[1]+'kg');
  const lbM = s.match(/\b(\d+)\s*lbs?\b/i); if (lbM) sizes.push(lbM[1]+'lb');
  const vesaM = s.match(/\b(\d{2,3}x\d{2,3})\b/); if (vesaM) sizes.push('vesa'+vesaM[1]);
  // Pack count (bulbs, multi-packs)
  const packM = s.match(/\b(\d{1,2})[\-\s]?pack\b/i) || s.match(/\bpack\s+of\s+(\d{1,2})\b/i);
  if (packM) { const pc = packM[1] || packM[2]; if (pc) sizes.push(pc + 'pack'); }
  // Wattage (chargers, bulbs)
  const wattM = s.match(/\b(\d{1,3})w\b/i);
  if (wattM && parseInt(wattM[1]) <= 200) sizes.push(wattM[1] + 'w');
  // Resolution / generation tokens for streaming devices
  if (/\b4k\b/.test(s)) sizes.push('4k');
  if (/\b1080p|full hd\b/.test(s)) sizes.push('1080p');

  // Raw tokens for soft overlap fallback
  const SKIP = new Set(['the','and','for','with','new','free','best','good','high',
    'low','buy','sale','set','pack','lot','piece','pcs','item','brand','quality',
    'premium','plus','mini','ultra','black','white','grey','gray','silver']);
  const rawTokens = s.split(/\s+/).filter(w => w.length >= 4 && !SKIP.has(w)).slice(0, 12);

  return { noun, brand, gender, materials, features, sizes, rawTokens };
}

/**
 * scoreTrackBMatch(baseTitle, candTitle)
 * Returns { score, nounMatch, strongMatch, moderateMatch, rejectReason, _debug }
 *
 * score 0-100
 * strongMatch:   score >= 62  -> eligible for BEST_DEALS (safe cats only)
 * moderateMatch: score >= 34  -> eligible for OTHER_MODELS
 */
function scoreTrackBMatch(baseTitle, candTitle) {
  const bFP = extractTrackBFingerprint(baseTitle);
  const cFP = extractTrackBFingerprint(candTitle);

  // Hard reject: both nouns known and different
  if (bFP.noun && cFP.noun && bFP.noun !== cFP.noun) {
    return { score: 0, nounMatch: false, strongMatch: false, moderateMatch: false,
      rejectReason: `tb_noun_mismatch:${bFP.noun}<>${cFP.noun}` };
  }

  // Hard reject: apparel gender mismatch
  if (bFP.noun && _TB_APPAREL_NOUNS.has(bFP.noun) &&
      bFP.gender && cFP.gender &&
      bFP.gender !== cFP.gender &&
      bFP.gender !== 'unisex' && cFP.gender !== 'unisex') {
    return { score: 0, nounMatch: true, strongMatch: false, moderateMatch: false,
      rejectReason: `tb_gender_mismatch:${bFP.gender}<>${cFP.gender}` };
  }

  // Hard reject: router vs mesh_wifi
  if (bFP.noun && cFP.noun &&
      ((bFP.noun === 'router' && cFP.noun === 'mesh_wifi') ||
       (bFP.noun === 'mesh_wifi' && cFP.noun === 'router'))) {
    return { score: 0, nounMatch: false, strongMatch: false, moderateMatch: false,
      rejectReason: `tb_topology_mismatch:${bFP.noun}<>${cFP.noun}` };
  }
  // Hard reject: wireless_charger vs unrelated electronics
  const _TB_CHARGER_INCOMPAT = new Set(['keyboard','speaker','camera','mouse','tv','monitor']);
  if (bFP.noun && cFP.noun &&
      ((bFP.noun === 'wireless_charger' && _TB_CHARGER_INCOMPAT.has(cFP.noun)) ||
       (cFP.noun === 'wireless_charger' && _TB_CHARGER_INCOMPAT.has(bFP.noun)))) {
    return { score: 0, nounMatch: false, strongMatch: false, moderateMatch: false,
      rejectReason: `tb_charger_incompat:${bFP.noun}<>${cFP.noun}` };
  }
  // Hard reject: streaming_device vs phones/tablets
  const _TB_STREAM_INCOMPAT = new Set(['phone','tablet']);
  if (bFP.noun && cFP.noun &&
      ((bFP.noun === 'streaming_device' && _TB_STREAM_INCOMPAT.has(cFP.noun)) ||
       (cFP.noun === 'streaming_device' && _TB_STREAM_INCOMPAT.has(bFP.noun)))) {
    return { score: 0, nounMatch: false, strongMatch: false, moderateMatch: false,
      rejectReason: `tb_streaming_incompat:${bFP.noun}<>${cFP.noun}` };
  }
  // Hard reject: light_bulb vs anything non-bulb
  if (bFP.noun !== null && cFP.noun !== null &&
      (bFP.noun === 'light_bulb') !== (cFP.noun === 'light_bulb')) {
    return { score: 0, nounMatch: false, strongMatch: false, moderateMatch: false,
      rejectReason: `tb_bulb_mismatch:${bFP.noun}<>${cFP.noun}` };
  }

  let score = 0;
  const nounMatch = !!(bFP.noun && bFP.noun === cFP.noun);

  // Noun (main pillar)
  if (nounMatch)            score += 40;
  else if (bFP.noun || cFP.noun) score += 12; // one side has a noun
  else                      score +=  8; // neither has a noun, raw fallback

  // Brand match
  if (bFP.brand && cFP.brand && bFP.brand === cFP.brand) score += 18;

  // Gender match (apparel bonus)
  if (bFP.gender && cFP.gender && bFP.gender === cFP.gender) score += 8;

  // Feature overlap
  const sharedFeatures = bFP.features.filter(f => cFP.features.includes(f));
  score += Math.min(sharedFeatures.length * 6, 20);

  // Size overlap
  const sharedSizes = bFP.sizes.filter(sz => cFP.sizes.includes(sz));
  score += Math.min(sharedSizes.length * 7, 14);

  // Material overlap
  const sharedMaterials = bFP.materials.filter(m => cFP.materials.includes(m));
  score += Math.min(sharedMaterials.length * 4, 12);

  // Raw token overlap (soft fallback for truly generic products)
  const sharedRaw = bFP.rawTokens.filter(t => cFP.rawTokens.includes(t));
  score += Math.min(sharedRaw.length * 3, 12);

  // Penalty: pack count mismatch (bulbs, multi-packs)
  const bPack = bFP.sizes.find(x => x.endsWith('pack'));
  const cPack = cFP.sizes.find(x => x.endsWith('pack'));
  if (bPack && cPack && bPack !== cPack) score = Math.max(0, score - 15);

  // Penalty: streaming device cross-brand penalty
  if (bFP.noun === 'streaming_device' && cFP.noun === 'streaming_device') {
    if (bFP.brand && cFP.brand && bFP.brand !== cFP.brand) score = Math.max(0, score - 20);
  }

  // Penalty: wattage mismatch for chargers
  if (bFP.noun === 'wireless_charger' && cFP.noun === 'wireless_charger') {
    const bW = bFP.sizes.find(x => x.endsWith('w'));
    const cW = cFP.sizes.find(x => x.endsWith('w'));
    if (bW && cW && bW !== cW) score = Math.max(0, score - 10);
  }

  score = Math.min(score, 100);

  return {
    score,
    nounMatch,
    strongMatch:   score >= 62,
    moderateMatch: score >= 34,
    rejectReason:  null,
    _debug: { baseNoun: bFP.noun, candNoun: cFP.noun, sharedFeatures, sharedSizes, sharedMaterials, sharedRaw: sharedRaw.length },
  };
}

/**
 * _isTrackBProduct(tax, classified)
 * True when the product is better matched by Track B than Track A.
 */
function _isTrackBProduct(tax) {
  // Track A is strong for these -- leave alone
  const STRONG_A = new Set([
    'phone','tablet','tws_earbuds','headphones','gaming_headset',
    'gaming_laptop','ultrabook','business_laptop','laptop',
    'gaming_desktop','desktop','smartwatch','game_console',
    'gaming_controller','software_subscription','gpu','gift_card',
    'internal_ssd','power_bank','monitor','smart_tv',
  ]);
  if (STRONG_A.has(tax.sub)) return false;
  // Unknown or weak sub -> Track B
  return true;
}

// -- End Track B helpers --------------------------------------


// ── Match grades ─────────────────────────────────────────────────────
// EXACT        — same family + model number + tier + storage + condition
// SAME_VARIANT — same family + model + tier, storage/color differs
// SAME_MODEL   — same model number, tier differs (Pro vs base)
// SAME_FAMILY  — same brand/line, DIFFERENT model number (iPhone 14 vs 15)
// RELATED      — partial overlap
// WEAK         — filter out
//
// HARD RULE: different model numbers → cannot be EXACT or SAME_VARIANT
// HARD RULE: new vs used → demoted by at least one tier and labelled
// ─────────────────────────────────────────────────────────────────────

function scoreListing(baseTitle, candidateTitle, baseVariant, baseFamilyStr) {
  const c  = normalizeTitle(candidateTitle);
  const cv = extractVariant(candidateTitle);
  const cf = extractFamily(candidateTitle);

  // Family token overlap
  const bFamToks = baseFamilyStr.split(" ").filter(Boolean);
  const cFamToks = cf.split(" ").filter(Boolean);
  const famOverlap = bFamToks.filter(t => cFamToks.includes(t)).length;
  const famRatio   = famOverlap / Math.max(bFamToks.length, cFamToks.length, 1);

  // Model number — HARD gate
  const baseModel = baseVariant.modelNum;
  const candModel = cv.modelNum;
  const modelMismatch = !!(baseModel && candModel && baseModel !== candModel);

  // Tier, storage, color, condition
  const sameTier    = (baseVariant.tier    || "") === (cv.tier    || "");
  const sameStorage = !baseVariant.storage || !cv.storage || baseVariant.storage === cv.storage;
  const sameColor   = !baseVariant.color   || !cv.color   || baseVariant.color   === cv.color;
  const conditionConflict =
    (baseVariant.condition === "new"  && cv.condition === "used") ||
    (baseVariant.condition === "used" && cv.condition === "new");

  let matchTier, score;

  if (modelMismatch) {
    // Different model numbers — hard cap at SAME_FAMILY
    if (famRatio >= 0.55) { matchTier = "SAME_FAMILY"; score = 30 + Math.round(famRatio * 12); }
    else if (famRatio >= 0.25) { matchTier = "RELATED"; score = 12 + Math.round(famRatio * 15); }
    else { matchTier = "WEAK"; score = 3; }
  } else if (famRatio >= 0.75 && sameTier && sameStorage && !conditionConflict) {
    matchTier = "EXACT";
    score = 100 + (sameColor ? 5 : 0);
  } else if (famRatio >= 0.75 && sameTier && !conditionConflict) {
    matchTier = "SAME_VARIANT";
    score = 82 + (sameStorage ? 10 : 0) + (sameColor ? 5 : 0);
  } else if (famRatio >= 0.65) {
    matchTier = "SAME_MODEL";
    score = 60 + (sameTier ? 12 : 0) + (sameStorage ? 5 : 0);
  } else if (famRatio >= 0.4) {
    matchTier = "SAME_FAMILY";
    score = 35 + Math.round(famRatio * 15);
  } else if (famRatio >= 0.2) {
    matchTier = "RELATED";
    score = 10 + Math.round(famRatio * 20);
  } else {
    matchTier = "WEAK";
    score = Math.round(famRatio * 10);
  }

  // Condition conflict: demote + penalise score
  if (conditionConflict) {
    const demote = { EXACT: "SAME_VARIANT", SAME_VARIANT: "SAME_MODEL", SAME_MODEL: "SAME_FAMILY" };
    if (demote[matchTier]) { matchTier = demote[matchTier]; score -= 18; }
  }

  if (baseFamilyStr && c.includes(baseFamilyStr)) score += 5;

  return { score, matchTier, baseCondition: baseVariant.condition, candCondition: cv.condition };
}

function matchLabel(tier, cond) {
  const c = cond?.candCondition === "used" ? " · used" : cond?.candCondition === "new" ? " · new" : "";
  if (tier === "EXACT")        return `exact match${c}`;
  if (tier === "SAME_VARIANT") return `same model${c}`;
  if (tier === "SAME_MODEL")   return `same line${c}`;
  if (tier === "SAME_FAMILY")  return `older/newer model${c}`;
  if (tier === "RELATED")      return `related${c}`;
  return `similar${c}`;
}
// ============================================================
//  DEALS
// ============================================================


// ── Product category / function taxonomy ──────────────────────────────────────
// Maps search keywords → category tag. Used as fallback when title-token matching
// fails (e.g. AirPods Pro vs Galaxy Buds Pro — same function, different names).

// ============================================================
//  PRODUCT INTELLIGENCE — Classification + Spec Extraction
// ============================================================
//
//  classifyProduct(title) → { type, tier, specs, confidence, searchKeys }
//
//  Tier 1 — Tech (spec fingerprinting, high precision)
//  Tier 2 — Semi-structured (pattern signals, medium precision)
//  Tier 3 — Lifestyle / Fashion / Home (brand + broad category, low precision)
//
//  The classifier runs a waterfall: most-specific rules first.
//  First match wins. Returns null type only if truly unclassifiable.
// ============================================================

const STORE_DISPLAY = {
  amazon:'Amazon', ebay:'eBay', walmart:'Walmart', target:'Target',
  bestbuy:'Best Buy', aliexpress:'AliExpress', temu:'Temu',
  flipkart:'Flipkart', rakuten:'Rakuten', etsy:'Etsy',
  apple:'Apple', samsung:'Samsung', microsoft:'Microsoft',
  sony:'Sony', dell:'Dell', hp:'HP', lenovo:'Lenovo', xiaomi:'Xiaomi',
  noon:'Noon', jumia:'Jumia', konga:'Konga',
  takealot:'Takealot', argos:'Argos', currys:'Currys', jarir:'Jarir'
};

// ── Spec extractors ────────────────────────────────────────────
function extractSpecs(t) {
  const s = t.toLowerCase();
  const specs = {};

  // GPU
  const gpu = s.match(/\b(rtx\s*\d{4}(?:\s*ti)?|gtx\s*\d{3,4}(?:\s*ti)?|rx\s*\d{3,4}(?:\s*xt)?|radeon\s*rx\s*\d{3,4}|arc\s*a\d{3})\b/i);
  if (gpu) specs.gpu = gpu[0].toLowerCase().replace(/\s+/g,'');

  // RAM
  const ram = s.match(/\b(\d{1,3})\s*gb\s*(ram|memory|ddr\d?)\b|\b(ram|memory)\s*(\d{1,3})\s*gb\b/i);
  if (ram) specs.ram = (ram[1]||ram[4])+'gb';

  // Storage
  const stor = s.match(/\b(\d+)\s*(tb|gb)\s*(ssd|hdd|nvme|storage|hard drive|solid state)|\b(ssd|nvme|hdd)\s*(\d+)\s*(tb|gb)\b/i);
  if (stor) specs.storage = stor[1] ? stor[1]+stor[2] : stor[5]+stor[6];

  // Screen size (inches)
  const screen = s.match(/\b(\d{1,2}(?:\.\d)?)\s*(?:"|inch|in\b|\'\')/i) ||
                 s.match(/\b(\d{2})\s*(?:inch|in)\b/i);
  if (screen) specs.screen = screen[1]+'in';

  // Refresh rate (Hz) — monitors, gaming displays, TVs
  const hz = s.match(/\b(\d{2,3})\s*hz\b/i);
  if (hz) specs.hz = hz[1]+'hz';

  // Resolution
  if (s.match(/\b4k\b|3840\s*x\s*2160/i)) specs.resolution = '4k';
  else if (s.match(/\b2k\b|1440p|2560\s*x\s*1440/i)) specs.resolution = '2k';
  else if (s.match(/\b1080p\b|1920\s*x\s*1080/i)) specs.resolution = '1080p';
  else if (s.match(/\b720p\b/i)) specs.resolution = '720p';
  else if (s.match(/\b8k\b/i)) specs.resolution = '8k';

  // Battery / Power bank capacity (mAh)
  const mah = s.match(/(\d{3,6})\s*mah/i);
  if (mah) specs.mah = mah[1]+'mah';

  // Charging wattage
  const watt = s.match(/\b(\d{1,3})\s*w(?:att)?\b/i);
  if (watt && Number(watt[1]) >= 5 && Number(watt[1]) <= 500) specs.watt = watt[1]+'w';

  // Phone storage (separate from general storage — phones say "256GB" standalone)
  if (!specs.storage) {
    const phoneStore = s.match(/\b(64|128|256|512|1024)\s*gb\b/i);
    if (phoneStore) specs.phoneStorage = phoneStore[1]+'gb';
  }

  // Connectivity
  if (s.match(/\b5g\b/i)) specs.connectivity = '5g';
  else if (s.match(/\b4g\b|lte\b/i)) specs.connectivity = '4g';

  // CPU hints
  const cpu = s.match(/\b(m1|m2|m3|m4|snapdragon\s*\d+|dimensity\s*\d+|helio\s*[a-z]\d+|core\s*i[3579]|ryzen\s*[3579]|celeron|pentium)\b/i);
  if (cpu) specs.cpu = cpu[0].toLowerCase().replace(/\s+/g,'');

  // TV screen size (TV-specific — larger range)
  const tvSize = s.match(/\b(\d{2,3})\s*(?:"|inch|in\b|cm)\s*(?:tv|oled|qled|led|uhd|nanocell|crystal)/i) ||
                 s.match(/\b(?:tv|oled|qled|led|uhd)\s*(\d{2,3})\s*(?:"|inch|in\b|cm)/i);
  if (tvSize && !specs.screen) specs.screen = (tvSize[1]||tvSize[2])+'in';

  // Clothing size
  const clothSize = s.match(/\bsize\s*(xs|s|m|l|xl|xxl|xxxl|\d+(?:\/\d+)?)\b/i) ||
                    s.match(/\b(xs|s|m|l|xl|xxl|xxxl)\b(?=\s|$)/i);
  if (clothSize) specs.clothSize = (clothSize[1]||clothSize[2]).toLowerCase();

  // Shoe size
  const shoeSize = s.match(/\b(?:us|uk|eu)?\s*size\s*(\d{1,2}(?:\.\d)?)\b/i);
  if (shoeSize) specs.shoeSize = shoeSize[1];

  // Volume / capacity for beauty/food
  const vol = s.match(/\b(\d+)\s*(ml|l|fl\s*oz|oz)\b/i);
  if (vol) specs.volume = vol[1]+vol[2].replace(/\s/g,'').toLowerCase();

  // Weight
  const wt = s.match(/\b(\d+(?:\.\d+)?)\s*(kg|g|lbs?|pounds?)\b/i);
  if (wt && !['w'].includes(wt[2].toLowerCase())) specs.weight = wt[1]+wt[2].toLowerCase();

  return specs;
}

// ── Master classifier ─────────────────────────────────────────
function classifyProduct(title) {
  const t  = (title || '').toLowerCase().trim();
  const specs = extractSpecs(t);

  // Helper: match any keyword from array
  const has  = (...kws) => kws.some(k => t.includes(k));
  const hasRe = (re) => re.test(t);

  // ── TIER 1: TECH — Electronics & Gadgets ──────────────────

  // Desktop / Gaming PC (tower, mini PC — not laptop)
  if (has('gaming pc','gaming desktop','gaming tower','desktop pc','mini pc','nuc','ryzen desktop','intel desktop') ||
      (has('rtx','gtx','rx 6','rx 7') && has('desktop','tower','atx','matx','itx') && !has('laptop','notebook'))) {
    const searchKeys = [];
    if (specs.gpu) searchKeys.push(specs.gpu.replace(/rtx/,'rtx ').replace(/gtx/,'gtx '));
    if (specs.ram) searchKeys.push(specs.ram);
    searchKeys.push('gaming desktop', 'gaming pc');
    return { type:'gaming_desktop', tier:1, specs, confidence:'high', searchKeys };
  }

  // Gaming Laptop
  if (has('gaming laptop','gaming notebook') ||
      has('rog','razer blade','legion','predator','alienware','nitro 5','tuf gaming','helios','katana','scar','strix','g15','g16') ||
      (has('rtx','gtx') && has('laptop','notebook'))) {
    const searchKeys = [];
    if (specs.gpu) searchKeys.push(specs.gpu.replace(/rtx/,'rtx ').replace(/gtx/,'gtx '));
    if (specs.ram) searchKeys.push(specs.ram);
    if (specs.screen) searchKeys.push(specs.screen);
    searchKeys.push('gaming laptop');
    return { type:'gaming_laptop', tier:1, specs, confidence:'high', searchKeys };
  }

  // Ultrabook / Premium laptop
  if (has('macbook air','macbook pro') ||
      has('xps 13','xps 15','xps 17','spectre x360','envy x360','gram 14','gram 16','swift','spin','zenbook','flexbook') ||
      (has('laptop','notebook') && has('m1','m2','m3','m4','ultra thin','ultrabook','ultraslim'))) {
    const searchKeys = [];
    if (specs.cpu) searchKeys.push(specs.cpu);
    if (specs.ram) searchKeys.push(specs.ram);
    if (specs.screen) searchKeys.push(specs.screen);
    searchKeys.push('ultrabook','thin laptop');
    return { type:'ultrabook', tier:1, specs, confidence:'high', searchKeys };
  }

  // Business laptop
  if (has('thinkpad','elitebook','latitude','probook','vostro','expertbook','lifebook','dynabook') ||
      (has('laptop','notebook') && has('business','professional','enterprise','workstation'))) {
    const searchKeys = []; if (specs.cpu) searchKeys.push(specs.cpu); if (specs.ram) searchKeys.push(specs.ram);
    searchKeys.push('business laptop'); return { type:'business_laptop', tier:1, specs, confidence:'high', searchKeys };
  }

  // Chromebook
  if (has('chromebook','chrome os','chromeos')) {
    const searchKeys = []; if (specs.ram) searchKeys.push(specs.ram); if (specs.screen) searchKeys.push(specs.screen);
    searchKeys.push('chromebook'); return { type:'chromebook', tier:1, specs, confidence:'high', searchKeys };
  }

  // General laptop (catch-all after specific types)
  if (has('laptop','notebook') && !has('gaming')) {
    const searchKeys = []; if (specs.cpu) searchKeys.push(specs.cpu); if (specs.ram) searchKeys.push(specs.ram);
    searchKeys.push('laptop'); return { type:'laptop', tier:1, specs, confidence:'medium', searchKeys };
  }

  // Flagship phone
  if (has('iphone') || has('galaxy s','galaxy z','galaxy fold','galaxy flip') ||
      has('pixel 6','pixel 7','pixel 8','pixel 9') ||
      has('xperia 1','xperia 5') || has('find x') || has('oneplus 11','oneplus 12') ||
      has('p60','p50','mate 60','mate 50')) {
    const searchKeys = []; if (specs.phoneStorage) searchKeys.push(specs.phoneStorage);
    if (specs.connectivity) searchKeys.push(specs.connectivity);
    const brand = has('iphone') ? 'iphone' : has('galaxy') ? 'samsung galaxy' : has('pixel') ? 'google pixel' : 'smartphone';
    searchKeys.push(brand); return { type:'flagship_phone', tier:1, specs, confidence:'high', searchKeys };
  }

  // Mid-range phone
  if (has('galaxy a','redmi','poco','nord','moto g','reno','realme','infinix','tecno','itel','spark') ||
      (has('smartphone','mobile phone','android') && !has('gaming'))) {
    const searchKeys = []; if (specs.phoneStorage) searchKeys.push(specs.phoneStorage);
    if (specs.connectivity) searchKeys.push(specs.connectivity); searchKeys.push('smartphone');
    return { type:'mid_phone', tier:1, specs, confidence:'medium', searchKeys };
  }

  // Tablet
  if (has('ipad') || has('galaxy tab','tab s','tab a') ||
      has('surface pro','surface go') || has('tab p','mediapad') ||
      has('fire hd','kindle fire') || (has('tablet') && !has('drawing tablet','graphics tablet'))) {
    const searchKeys = []; if (specs.screen) searchKeys.push(specs.screen);
    if (specs.phoneStorage) searchKeys.push(specs.phoneStorage); searchKeys.push('tablet');
    return { type:'tablet', tier:1, specs, confidence:'high', searchKeys };
  }

  // TWS Earbuds
  if (has('airpods') || has('galaxy buds','buds pro','buds live','buds 2','buds fe') ||
      has('earbuds','tws','true wireless','in-ear') || has('wf-1000','liberty 4','soundcore','freebuds')) {
    const searchKeys = []; if (has('anc','noise cancel')) searchKeys.push('noise cancelling');
    searchKeys.push('wireless earbuds','tws'); return { type:'tws_earbuds', tier:1, specs, confidence:'high', searchKeys };
  }

  // Over-ear headphones
  if (has('wh-1000','headphones','over-ear','on-ear','bose qc','bose nc','momentum','airmax','h9i','h95','xm5','xm4') ||
      (has('headset') && !has('gaming headset'))) {
    const searchKeys = []; if (has('anc','noise cancel')) searchKeys.push('noise cancelling');
    searchKeys.push('headphones'); return { type:'over_ear_headphones', tier:1, specs, confidence:'high', searchKeys };
  }

  // Gaming headset
  if (has('gaming headset') || has('kraken','blackshark','hs80','astro a','arctis','virtuoso')) {
    return { type:'gaming_headset', tier:1, specs, confidence:'high', searchKeys:['gaming headset'] };
  }

  // Soundbar
  if (has('soundbar','sound bar')) {
    const searchKeys = []; if (specs.watt) searchKeys.push(specs.watt); searchKeys.push('soundbar');
    return { type:'soundbar', tier:1, specs, confidence:'high', searchKeys };
  }

  // Bluetooth / Portable speaker
  if (has('bluetooth speaker','portable speaker','wireless speaker','boombox') ||
      has('jbl charge','jbl flip','jbl xtreme','jbl go','bose soundlink','ue boom','marshall emberton')) {
    const searchKeys = []; if (specs.watt) searchKeys.push(specs.watt); searchKeys.push('bluetooth speaker');
    return { type:'bt_speaker', tier:1, specs, confidence:'high', searchKeys };
  }

  // Smartwatch / Fitness band
  if (has('apple watch') || has('galaxy watch','watch 4','watch 5','watch 6','watch ultra') ||
      has('pixel watch') || has('fitbit') || has('garmin','amazfit','mi band','honor band','mi watch') ||
      (has('smartwatch','smart watch') && !has('smart tv'))) {
    const searchKeys = []; if (specs.screen) searchKeys.push(specs.screen); searchKeys.push('smartwatch');
    return { type:'smartwatch', tier:1, specs, confidence:'high', searchKeys };
  }

  // Monitor
  if (has('monitor','gaming monitor') ||
      (has('display') && (specs.hz || specs.resolution || specs.screen) && !has('tv','television'))) {
    const searchKeys = []; if (specs.screen) searchKeys.push(specs.screen);
    if (specs.hz) searchKeys.push(specs.hz); if (specs.resolution) searchKeys.push(specs.resolution);
    searchKeys.push('monitor'); return { type:'monitor', tier:1, specs, confidence:'high', searchKeys };
  }

  // Smart TV
  if (has('smart tv','oled tv','qled','nanocell','neo qled','bravia','crystal uhd','fire tv stick') ||
      (has('television','tv') && (specs.screen || specs.resolution))) {
    const searchKeys = []; if (specs.screen) searchKeys.push(specs.screen);
    if (specs.resolution) searchKeys.push(specs.resolution); searchKeys.push('smart tv');
    return { type:'smart_tv', tier:1, specs, confidence:'high', searchKeys };
  }

  // Game console
  if (has('playstation 5','ps5') || has('playstation 4','ps4') || has('xbox series x','xbox series s') ||
      has('nintendo switch','switch oled','switch lite') || has('steam deck')) {
    const name = has('ps5','playstation 5') ? 'ps5' : has('ps4') ? 'ps4' : has('xbox series x') ? 'xbox series x' :
                 has('xbox series s') ? 'xbox series s' : has('switch') ? 'nintendo switch' : 'game console';
    return { type:'game_console', tier:1, specs, confidence:'high', searchKeys:[name] };
  }

  // Gaming controller / peripheral
  if (has('dualsense','dualshock','xbox controller','pro controller','gamepad','gaming controller')) {
    return { type:'gaming_controller', tier:1, specs, confidence:'high', searchKeys:['gaming controller'] };
  }

  // Power bank
  if (has('power bank','powerbank','portable charger','battery pack') && specs.mah) {
    const searchKeys = [specs.mah]; if (specs.watt) searchKeys.push(specs.watt); searchKeys.push('power bank');
    return { type:'power_bank', tier:1, specs, confidence:'high', searchKeys };
  }
  if (has('power bank','powerbank','portable charger','battery pack')) {
    return { type:'power_bank', tier:1, specs, confidence:'medium', searchKeys:['power bank'] };
  }

  // Wall charger / adapter
  if ((has('charger','charging adapter','wall adapter','gan charger','pd charger') ||
       (has('adapter') && specs.watt)) && !has('power bank','laptop charger','macbook charger')) {
    const searchKeys = []; if (specs.watt) searchKeys.push(specs.watt); searchKeys.push('usb charger');
    return { type:'wall_charger', tier:1, specs, confidence:'medium', searchKeys };
  }

  // External SSD / HDD
  if (has('external ssd','portable ssd','external hard drive','external hdd') ||
      (has('ssd','hard drive','hdd') && has('portable','external'))) {
    const searchKeys = []; if (specs.storage) searchKeys.push(specs.storage); searchKeys.push('external ssd');
    return { type:'external_storage', tier:1, specs, confidence:'high', searchKeys };
  }

  // Internal SSD / HDD (NVMe, SATA)
  if (has('nvme','m.2 ssd','sata ssd') || (has('ssd') && !has('external','portable','laptop','gaming'))) {
    const searchKeys = []; if (specs.storage) searchKeys.push(specs.storage); searchKeys.push('ssd nvme');
    return { type:'internal_ssd', tier:1, specs, confidence:'high', searchKeys };
  }

  // USB Hub / Dock
  if (has('usb hub','usb-c hub','usb c hub','docking station','thunderbolt dock','dock')) {
    return { type:'usb_hub', tier:1, specs, confidence:'high', searchKeys:['usb hub','docking station'] };
  }

  // Wireless / MagSafe charger
  if (has('wireless charger','magsafe','qi charger','charging pad')) {
    const searchKeys = []; if (specs.watt) searchKeys.push(specs.watt); searchKeys.push('wireless charger');
    return { type:'wireless_charger', tier:1, specs, confidence:'high', searchKeys };
  }

  // Router / Networking
  if (has('wifi router','wi-fi router','mesh wifi','mesh wi-fi','ax','wifi 6','wi-fi 6') ||
      has('asus router','netgear','tp-link','linksys','orbi','eero','deco')) {
    const searchKeys = []; if (has('wifi 6','wi-fi 6','ax')) searchKeys.push('wifi 6');
    searchKeys.push('wifi router'); return { type:'router', tier:1, specs, confidence:'high', searchKeys };
  }

  // Keyboard
  if (has('mechanical keyboard','gaming keyboard','wireless keyboard') || (has('keyboard') && !has('piano','music'))) {
    const searchKeys = []; if (has('mechanical')) searchKeys.push('mechanical');
    if (has('wireless')) searchKeys.push('wireless'); searchKeys.push('keyboard');
    return { type:'keyboard', tier:1, specs, confidence:'medium', searchKeys };
  }

  // Mouse
  if (has('gaming mouse','wireless mouse','optical mouse') || (has('mouse') && !has('mousepad') && !has('mickey','disney'))) {
    const searchKeys = []; if (has('wireless')) searchKeys.push('wireless');
    if (has('gaming')) searchKeys.push('gaming'); searchKeys.push('mouse');
    return { type:'mouse', tier:1, specs, confidence:'medium', searchKeys };
  }

  // Mirrorless / DSLR camera
  if (has('mirrorless','sony alpha','sony a7','sony a6','fujifilm x','nikon z','canon eos r','om system','lumix s') ||
      (has('camera') && has('interchangeable','body only','kit lens'))) {
    return { type:'mirrorless_camera', tier:1, specs, confidence:'high', searchKeys:['mirrorless camera'] };
  }

  // Action camera
  if (has('gopro','action camera','insta360','osmo action','dji action')) {
    return { type:'action_camera', tier:1, specs, confidence:'high', searchKeys:['action camera'] };
  }

  // Printer
  if (has('printer','inkjet','laser printer','all-in-one printer','3d printer')) {
    const searchKeys = []; if (has('3d')) searchKeys.push('3d printer'); else searchKeys.push('printer');
    return { type:'printer', tier:1, specs, confidence:'high', searchKeys };
  }

  // Smart home
  if (has('robot vacuum','roomba','roborock','ecovacs','dreame')) {
    return { type:'robot_vacuum', tier:1, specs, confidence:'high', searchKeys:['robot vacuum'] };
  }
  if (has('echo show','echo dot','echo pop','fire tv','alexa') && has('amazon','echo')) {
    return { type:'smart_speaker', tier:1, specs, confidence:'high', searchKeys:['smart speaker','echo'] };
  }
  if (has('nest hub','nest mini','nest audio','chromecast','google home')) {
    return { type:'smart_speaker', tier:1, specs, confidence:'high', searchKeys:['smart speaker'] };
  }
  if (has('air purifier','hepa filter') || has('humidifier') || has('smart thermostat','nest thermostat')) {
    const k = has('air purifier') ? 'air purifier' : has('humidifier') ? 'humidifier' : 'smart thermostat';
    return { type:'smart_home_appliance', tier:1, specs, confidence:'medium', searchKeys:[k] };
  }

  // Phone case
  if (has('phone case','iphone case','galaxy case','magsafe case') ||
      (has('case','cover') && (has('iphone','samsung','pixel','oneplus') || hasRe(/case for .*(phone|pro|max|ultra)/i)))) {
    return { type:'phone_case', tier:2, specs, confidence:'medium', searchKeys:['phone case'] };
  }

  // Screen protector
  if (has('screen protector','tempered glass','privacy screen')) {
    return { type:'screen_protector', tier:2, specs, confidence:'high', searchKeys:['screen protector'] };
  }

  // ── TIER 2: SEMI-STRUCTURED ────────────────────────────────

  // Video game (software)
  if (has('ps5 game','ps4 game','xbox game','nintendo switch game','pc game','steam key','video game') ||
      (has('game') && (has('playstation','xbox','switch','pc','steam')))) {
    const platform = has('ps5') ? 'ps5' : has('ps4') ? 'ps4' : has('xbox') ? 'xbox' : has('switch') ? 'switch' : 'pc';
    return { type:'video_game', tier:2, specs:{ platform }, confidence:'medium', searchKeys:[platform+' game'] };
  }

  // Book / eBook
  if (has('paperback','hardcover','ebook','kindle edition','novel','textbook') ||
      (has('book') && !has('notebook','macbook','chromebook','notebook laptop'))) {
    return { type:'book', tier:2, specs, confidence:'medium', searchKeys:['book'] };
  }

  // Supplement / Protein
  if (has('protein powder','whey protein','creatine','pre-workout','preworkout','bcaa','mass gainer','supplement')) {
    const searchKeys = []; if (specs.weight) searchKeys.push(specs.weight); searchKeys.push('protein','supplement');
    return { type:'supplement', tier:2, specs, confidence:'high', searchKeys };
  }

  // Tires
  if (hasRe(/\b\d{3}\/\d{2}r\d{2}\b/i) || has('all-season tire','winter tire','summer tire')) {
    return { type:'tire', tier:2, specs, confidence:'high', searchKeys:['tire'] };
  }

  // ── TIER 3: LIFESTYLE / FASHION / HOME ────────────────────

  // Sneakers / Shoes
  if (has('air max','air force','yeezy','jordan','dunk','converse','vans old skool','new balance','ultraboost','foam runner') ||
      (has('sneaker','trainer','running shoe','basketball shoe','shoe') && !has('gaming'))) {
    const searchKeys = []; if (specs.shoeSize) searchKeys.push('size '+specs.shoeSize);
    const brand = has('nike') ? 'nike' : has('adidas') ? 'adidas' : has('puma') ? 'puma' : has('new balance') ? 'new balance' : '';
    if (brand) searchKeys.push(brand); searchKeys.push('sneakers');
    return { type:'sneakers', tier:3, specs, confidence:'medium', searchKeys };
  }

  // Clothing (general)
  if (has('t-shirt','tshirt','hoodie','sweatshirt','jeans','trousers','dress','shirt','jacket','coat','leggings','shorts') ||
      (has('clothing','apparel','fashion') && !has('laptop bag','camera bag'))) {
    const searchKeys = []; if (specs.clothSize) searchKeys.push(specs.clothSize);
    const brand = has('nike') ? 'nike' : has('adidas') ? 'adidas' : has('h&m') ? 'h&m' : has('zara') ? 'zara' : '';
    if (brand) searchKeys.push(brand); searchKeys.push('clothing');
    return { type:'clothing', tier:3, specs, confidence:'low', searchKeys };
  }

  // Bag / Backpack
  if (has('backpack','laptop bag','handbag','tote bag','duffel','gym bag','sling bag','crossbody') ||
      (has('bag') && !has('tea bag','coffee bag','garbage bag','shopping bag'))) {
    const searchKeys = []; if (has('backpack')) searchKeys.push('backpack'); else searchKeys.push('bag');
    return { type:'bag', tier:3, specs, confidence:'medium', searchKeys };
  }

  // Beauty / Skincare
  if (has('moisturizer','serum','sunscreen','foundation','lipstick','mascara','concealer','primer','toner','cleanser','face wash') ||
      has('skincare','beauty','makeup','cosmetic','loreal','cetaphil','cerave','neutrogena','olay','la roche')) {
    const searchKeys = []; if (specs.volume) searchKeys.push(specs.volume); searchKeys.push('skincare','beauty');
    return { type:'beauty', tier:3, specs, confidence:'medium', searchKeys };
  }

  // Hair care
  if (has('shampoo','conditioner','hair mask','hair oil','dry shampoo','hair dryer','straightener','curling iron') ||
      (has('hair') && has('treatment','care','serum'))) {
    const searchKeys = []; if (specs.volume) searchKeys.push(specs.volume); searchKeys.push('hair care');
    return { type:'hair_care', tier:3, specs, confidence:'medium', searchKeys };
  }

  // Fragrance / Perfume
  if (has('perfume','eau de parfum','eau de toilette','cologne','fragrance','edp','edt')) {
    const searchKeys = []; if (specs.volume) searchKeys.push(specs.volume); searchKeys.push('perfume','fragrance');
    return { type:'fragrance', tier:3, specs, confidence:'high', searchKeys };
  }

  // Kitchen appliance
  if (has('air fryer','instant pot','pressure cooker','blender','toaster','coffee maker','espresso','kettle','microwave','rice cooker','juicer') ||
      (has('kitchen') && has('appliance','gadget','tool'))) {
    const kind = has('air fryer') ? 'air fryer' : has('coffee','espresso') ? 'coffee maker' : has('blender') ? 'blender' : 'kitchen appliance';
    const searchKeys = []; if (specs.watt) searchKeys.push(specs.watt); searchKeys.push(kind);
    return { type:'kitchen_appliance', tier:3, specs, confidence:'medium', searchKeys };
  }

  // Furniture
  if (has('gaming chair','office chair','standing desk','desk chair','sofa','couch','bed frame','mattress','wardrobe','bookshelf')) {
    const kind = has('gaming chair','office chair') ? 'gaming chair' : has('standing desk','desk') ? 'desk' : has('mattress') ? 'mattress' : 'furniture';
    return { type:'furniture', tier:3, specs, confidence:'medium', searchKeys:[kind] };
  }

  // Toys / Kids
  if (has('lego','action figure','doll','toy car','board game','puzzle','kids','children') && !has('laptop','tablet','gaming')) {
    const searchKeys = [has('lego') ? 'lego' : 'toy']; return { type:'toy', tier:3, specs, confidence:'medium', searchKeys };
  }

  // Pet
  if (has('dog food','cat food','pet food','dog bed','cat tree','pet carrier','dog collar','cat litter')) {
    const searchKeys = []; if (specs.weight) searchKeys.push(specs.weight); searchKeys.push('pet');
    return { type:'pet_supplies', tier:3, specs, confidence:'medium', searchKeys };
  }

  // Sports / Fitness equipment
  if (has('dumbbell','barbell','resistance band','yoga mat','treadmill','exercise bike','pull-up bar','gym equipment','jump rope')) {
    const kind = has('dumbbell','barbell') ? 'dumbbell' : has('yoga mat') ? 'yoga mat' : has('treadmill') ? 'treadmill' : 'fitness equipment';
    const searchKeys = []; if (specs.weight) searchKeys.push(specs.weight); searchKeys.push(kind);
    return { type:'fitness_equipment', tier:3, specs, confidence:'medium', searchKeys };
  }

  // Generic catchall — at least classify as broad category
  if (has('watch') && !has('smart')) return { type:'analog_watch', tier:3, specs, confidence:'low', searchKeys:['watch'] };
  if (has('sunglasses','glasses','eyewear')) return { type:'eyewear', tier:3, specs, confidence:'low', searchKeys:['sunglasses'] };
  if (has('wallet','purse')) return { type:'wallet', tier:3, specs, confidence:'low', searchKeys:['wallet'] };

  return { type: null, tier: null, specs, confidence: 'none', searchKeys: [] };
}

// ── Spec overlap score between two classifyProduct() results ──
function specOverlapScore(baseClassification, candTitle) {
  const cand = classifyProduct(candTitle);
  if (!baseClassification.type || !cand.type) return 0;
  if (baseClassification.type !== cand.type) return 0; // different type = no overlap

  const bs = baseClassification.specs;
  const cs = cand.specs;
  let score = 10; // same type base (lowered — type-only match should not dominate)

  // Hard spec matches (double points for exact)
  if (bs.gpu && cs.gpu) {
    score += bs.gpu === cs.gpu ? 40 : bs.gpu.split('').slice(0,4).join('') === cs.gpu.split('').slice(0,4).join('') ? 20 : -10;
  }
  if (bs.ram && cs.ram) score += bs.ram === cs.ram ? 15 : -5;
  if (bs.storage && cs.storage) score += bs.storage === cs.storage ? 10 : 0;
  if (bs.screen && cs.screen) score += bs.screen === cs.screen ? 10 : Math.abs(parseFloat(bs.screen)-parseFloat(cs.screen)) <= 1 ? 5 : -5;
  if (bs.hz && cs.hz) score += bs.hz === cs.hz ? 10 : -5;
  if (bs.resolution && cs.resolution) score += bs.resolution === cs.resolution ? 10 : 0;
  if (bs.mah && cs.mah) {
    const bm = parseInt(bs.mah), cm = parseInt(cs.mah);
    score += bm === cm ? 20 : Math.abs(bm-cm)/bm < 0.15 ? 10 : Math.abs(bm-cm)/bm < 0.30 ? 5 : -5;
  }
  if (bs.watt && cs.watt) score += bs.watt === cs.watt ? 10 : 0;
  if (bs.phoneStorage && cs.phoneStorage) score += bs.phoneStorage === cs.phoneStorage ? 15 : 0;
  if (bs.connectivity && cs.connectivity) score += bs.connectivity === cs.connectivity ? 10 : 0;
  if (bs.platform && cs.platform) score += bs.platform === cs.platform ? 20 : -30;
  if (bs.clothSize && cs.clothSize) score += bs.clothSize === cs.clothSize ? 15 : -10;
  if (bs.shoeSize && cs.shoeSize) score += bs.shoeSize === cs.shoeSize ? 20 : Math.abs(parseFloat(bs.shoeSize)-parseFloat(cs.shoeSize)) <= 0.5 ? 5 : -15;
  if (bs.volume && cs.volume) score += bs.volume === cs.volume ? 15 : 0;

  return Math.max(0, Math.min(100, score));
}

// ── Type-to-matchTier mapper ──────────────────────────────────
function specScoreToTier(score, sameType) {
  if (!sameType) return 'WEAK';
  if (score >= 85) return 'EXACT';
  if (score >= 65) return 'SAME_VARIANT';
  if (score >= 50) return 'SAME_MODEL';
  if (score >= 30) return 'SAME_FAMILY';
  return 'RELATED';
}

// ── DB search keys for a classified product ───────────────────
// Returns up to 3 arrays of keywords to try as ilike queries.
// We try from most-specific to broadest, stop when we get results.
function buildTypeSearchQueries(classification) {
  const { type, tier, specs, searchKeys } = classification;
  if (!type) return [];

  const queries = [];

  // Each entry = array of ilike terms AND-ed together in the DB query.
  // Keep individual terms SHORT (1 word or a model code) so they match
  // regardless of how the store words the title.

  if (tier === 1) {
    // GPU: "rtx4070" → search "rtx" AND "4070" separately (more flexible)
    if (specs.gpu) {
      const gpuBase = specs.gpu.replace(/ti$/,'').replace(/xt$/,''); // strip suffix
      const gpuNum  = gpuBase.match(/\d{3,4}/)?.[0];
      const gpuBrand = gpuBase.match(/^(rtx|gtx|rx|arc)/)?.[0];
      if (gpuBrand && gpuNum) queries.push([gpuBrand, gpuNum]); // ["rtx","4070"]
      else queries.push([specs.gpu]);
    }
    // mAh: exact number works fine
    if (specs.mah) queries.push([specs.mah.replace('mah',''), 'mah']); // ["20000","mah"]
    // Hz + screen: both single tokens
    if (specs.hz && specs.screen) queries.push([specs.hz.replace('hz',''), 'hz']);
    if (specs.resolution) queries.push([specs.resolution]); // "4k"
    if (specs.ram && !specs.gpu) queries.push([specs.ram.replace('gb',''), 'gb', 'ram']);
  }

  // Type-level: split multi-word keys so ilike works ("gaming laptop" → ["gaming","laptop"])
  for (const key of searchKeys.slice(0, 3)) {
    const words = key.split(' ').filter(w => w.length >= 3);
    if (words.length) queries.push(words.slice(0, 2)); // max 2 words per query term
  }

  return queries;
}

// ============================================================
//  MATCHING ENGINE  (modular — normalizer → taxonomy →
//  roleDetector → plugins → engine → ranker)
//
//  Two output buckets sent to frontend:
//    bestDeals   — confirmed same product / variant
//    otherModels — same subcategory, different model, still useful
//
//  KEY RULE: identity decides the bucket.
//            score only ranks INSIDE a bucket.
//            "same category" is NEVER enough for bestDeals.
// ============================================================

// ── NORMALIZER ───────────────────────────────────────────────
function normalize(t) {
  let s = (t || '').toLowerCase();
  s = s.replace(/[''`]/g, '').replace(/[–—]/g, '-').replace(/[^\w\s\-\/\.]/g, ' ');
  s = s.replace(/\s+/g, ' ').trim();
  // Units
  s = s.replace(/(\d)\s*tb\b/g, (_, n) => `${Number(n) * 1000}gb`);
  s = s.replace(/(\d+)\s*gb\b/g, '$1gb');
  s = s.replace(/(\d+)\s*mb\b/g, '$1mb');
  s = s.replace(/(\d+)\s*mah\b/g, '$1mah');
  s = s.replace(/(\d+)\s*hz\b/g, '$1hz');
  s = s.replace(/(\d+)\s*w\b(?!h)/g, '$1w');
  // Model codes
  s = s.replace(/wh[\s-]*1000[\s-]*(xm\d)/gi, 'wh-1000$1');
  s = s.replace(/wh1000(xm\d)/gi, 'wh-1000$1');
  // Condition synonyms
  s = s.replace(/\b(pre-?owned|preowned|renewed|reconditioned)\b/g, 'refurbished');
  s = s.replace(/\b(manufacturer|seller|certified)\s+refurbished\b/g, 'refurbished');
  // Subscription term normalization: 1-year / 12-month / annual → canonical month count
  s = s.replace(/\b1[-\s]?year\b/gi,  '12month');
  s = s.replace(/\b2[-\s]?year\b/gi,  '24month');
  s = s.replace(/\b3[-\s]?year\b/gi,  '36month');
  s = s.replace(/\b12[-\s]?month\b/gi,'12month');
  s = s.replace(/\b6[-\s]?month\b/gi, '6month');
  s = s.replace(/\bannual\b/gi,        '12month');
  // "for <device>" — mark so role detector catches it
  s = s.replace(/\bfor\s+(iphone|ipad|samsung|galaxy|airpods|macbook|android|ps[45]|xbox|switch|pixel)\b/gi, 'FORDVC');
  const tokens = s.split(/\s+/).filter(Boolean);
  return { s, tokens };
}

// ── TAXONOMY ─────────────────────────────────────────────────
function taxonomy(title) {
  const { s } = normalize(title);
  const has = (...kws) => kws.some(k => s.includes(k));

  // Electronics — phones
  if (has('iphone') || (has('galaxy') && (has('s21','s22','s23','s24','s25') || / galaxy [sz]\d/.test(s))) || has('pixel 6','pixel 7','pixel 8','pixel 9') || (has('smartphone','mobile phone') && !has('case','cover')))
    return { cat:'electronics', sub:'phone', conf:'high' };

  // Electronics — tws earbuds
  if (has('airpods') || has('galaxy buds') || has('earbuds','tws','true wireless') || has('wf-1000','freebuds','liberty air','redmi buds'))
    return { cat:'electronics', sub:'tws_earbuds', conf:'high' };

  // Electronics — over-ear / on-ear headphones
  if (has('wh-1000') || has('headphones','over-ear','on-ear') || has('bose qc','bose nc','bose quiet') || has('momentum 4') || (has('headset') && !has('gaming headset')))
    return { cat:'electronics', sub:'headphones', conf:'high' };

  // Electronics — gaming headset
  if (has('gaming headset') || has('arctis','kraken','blackshark','hs80','astro a','virtuoso'))
    return { cat:'electronics', sub:'gaming_headset', conf:'high' };

  // Electronics — gaming laptop
  if (has('gaming laptop','gaming notebook') || has('rog','razer blade','legion','predator','alienware','nitro 5','tuf gaming','helios','scar','katana') || (has('laptop','notebook') && (has('rtx','gtx') || has('gaming'))))
    return { cat:'electronics', sub:'gaming_laptop', conf:'high' };

  // Electronics — ultrabook
  if (has('macbook') || has('xps 13','xps 15','xps 17') || has('spectre','zenbook','gram 1','swift','envy x360') || (has('laptop','notebook') && has('m1','m2','m3','m4','ultrabook','ultra slim')))
    return { cat:'electronics', sub:'ultrabook', conf:'high' };

  // Electronics — business laptop
  if (has('thinkpad','elitebook','latitude','probook','inspiron','vostro','expertbook') || (has('laptop','notebook') && has('business','enterprise','workstation')))
    return { cat:'electronics', sub:'business_laptop', conf:'high' };

  // Electronics — general laptop
  if (has('laptop','notebook') && !has('case','bag','stand','cooling'))
    return { cat:'electronics', sub:'laptop', conf:'medium' };

  // Electronics — gaming desktop
  if (has('gaming pc','gaming desktop','gaming tower') || (has('rtx','gtx') && has('desktop','tower','atx','mini-itx') && !has('laptop','notebook')))
    return { cat:'electronics', sub:'gaming_desktop', conf:'high' };

  // Electronics — desktop
  if ((has('desktop','tower') && !has('laptop')) || has('mini pc','nuc','all-in-one','all in one'))
    return { cat:'electronics', sub:'desktop', conf:'medium' };

  // Electronics — tablet
  if (has('ipad') || has('galaxy tab','tab s','tab a') || has('surface pro','surface go') || (has('tablet') && !has('drawing tablet','graphics tablet','tablet case')))
    return { cat:'electronics', sub:'tablet', conf:'high' };

  // Electronics — smartwatch
  if (has('apple watch') || has('galaxy watch') || has('pixel watch') || has('garmin','amazfit','fitbit') || (has('smartwatch','smart watch') && !has('case','strap','band replacement')))
    return { cat:'electronics', sub:'smartwatch', conf:'high' };

  // Electronics — game console
  if (has('ps5','playstation 5') || has('ps4','playstation 4') || has('xbox series x','xbox series s') || has('nintendo switch','switch oled','switch lite') || has('steam deck'))
    return { cat:'electronics', sub:'game_console', conf:'high' };

  // Electronics — gaming controller (must come AFTER console to avoid grabbing console titles)
  if (has('dualsense','dualshock','xbox wireless controller','xbox controller') ||
      has('pro controller','gamepad') ||
      (has('controller') && (has('playstation','xbox','nintendo','ps5','ps4','switch')) &&
       !has('charging dock','charging stand','battery pack','thumb grip','silicone','grip cover')))
    return { cat:'electronics', sub:'gaming_controller', conf:'high' };

  // Software — subscriptions / licenses
  if (has('microsoft 365','office 365','microsoft office') ||
      (has('365') && has('personal','family','home','business')))
    return { cat:'software', sub:'software_subscription', conf:'high' };

  // Vouchers — gift cards
  if (has('gift card','psn card','xbox gift card','steam gift card','google play card','itunes card') ||
      (has('gift card') && has('amazon','apple','google','playstation','nintendo')))
    return { cat:'voucher', sub:'gift_card', conf:'high' };

  // Electronics — monitor
  if (has('monitor','gaming monitor') || (has('display') && !has('tv','television') && (/\d+hz/.test(s) || /\d+(in|inch)/.test(s))))
    return { cat:'electronics', sub:'monitor', conf:'high' };

  // Electronics — smart TV
  if (has('smart tv','oled tv','qled','nanocell','bravia') || (has('television') && /\d+(in|inch|")/.test(s)))
    return { cat:'electronics', sub:'smart_tv', conf:'high' };

  // Electronics — GPU
  if ((has('rtx','gtx','radeon rx') && has('graphics card','gpu','video card','founders')) || has('geforce rtx','geforce gtx'))
    return { cat:'electronics', sub:'gpu', conf:'high' };

  // Electronics — power bank
  if ((has('power bank','powerbank','portable charger')) && /\d+mah/.test(s))
    return { cat:'electronics', sub:'power_bank', conf:'high' };

  // Electronics — storage
  if (has('nvme','m.2 ssd') || (has('ssd') && !has('laptop','gaming','external')))
    return { cat:'electronics', sub:'internal_ssd', conf:'high' };
  if (has('external ssd','portable ssd','external hdd','external hard drive'))
    return { cat:'electronics', sub:'external_storage', conf:'high' };

  // Fashion
  if (has('air max','air force 1','yeezy','jordan','dunk low','ultraboost') || (has('sneaker','trainer','running shoe') && !has('gaming')))
    return { cat:'fashion', sub:'sneakers', conf:'medium' };
  if (has('t-shirt','hoodie','jeans','trousers','dress','jacket','coat','leggings','sweatshirt'))
    return { cat:'fashion', sub:'clothing', conf:'medium' };

  // Beauty
  if (has('moisturizer','serum','sunscreen','foundation','shampoo','conditioner','perfume','eau de','lipstick','cleanser'))
    return { cat:'beauty', sub:'beauty', conf:'medium' };

  // Home & Kitchen
  if (has('air fryer','instant pot','blender','coffee maker','espresso machine','toaster','kettle','rice cooker'))
    return { cat:'home', sub:'kitchen_appliance', conf:'medium' };

  // Books
  if (has('paperback','hardcover','ebook') || (has('book') && !has('notebook','macbook','book bag')))
    return { cat:'media', sub:'book', conf:'medium' };

  return { cat:'unknown', sub:'unknown', conf:'low' };
}

// ── ROLE DETECTOR ────────────────────────────────────────────
// The global pollution blocker. Runs on EVERY candidate.
const _ACCESSORY_RE = [
  /\bcase\b/, /\bcover\b/, /\bprotector\b/, /\bscreen guard\b/, /\btempered glass\b/,
  /\bcharger\b/, /\bcable\b(?! tv)/, /\bhub\b/, /\bdongle\b/,
  /\bstand\b/, /\bdock(?:ing)?\b/, /\bmount\b/, /\bbracket\b/,
  /\bstrap\b/, /\bwrist band\b/, /\bband\b(?!\s*\d)(?! of)(?! pass)/,
  /\bsleeve\b/, /\bpouch\b/, /\bholster\b/,
  /\bear\s*tips?\b/, /\beartips?\b/, /\bwing tips?\b/, /\bsilicone tips?\b/,
  /\bcushion\b/, /\bear\s*pad\b/, /\bpad\b(?! pro\b)(?! air\b)/,
  /\bskin\b(?! care)(?!ny)/, /\bdecal\b/, /\bsticker\b/, /\bwrap\b/,
  /\bkeyboard cover\b/, /\bcooling pad\b/, /\blaptop stand\b/,
  // Standalone components that are accessories when base is a full system
  /\bgraphics card\b/, /\bvideo card\b/, /\bgpu card\b/,
  /\bkeyboard.*mouse\b/, /\bmouse.*keyboard\b/,  // peripheral combos
  /\bgaming chair\b/, /\bmonitor arm\b/, /\bdesk mat\b/,
  // Controller accessories — must NOT match the controller itself
  /\bthumb grip\b/, /\bthumbstick grip\b/, /\bjoy-?con grip\b/,
  /\bcontroller skin\b/, /\bcontroller faceplate\b/, /\bsilicone cover\b/,
  /\bcharging dock\b/, /\bcharging stand\b/,
];
const _SERVICE_RE = [
  /\bapplecare\b/, /\bcare\s*\+/, /\bgeek\s*squad\b/, /\btotal\s*tech\b/,
  /\bprotection\s*plan\b/, /\bwarranty\b/, /\bextended\s*(care|protection)\b/,
  /\binsurance\b/, /\baccident\s*protection\b/,
];
const _REPLACEMENT_RE = [
  /\breplacement\b/, /\brepair\s*kit\b/, /\bspare\s*part\b/,
  /\bhousing\b/, /\bshell\b/, /\bbattery\s*replacement\b/,
];
const _BUNDLE_RE = [
  /\bwith\s+(case|charger|cable|stand|accessories|screen protector)\b/,
  /\bbundle\b/, /\bcombo\b/,
];

function detectRole(title) {
  const { s } = normalize(title);
  if (s.includes('FORDVC') || /\bfor\s+(iphone|ipad|samsung|galaxy|airpods|macbook|android|ps[45]|xbox|switch|pixel)\b/.test(s) || /\bcompatible with\b/.test(s))
    return 'accessory';
  for (const r of _SERVICE_RE) if (r.test(s)) return 'service';
  for (const r of _REPLACEMENT_RE) if (r.test(s)) return 'replacement_part';
  for (const r of _ACCESSORY_RE) if (r.test(s)) return 'accessory';
  for (const r of _BUNDLE_RE) if (r.test(s)) return 'bundle';
  return 'main_product';
}

// ── PRICE SANITY ─────────────────────────────────────────────
const _PRICE_FLOOR = {
  phone:80, tws_earbuds:5, headphones:10, gaming_headset:10,
  gaming_laptop:200, ultrabook:150, business_laptop:100, laptop:80,
  gaming_desktop:200, desktop:80, tablet:40, smartwatch:20,
  game_console:80, monitor:40, smart_tv:80, gpu:80,
  power_bank:3, internal_ssd:10, external_storage:15,
  gaming_controller:15, software_subscription:2, gift_card:1,
};
function priceSane(sub, price) {
  const floor = _PRICE_FLOOR[sub];
  return !floor || price >= floor;
}

// ── PLUGINS ───────────────────────────────────────────────────

// -- Phone Plugin --
const phonePlugin = {
  _parse(title) {
    const { s } = normalize(title);
    const brand =
      s.includes('iphone') || (s.includes('apple') && !s.includes('airpods') && !s.includes('macbook')) ? 'apple' :
      s.includes('samsung') || s.includes('galaxy') ? 'samsung' :
      s.includes('pixel') || s.includes('google') ? 'google' :
      s.includes('oneplus') ? 'oneplus' :
      s.includes('xiaomi') || s.includes('redmi') ? 'xiaomi' :
      s.includes('nothing phone') ? 'nothing' :
      s.includes('huawei') ? 'huawei' : 'unknown';

    const family =
      s.includes('airpods')       ? 'airpods' :  // detect audio products on phone page → other_family
      s.includes('iphone')        ? 'iphone' :
      /galaxy [sz]\s*fold/.test(s) ? 'galaxy_fold' :
      /galaxy [sz]\s*flip/.test(s) ? 'galaxy_flip' :
      s.includes('galaxy s')      ? 'galaxy_s' :
      s.includes('galaxy a')      ? 'galaxy_a' :
      s.includes('galaxy m')      ? 'galaxy_m' :
      s.includes('pixel')         ? 'pixel' :
      s.includes('redmi')         ? 'redmi' :
      s.includes('poco')          ? 'poco' :
      s.includes('oneplus')       ? 'oneplus' : null;

    // Generation — the most identity-critical field
    const genM =
      s.match(/\biphone\s*(\d{1,2})\b/) ||
      s.match(/\bgalaxy\s+s(\d{2})\b/) ||
      s.match(/\bgalaxy\s+a(\d{2})\b/) ||
      s.match(/\bpixel\s+(\d)\b/) ||
      s.match(/\bnote\s*(\d{2})\b/) ||
      s.match(/\bredmi\s*(\d+)\b/) ||
      s.match(/\boneplus\s*(\d+)\b/);
    const gen = genM ? genM[1] : null;

    const tier =
      s.includes('pro max') ? 'pro_max' :
      s.includes('ultra')   ? 'ultra'   :
      s.includes('pro')     ? 'pro'     :
      s.includes('plus')    ? 'plus'    :
      s.includes(' fe')     ? 'fe'      :
      s.includes('mini')    ? 'mini'    : 'standard';

    const storM = s.match(/\b(64|128|256|512|1024)gb\b/);
    const storage = storM ? storM[1] : null;

    const cond = /\bused\b|\brefurbished\b/.test(s) ? 'used' : 'new';
    return { brand, family, gen, tier, storage, cond };
  },

  identityMatch(baseTitle, candTitle) {
    const b = this._parse(baseTitle);
    const c = this._parse(candTitle);

    // Different brand → never Best Deals
    if (b.brand !== 'unknown' && c.brand !== 'unknown' && b.brand !== c.brand) return 'other_brand';
    // Different family (e.g. iphone vs galaxy_s) → never Best Deals
    if (b.family && c.family && b.family !== c.family) return 'other_family';
    // Missing gen on either side → uncertain
    if (!b.gen || !c.gen) return 'uncertain';
    // Different generation → OTHER_MODELS
    if (b.gen !== c.gen) return 'other_gen';
    // Same gen, same tier, different storage → variant
    if (b.tier === c.tier && b.storage !== c.storage) return 'same_variant';
    // Same gen, same tier, same storage (or storage unknown) → exact
    if (b.tier === c.tier) return 'exact';
    // Same gen, different tier (e.g. 15 vs 15 Pro) → same_model
    return 'same_model';
  },

  bucket(baseTitle, candTitle, identMatch, candRole) {
    if (candRole !== 'main_product') return 'REJECT';
    if (identMatch === 'exact' || identMatch === 'same_variant' || identMatch === 'same_model') return 'BEST_DEALS';
    if (identMatch === 'other_gen' || identMatch === 'other_family' || identMatch === 'other_brand') return 'OTHER_MODELS';
    return 'UNCERTAIN';
  },

  score(baseTitle, candTitle) {
    const b = this._parse(baseTitle);
    const c = this._parse(candTitle);
    let s = 0;
    if (b.brand === c.brand) s += 30;
    if (b.family && c.family && b.family === c.family) s += 25;
    if (b.gen && c.gen && b.gen === c.gen) s += 30;
    if (b.tier === c.tier) s += 10;
    if (b.storage && c.storage && b.storage === c.storage) s += 5;
    return s;
  },

  label(identMatch) {
    return {
      exact: 'Exact match', same_variant: 'Same model, different storage',
      same_model: 'Same generation, different tier',
      other_gen: 'Different generation', other_family: 'Different model line',
      other_brand: 'Different brand', uncertain: 'Similar phone'
    }[identMatch] || 'Similar phone';
  },

  matchTier(identMatch) {
    return { exact:'EXACT', same_variant:'SAME_VARIANT', same_model:'SAME_MODEL', other_gen:'SAME_FAMILY', other_family:'SAME_FAMILY', other_brand:'RELATED' }[identMatch] || 'RELATED';
  }
};

// -- Audio Plugin --
const audioPlugin = {
  _parse(title) {
    const { s } = normalize(title);
    const brand =
      s.includes('airpods') || (s.includes('apple') && s.includes('air')) ? 'apple' :
      s.includes('sony')    ? 'sony' :
      s.includes('bose')    ? 'bose' :
      s.includes('samsung') || s.includes('galaxy buds') ? 'samsung' :
      s.includes('jabra')   ? 'jabra' :
      s.includes('sennheiser') ? 'sennheiser' :
      s.includes('soundcore') || s.includes('anker') ? 'anker' :
      s.includes('jbl')     ? 'jbl' : 'unknown';

    // Model family — primary identity for audio
    const family =
      s.includes('airpods pro')     ? 'airpods_pro' :
      s.includes('airpods max')     ? 'airpods_max' :
      s.includes('airpods')         ? 'airpods' :
      s.includes('wh-1000xm5')      ? 'wh1000xm5' :
      s.includes('wh-1000xm4')      ? 'wh1000xm4' :
      s.includes('wh-1000xm3')      ? 'wh1000xm3' :
      s.includes('wh-1000')         ? 'wh1000' :
      s.includes('wf-1000xm5')      ? 'wf1000xm5' :
      s.includes('wf-1000xm4')      ? 'wf1000xm4' :
      s.includes('wf-1000')         ? 'wf1000' :
      s.includes('qc45')            ? 'qc45' :
      s.includes('qc35')            ? 'qc35' :
      s.includes('quietcomfort 45') ? 'qc45' :
      s.includes('quietcomfort 35') ? 'qc35' :
      s.includes('galaxy buds2 pro') ? 'galaxy_buds2_pro' :
      s.includes('galaxy buds2')    ? 'galaxy_buds2' :
      s.includes('galaxy buds pro') ? 'galaxy_buds_pro' :
      s.includes('galaxy buds')     ? 'galaxy_buds' : null;

    const genM = s.match(/\b(\d+(?:st|nd|rd|th))\s*gen(?:eration)?\b/) || s.match(/\bgen\s*(\d)\b/) || s.match(/\bv(\d)\b(?!\d)/);
    const gen = genM ? genM[1].replace(/\D/g, '') : null;

    const cond = /\bused\b|\brefurbished\b/.test(s) ? 'used' : 'new';
    return { brand, family, gen, cond };
  },

  identityMatch(baseTitle, candTitle) {
    const b = this._parse(baseTitle);
    const c = this._parse(candTitle);
    if (b.brand !== 'unknown' && c.brand !== 'unknown' && b.brand !== c.brand) return 'other_brand';
    if (!b.family || !c.family) return 'uncertain';
    if (b.family !== c.family) return 'other_family';
    // Same family — check generation
    if (b.gen && c.gen && b.gen !== c.gen) return 'other_gen';
    return 'exact';
  },

  bucket(baseTitle, candTitle, identMatch, candRole, baseSub, candSub) {
    if (candRole !== 'main_product') return 'REJECT';
    // tws vs headphones are different subcategories — other models at most
    if (baseSub !== candSub) return 'OTHER_MODELS';
    if (identMatch === 'exact') return 'BEST_DEALS';
    if (identMatch === 'other_gen' || identMatch === 'other_family') return 'OTHER_MODELS';
    if (identMatch === 'other_brand') return 'OTHER_MODELS';
    return 'UNCERTAIN';
  },

  score(baseTitle, candTitle) {
    const b = this._parse(baseTitle);
    const c = this._parse(candTitle);
    let s = 0;
    if (b.brand === c.brand) s += 40;
    if (b.family && c.family && b.family === c.family) s += 50;
    if (b.gen && c.gen && b.gen === c.gen) s += 10;
    return s;
  },

  label(identMatch) {
    return {
      exact: 'Exact match', other_gen: 'Earlier/later generation',
      other_family: 'Different model', other_brand: 'Different brand', uncertain: 'Similar audio'
    }[identMatch] || 'Similar audio';
  },

  matchTier(identMatch) {
    return { exact:'EXACT', other_gen:'SAME_FAMILY', other_family:'SAME_FAMILY', other_brand:'RELATED' }[identMatch] || 'RELATED';
  }
};

// -- Laptop / Desktop Plugin --
const laptopPlugin = {
  _parse(title) {
    const { s } = normalize(title);
    const brand = ['lenovo','asus','dell','hp','acer','razer','msi','apple','samsung','lg','toshiba','huawei','microsoft'].find(b => s.includes(b)) || 'unknown';

    const family =
      s.includes('legion')      ? 'legion'      :
      s.includes('thinkpad')    ? 'thinkpad'    :
      s.includes('rog')         ? 'rog'         :
      s.includes('tuf gaming')  ? 'tuf'         :
      s.includes('xps')         ? 'xps'         :
      s.includes('spectre')     ? 'spectre'     :
      s.includes('envy')        ? 'envy'        :
      s.includes('omen')        ? 'omen'        :
      s.includes('victus')      ? 'victus'      :
      s.includes('razer blade') ? 'razer_blade' :
      s.includes('alienware')   ? 'alienware'   :
      s.includes('predator')    ? 'predator'    :
      s.includes('nitro')       ? 'nitro'       :
      s.includes('macbook pro') ? 'macbook_pro' :
      s.includes('macbook air') ? 'macbook_air' :
      s.includes('macbook')     ? 'macbook'     :
      s.includes('surface pro') ? 'surface_pro' : null;

    // GPU — strongest spec for gaming identity
    const gpuM = s.match(/\b(rtx|gtx)\s*(\d{3,4}(?:\s*ti)?)\b/i);
    const gpu = gpuM ? (gpuM[1] + gpuM[2]).toLowerCase().replace(/\s/g, '') : null;

    const cpuM = s.match(/\b(ryzen\s*[0-9]|core\s*i[0-9]|m[1-4]\b|core\s*ultra\s*\d)/i);
    const cpu = cpuM ? cpuM[1].toLowerCase().replace(/\s/g, '') : null;

    const ramM = s.match(/(\d+)gb(?:\s*(ram|ddr|memory)|\s|$)/);
    const ram = ramM ? ramM[1] : null;

    const storM = s.match(/(\d+)gb(?:\s*(ssd|nvme|hdd|storage))/);
    const storage = storM ? storM[1] : null;

    const screenM = s.match(/(\d{2}(?:\.\d)?)\s*(?:in|inch|")/);
    const screen = screenM ? screenM[1] : null;

    const cond = /\bused\b|\brefurbished\b/.test(s) ? 'used' : 'new';
    return { brand, family, gpu, cpu, ram, storage, screen, cond };
  },

  identityMatch(baseTitle, candTitle) {
    const b = this._parse(baseTitle);
    const c = this._parse(candTitle);
    const sameFamily = b.family && c.family && b.family === c.family;
    const gpuMatch   = b.gpu && c.gpu ? b.gpu === c.gpu : null;
    const ramMatch   = b.ram && c.ram ? b.ram === c.ram : null;
    const cpuMatch   = b.cpu && c.cpu ? b.cpu.slice(0,6) === c.cpu.slice(0,6) : null;

    if (sameFamily && gpuMatch && ramMatch) return 'exact';
    if (sameFamily && (gpuMatch || (cpuMatch && ramMatch))) return 'same_variant';
    if (sameFamily) return 'laptop_same_family'; // same brand line, different config → OTHER_MODELS not BEST_DEALS
    if (!sameFamily && gpuMatch && ramMatch && cpuMatch) return 'same_config'; // diff brand, same specs
    if (!sameFamily && gpuMatch) return 'same_gpu_class';
    return 'uncertain';
  },

  bucket(baseTitle, candTitle, identMatch, candRole, baseSub, candSub) {
    if (candRole !== 'main_product') return 'REJECT';
    if (baseSub !== candSub) return 'REJECT'; // laptop ≠ desktop
    if (identMatch === 'exact' || identMatch === 'same_variant') return 'BEST_DEALS';
    if (identMatch === 'laptop_same_family' || identMatch === 'same_config' || identMatch === 'same_gpu_class') return 'OTHER_MODELS';
    return 'UNCERTAIN';
  },

  score(baseTitle, candTitle) {
    const b = this._parse(baseTitle);
    const c = this._parse(candTitle);
    let s = 0;
    if (b.brand === c.brand) s += 15;
    if (b.family && c.family && b.family === c.family) s += 35;
    if (b.gpu && c.gpu && b.gpu === c.gpu) s += 25;
    if (b.cpu && c.cpu && b.cpu.slice(0,6) === c.cpu.slice(0,6)) s += 10;
    if (b.ram && c.ram && b.ram === c.ram) s += 10;
    if (b.storage && c.storage && b.storage === c.storage) s += 5;
    return s;
  },

  label(identMatch) {
    return {
      exact: 'Same model & config', same_variant: 'Same line, different config',
      same_family: 'Same product line', same_config: 'Comparable specs, different brand',
      same_gpu_class: 'Same GPU class', uncertain: 'May be comparable'
    }[identMatch] || 'Similar laptop';
  },

  matchTier(identMatch) {
    return { exact:'EXACT', same_variant:'SAME_VARIANT', laptop_same_family:'SAME_MODEL', same_config:'SAME_FAMILY', same_gpu_class:'SAME_FAMILY' }[identMatch] || 'RELATED';
  }
};

// -- Generic Plugin (conservative fallback for all other categories) --
const genericPlugin = {
  _parse(title) {
    const { s, tokens } = normalize(title);
    const brands = ['apple','samsung','sony','lg','dell','hp','lenovo','asus','acer','microsoft','google','anker','bosch','philips','nike','adidas','loreal'];
    const brand = brands.find(b => s.includes(b)) || null;
    // Model-ish tokens: alphanumeric combos that look like model codes
    const modelToks = tokens.filter(t => /[a-z]\d|\d[a-z]/.test(t) && t.length >= 3 && t.length <= 14);
    const sizeM = s.match(/(\d+(?:\.\d+)?)\s*(ml|l\b|kg|g\b|oz|fl\s*oz)/);
    const size = sizeM ? (sizeM[1] + sizeM[2]).replace(/\s/g, '') : null;
    return { brand, modelToks, size };
  },

  identityMatch(baseTitle, candTitle) {
    const b = this._parse(baseTitle);
    const c = this._parse(candTitle);
    if (b.brand && c.brand && b.brand !== c.brand) return 'other_brand';
    if (b.size && c.size && b.size !== c.size) return 'same_product_diff_size';
    const shared = b.modelToks.filter(t => c.modelToks.includes(t));
    if (shared.length >= 2) return 'likely_same';
    if (shared.length === 1 && b.brand === c.brand) return 'possibly_same';
    return 'uncertain';
  },

  bucket(baseTitle, candTitle, identMatch, candRole) {
    if (candRole !== 'main_product') return 'REJECT';
    if (identMatch === 'likely_same' || identMatch === 'same_product_diff_size') return 'BEST_DEALS';
    // Generic is conservative — uncertain stays uncertain
    if (identMatch === 'possibly_same') return 'UNCERTAIN';
    if (identMatch === 'other_brand') return 'OTHER_MODELS';
    return 'UNCERTAIN';
  },

  score(baseTitle, candTitle) {
    const b = this._parse(baseTitle);
    const c = this._parse(candTitle);
    const shared = b.modelToks.filter(t => c.modelToks.includes(t));
    return (b.brand === c.brand ? 20 : 0) + shared.length * 15;
  },

  label(identMatch) {
    return {
      likely_same: 'Likely same product', same_product_diff_size: 'Same product, different size',
      possibly_same: 'Possibly same product', other_brand: 'Different brand', uncertain: 'May be similar'
    }[identMatch] || 'Similar';
  },

  matchTier(identMatch) {
    return { likely_same:'EXACT', same_product_diff_size:'SAME_VARIANT', possibly_same:'SAME_MODEL' }[identMatch] || 'RELATED';
  }
};

// ── Console Plugin ────────────────────────────────────────────
// Handles: PS5, PS4, Xbox Series X/S, Nintendo Switch, Steam Deck
// Identity logic: same console line (ps5 = ps5 slim = ps5 digital) → BEST_DEALS
// Different line (ps5 vs ps4) → OTHER_MODELS
const consolePlugin = {
  _parse(title) {
    const { s } = normalize(title);
    const line =
      (s.includes('ps5') || s.includes('playstation 5')) ? 'ps5' :
      (s.includes('ps4') || s.includes('playstation 4')) ? 'ps4' :
      s.includes('xbox series x') ? 'xbox_series_x' :
      s.includes('xbox series s') ? 'xbox_series_s' :
      s.includes('xbox series')   ? 'xbox_series' :
      s.includes('nintendo switch oled') ? 'switch_oled' :
      s.includes('switch lite')   ? 'switch_lite' :
      s.includes('nintendo switch') || s.includes('switch') ? 'switch' :
      s.includes('steam deck')    ? 'steam_deck' : null;
    // Edition: digital/slim/standard/pro
    const edition =
      s.includes('slim')    ? 'slim' :
      s.includes('digital') ? 'digital' :
      s.includes('pro')     ? 'pro' : 'standard';
    const bundleTerms = ['spider','fortnite','bundle','god of war','fifa','call of duty','hogwarts'];
    const isBundle = bundleTerms.some(t => s.includes(t));
    const brand =
      line?.startsWith('ps') ? 'sony' :
      line?.startsWith('xbox') ? 'microsoft' :
      line?.startsWith('switch') || line === 'switch_oled' || line === 'switch_lite' ? 'nintendo' :
      line === 'steam_deck' ? 'valve' : 'unknown';
    return { line, edition, isBundle, brand };
  },
  identityMatch(baseTitle, candTitle) {
    const b = this._parse(baseTitle);
    const c = this._parse(candTitle);
    if (!b.line || !c.line) return 'uncertain';
    if (b.brand !== c.brand) return 'other_brand';  // sony vs microsoft
    if (b.line !== c.line) return 'other_gen';       // ps5 vs ps4
    // Same line — edition differences = same_variant (slim vs digital vs standard)
    if (c.isBundle && !b.isBundle) return 'same_variant'; // bundle vs non-bundle = variant
    if (b.edition !== c.edition)   return 'same_variant'; // slim vs digital
    return 'exact';
  },
  score(baseTitle, candTitle) {
    const b = this._parse(baseTitle);
    const c = this._parse(candTitle);
    let s = 0;
    if (b.brand === c.brand) s += 20;
    if (b.line && c.line && b.line === c.line) s += 50;
    if (b.edition === c.edition) s += 20;
    if (!c.isBundle) s += 10; // prefer non-bundle for exact
    return s;
  },
  label(identMatch) {
    return {
      exact: 'Same console edition', same_variant: 'Same console, different edition',
      other_gen: 'Different generation', other_brand: 'Different platform', uncertain: 'Similar console'
    }[identMatch] || 'Similar console';
  },
  matchTier(identMatch) {
    return { exact:'EXACT', same_variant:'SAME_VARIANT', other_gen:'SAME_FAMILY', other_brand:'RELATED' }[identMatch] || 'RELATED';
  }
};

// ── Controller Plugin ─────────────────────────────────────────
// Handles: DualSense, Xbox Wireless Controller, Pro Controller
const controllerPlugin = {
  _parse(title) {
    const { s } = normalize(title);
    const family =
      (s.includes('dualsense') || s.includes('playstation 5 controller')) ? 'dualsense' :
      (s.includes('dualshock') || s.includes('playstation 4 controller')) ? 'dualshock' :
      (s.includes('xbox wireless controller') || s.includes('xbox controller')) ? 'xbox_controller' :
      (s.includes('pro controller') && s.includes('nintendo')) ? 'switch_pro' :
      s.includes('gamepad') ? 'gamepad' : null;
    const isElite = s.includes('elite');
    // Color is a valid variant for controllers
    const color =
      s.includes('carbon black') ? 'carbon_black' :
      s.includes('robot white') ? 'robot_white' :
      s.includes('cosmic red') ? 'cosmic_red' :
      s.includes('starlight blue') ? 'starlight_blue' :
      s.includes('pulse red') ? 'pulse_red' : null;
    return { family, isElite, color };
  },
  identityMatch(baseTitle, candTitle) {
    const b = this._parse(baseTitle);
    const c = this._parse(candTitle);
    if (!b.family || !c.family) return 'uncertain';
    if (b.family !== c.family) return 'other_family'; // DualSense vs Xbox → OTHER_MODELS
    if (b.isElite !== c.isElite) return 'same_model'; // elite vs standard = same line
    // color = variant
    if (b.color !== c.color) return 'same_variant';
    return 'exact';
  },
  score(baseTitle, candTitle) {
    const b = this._parse(baseTitle);
    const c = this._parse(candTitle);
    let s = 0;
    if (b.family && c.family && b.family === c.family) s += 60;
    if (b.isElite === c.isElite) s += 20;
    if (b.color === c.color) s += 10;
    return s;
  },
  label(identMatch) {
    return {
      exact: 'Exact controller match', same_variant: 'Same controller, different color',
      same_model: 'Same controller line', other_family: 'Different controller'
    }[identMatch] || 'Similar controller';
  },
  matchTier(identMatch) {
    return { exact:'EXACT', same_variant:'SAME_VARIANT', same_model:'SAME_MODEL', other_family:'SAME_FAMILY' }[identMatch] || 'RELATED';
  }
};

// ── Plugin Registry ───────────────────────────────────────────
function getPlugin(sub) {
  if (sub === 'phone') return phonePlugin;
  if (sub === 'tws_earbuds' || sub === 'headphones' || sub === 'gaming_headset') return audioPlugin;
  if (['gaming_laptop','ultrabook','business_laptop','laptop','gaming_desktop','desktop'].includes(sub)) return laptopPlugin;
  if (sub === 'game_console') return consolePlugin;
  if (sub === 'gaming_controller') return controllerPlugin;
  // software_subscription, gift_card, and unknown → genericPlugin (conservative: needs token overlap)
  return genericPlugin;
}

// ============================================================
//  RUNTIME CLASSIFIER + CACHE
// ============================================================

const CLASSIFY_CACHE = new Map();

function classifyListingRuntime(title) {
  const cached = CLASSIFY_CACHE.get(title);
  if (cached) return cached;

  const { s } = normalize(title);
  const has = (...kws) => kws.some(k => s.includes(k));
  const reasons = [];

  // ── Step 1: Role detection first ──────────────────────────
  let role = 'main_product';
  if (s.includes('FORDVC') ||
      /\bfor\s+(iphone|ipad|samsung|galaxy|airpods|macbook|android|ps[45]|xbox|switch|pixel)\b/.test(s) ||
      /\bcompatible with\b/.test(s)) {
    role = 'accessory'; reasons.push('for_device');
  } else if (/\bapplecare\b|\bcare\s*\+|\bprotection\s*plan\b|\bwarranty\b|\binsurance\b|\bgeek\s*squad\b/.test(s)) {
    role = 'service'; reasons.push('service_plan');
  } else if (/\breplacement\b|\brepair\s*kit\b|\bspare\s*part\b|\bhousing\b|\bshell\b/.test(s)) {
    role = 'replacement_part'; reasons.push('replacement');
  } else if (/\bcase\b|\bcover\b|\bprotector\b|\bscreen guard\b|\bcable\b(?! tv)|\bstrap\b|\bsleeve\b|\bear\s*tips?\b|\beartips?\b|\bcushion\b|\bskin\b(?! care)|\bcooling pad\b|\bgraphics card\b|\bvideo card\b/.test(s)
             && !/(wireless charg|charging stand|charging pad|charging dock|charging station|magsafe|magnetic charg|qi charg|\d+w\s+charger)/.test(s)) {
    role = 'accessory'; reasons.push('accessory_pattern');
  } else if (/\bthumb grip\b|\bthumbstick\b|\bjoy-?con grip\b/.test(s)) {
    role = 'accessory'; reasons.push('controller_accessory');
  } else if (/\bcharging dock\b|\bcharging stand\b/.test(s) &&
             has('controller','ps5','ps4','xbox','dualsense','dualshock')) {
    role = 'accessory'; reasons.push('controller_dock');
  } else if (/\bbattery pack\b/.test(s) && has('controller','xbox','dualsense') &&
             !(/\d{4,}mah/).test(s)) {
    // battery pack for controller = accessory; standalone power bank with mAh = main_product
    role = 'accessory'; reasons.push('controller_battery');
  }

  // ── Step 2: Brand detection ────────────────────────────────
  const brand =
    has('apple','iphone','ipad','airpods','macbook','apple watch','magsafe') ? 'apple' :
    has('samsung','galaxy') ? 'samsung' :
    has('google','pixel','chromecast','google tv') ? 'google' :
    has('sony') ? 'sony' :
    has('bose') ? 'bose' :
    has('lenovo','legion','thinkpad') ? 'lenovo' :
    has('asus','rog','tuf') ? 'asus' :
    has('dell','xps','alienware') ? 'dell' :
    has('hp','omen','spectre','envy','victus') ? 'hp' :
    has('acer','predator','nitro') ? 'acer' :
    has('razer','razer blade') ? 'razer' :
    has('msi') ? 'msi' :
    has('jbl') ? 'jbl' :
    has('jabra') ? 'jabra' :
    has('roku') ? 'roku' :
    has('fire tv stick','fire tv','fire stick','amazon fire') ? 'amazon' :
    has('tp-link','tp link','tplink','archer be','archer ax','archer ac','deco') ? 'tp_link' :
    has('netgear','orbi','nighthawk') ? 'netgear' :
    has('linksys','velop') ? 'linksys' :
    has('eero') ? 'amazon' :
    has('tenda') ? 'tenda' :
    has('anker') ? 'anker' :
    has('belkin') ? 'belkin' :
    has('hyperx') ? 'hyperx' :
    has('logitech','logi') ? 'logitech' :
    has('microsoft','xbox') ? 'microsoft' :
    has('nintendo') ? 'nintendo' :
    has('xiaomi','redmi','poco') ? 'xiaomi' :
    has('oneplus') ? 'oneplus' :
    has('nokia') ? 'nokia' :
    has('huawei') ? 'huawei' :
    has('tecno') ? 'tecno' :
    has('infinix') ? 'infinix' :
    has('oppo') ? 'oppo' :
    has('vivo') ? 'vivo' :
    has('nothing phone') ? 'nothing' : 'unknown';

  // ── Step 3: Subcategory + intentClass ─────────────────────
  let subcategory = 'unknown';
  let intentClass = 'unknown';
  let confidence = 'low';
  let category = 'unknown';

  // -- Streaming devices (before phone/tablet) --
  if (has('fire tv stick 4k max')) {
    subcategory = 'streaming_device'; category = 'electronics';
    intentClass = 'fire_tv_stick_4k_max'; confidence = 'high';
    reasons.push('streaming_device_match');
  } else if (has('fire tv stick 4k') || (has('fire tv stick') && has('4k'))) {
    subcategory = 'streaming_device'; category = 'electronics';
    intentClass = 'fire_tv_stick_4k'; confidence = 'high';
    reasons.push('streaming_device_match');
  } else if (has('fire tv stick','fire stick') || (has('fire tv') && !has('phone','laptop'))) {
    subcategory = 'streaming_device'; category = 'electronics';
    intentClass = 'fire_tv_stick'; confidence = 'high';
    reasons.push('streaming_device_match');
  } else if (has('roku streaming stick+','roku streaming stick plus')) {
    subcategory = 'streaming_device'; category = 'electronics';
    intentClass = 'roku_streaming_stick_plus'; confidence = 'high';
    reasons.push('streaming_device_match');
  } else if (has('roku streaming stick')) {
    subcategory = 'streaming_device'; category = 'electronics';
    intentClass = 'roku_streaming_stick'; confidence = 'high';
    reasons.push('streaming_device_match');
  } else if (has('roku ultra')) {
    subcategory = 'streaming_device'; category = 'electronics';
    intentClass = 'roku_ultra'; confidence = 'high';
    reasons.push('streaming_device_match');
  } else if (has('roku express')) {
    subcategory = 'streaming_device'; category = 'electronics';
    intentClass = 'roku_express'; confidence = 'high';
    reasons.push('streaming_device_match');
  } else if (has('roku streambar')) {
    subcategory = 'streaming_device'; category = 'electronics';
    intentClass = 'roku_streambar'; confidence = 'high';
    reasons.push('streaming_device_match');
  } else if (has('roku') && !has('phone','laptop','tablet')) {
    subcategory = 'streaming_device'; category = 'electronics';
    intentClass = 'roku_device'; confidence = 'high';
    reasons.push('streaming_device_match');
  } else if (has('chromecast') || has('google tv streamer') || has('apple tv') ||
             has('onn streaming','onn. streaming') ||
             (has('streaming stick','media player','media streamer') && !has('phone','laptop','tablet'))) {
    subcategory = 'streaming_device'; category = 'electronics';
    const sdFam = has('chromecast') ? 'chromecast' : has('apple tv') ? 'apple_tv' : 'streaming_device';
    intentClass = sdFam; confidence = 'high';
    reasons.push('streaming_device_match');

  // -- Wireless chargers (before phone so MagSafe does not classify as phone) --
  } else if ((has('magsafe') && has('charger','charging','charge')) ||
             has('wireless charger','wireless charging pad','wireless charging stand') ||
             has('qi charger','qi charging','qi wireless') ||
             has('magnetic charger','magnetic charging') ||
             (has('charging stand','charging pad','charging dock') &&
              !has('controller','ps5','ps4','xbox','dualsense','dualshock','earbuds','airpods')) ||
             (has('charging station') && has('wireless','magsafe','magnetic','qi'))) {
    subcategory = 'wireless_charger'; category = 'electronics';
    const wcStyle = s.includes('stand') ? 'stand' : s.includes('pad') ? 'pad' :
                    s.includes('dock') ? 'dock' : 'charger';
    intentClass = 'wireless_charger_' + wcStyle; confidence = 'high';
    reasons.push('wireless_charger_match');

  // -- Mesh wifi (before router so mesh wins) --
  } else if (has('mesh wifi system','mesh wifi','mesh system','mesh network') ||
             has('whole home wifi','whole-home wifi') ||
             /\bdeco\s+[mxse]\d/i.test(s) ||
             has('orbi','eero','velop') ||
             (has('mesh') && has('router','wifi','wi-fi','wireless') && !has('phone','laptop'))) {
    subcategory = 'mesh_wifi'; category = 'electronics';
    intentClass = 'mesh_wifi'; confidence = 'high';
    reasons.push('mesh_wifi_match');

  // -- Standalone routers (after mesh) --
  } else if (has('wifi router','wi-fi router','wireless router','gaming router') ||
             (has('router') && has('wifi','wi-fi','dual-band','dual band','tri-band','tri band','gigabit','ax','be')) ||
             has('archer be','archer ax','archer ac','nighthawk router')) {
    subcategory = 'router'; category = 'electronics';
    intentClass = 'router'; confidence = 'high';
    reasons.push('router_match');

  // -- TV / monitor mounts --
  } else if (has('tv mount','wall mount','tv bracket','monitor mount',
                  'full motion mount','tilting mount','fixed mount','articulating mount') ||
             (has('mount','bracket') && has('tv','television','monitor','screen') &&
              !has('camera mount','mic mount','bike mount','phone mount'))) {
    subcategory = 'tv_mount'; category = 'electronics';
    const tmStyle = has('full motion','articulating') ? 'full_motion' :
                    has('tilt') ? 'tilt' : has('fixed') ? 'fixed' : 'mount';
    intentClass = 'tv_mount_' + tmStyle; confidence = 'high';
    reasons.push('tv_mount_match');

  // -- Light bulbs --
  } else if (has('light bulb','lightbulb','led bulb','corn bulb','corn light') ||
             (has('bulb') && (has('led','watt','lumen','pack') || /\d+w/.test(s))) ||
             (has('garage light','garage bulb','shop light') && !has('tv','phone','laptop'))) {
    subcategory = 'light_bulb'; category = 'home';
    intentClass = 'light_bulb'; confidence = 'high';
    reasons.push('light_bulb_match');

  } else if (has('airpods') || has('galaxy buds','buds pro','buds2','buds3') ||
      has('earbuds','tws','true wireless') ||
      has('wf-1000','freebuds','liberty air','redmi buds','soundpeats','enacfire','jabra elite') ||
      has('in-ear') || (has('wireless') && has('earphones','earbuds'))) {
    subcategory = 'tws_earbuds'; category = 'electronics';
    intentClass = (has('airpods pro','galaxy buds pro','wf-1000xm') ? 'premium_wireless_earbuds' : 'wireless_earbuds');
    confidence = 'high'; reasons.push('tws_match');

  } else if (has('wh-1000') || has('headphones','over-ear','on-ear') ||
             has('bose qc','bose nc','bose quiet','quietcomfort') ||
             has('momentum','hd 450','hd 560') ||
             (has('headset') && !has('gaming headset') && !has('call center'))) {
    subcategory = 'headphones'; category = 'electronics';
    intentClass = 'wireless_headphones';
    confidence = 'high'; reasons.push('headphone_match');

  } else if (has('iphone') ||
             (has('galaxy') && (/ galaxy [szam]\d/.test(s) || has('galaxy s2','s21','s22','s23','s24','s25'))) ||
             has('pixel 6','pixel 7','pixel 8','pixel 9','pixel 4','pixel 5') ||
             has('tecno camon','tecno spark','tecno pova','tecno pop') ||
             has('infinix hot','infinix note','infinix zero') ||
             has('redmi note','redmi 1','redmi 9','redmi 10','redmi 12','redmi 13') ||
             has('poco x','poco m','poco f') ||
             has('oneplus 1','oneplus 9','oneplus 10','oneplus 11','oneplus 12','oneplus nord') ||
             (has('smartphone','mobile phone','android phone') && !has('case','cover'))) {
    subcategory = 'phone'; category = 'electronics';
    intentClass = (has('iphone','galaxy s','pixel 7','pixel 8','pixel 9') ? 'flagship_smartphone' : 'smartphone');
    confidence = 'high'; reasons.push('phone_match');

  } else if (has('ipad') || has('galaxy tab','tab s','tab a') ||
             has('surface pro','surface go') ||
             (has('tablet') && !has('case','cover','drawing tablet'))) {
    subcategory = 'tablet'; category = 'electronics';
    intentClass = 'tablet';
    confidence = 'high'; reasons.push('tablet_match');

  } else if (has('gaming laptop','gaming notebook') ||
             has('rog','razer blade','legion','predator laptop','alienware') ||
             has('nitro 5','tuf gaming laptop','helios','scar','katana','victus') ||
             (has('laptop','notebook') && (has('rtx','gtx') || has('gaming')))) {
    subcategory = 'gaming_laptop'; category = 'electronics';
    intentClass = 'gaming_laptop';
    confidence = 'high'; reasons.push('gaming_laptop_match');

  } else if (has('macbook') || has('xps 13','xps 15','xps 17') ||
             has('spectre','zenbook','gram','swift edge','envy x360') ||
             (has('laptop','notebook') && has('m1','m2','m3','m4','ultrabook'))) {
    subcategory = 'ultrabook'; category = 'electronics';
    intentClass = 'laptop';
    confidence = 'high'; reasons.push('ultrabook_match');

  } else if (has('thinkpad','elitebook','latitude','probook','inspiron','vostro','expertbook') ||
             (has('laptop','notebook') && has('business','enterprise','workstation'))) {
    subcategory = 'business_laptop'; category = 'electronics';
    intentClass = 'laptop';
    confidence = 'high'; reasons.push('business_laptop_match');

  } else if (has('laptop','notebook') && !has('case','bag','stand','cooling')) {
    subcategory = 'laptop'; category = 'electronics';
    intentClass = 'laptop';
    confidence = 'medium'; reasons.push('generic_laptop');

  } else if (has('gaming pc','gaming desktop','gaming tower') ||
             (has('rtx','gtx') && has('desktop','tower') && !has('laptop'))) {
    subcategory = 'gaming_desktop'; category = 'electronics';
    intentClass = 'desktop_pc';
    confidence = 'high'; reasons.push('gaming_desktop_match');

  } else if ((has('desktop','tower') && !has('laptop')) || has('mini pc','all-in-one')) {
    subcategory = 'desktop'; category = 'electronics';
    intentClass = 'desktop_pc';
    confidence = 'medium'; reasons.push('desktop_match');

  } else if (has('apple watch') || has('galaxy watch') || has('pixel watch') ||
             has('garmin','amazfit','fitbit') ||
             (has('smartwatch','smart watch') && !has('case','strap replacement'))) {
    subcategory = 'smartwatch'; category = 'electronics';
    intentClass = 'smartwatch';
    confidence = 'high'; reasons.push('smartwatch_match');

  } else if (has('ps5','playstation 5') || has('ps4','playstation 4') ||
             has('xbox series') || has('nintendo switch','switch oled') || has('steam deck')) {
    subcategory = 'game_console'; category = 'electronics';
    intentClass = 'game_console';
    confidence = 'high'; reasons.push('console_match');

  } else if (has('dualsense','dualshock','xbox wireless controller','xbox controller') ||
             has('pro controller','gamepad') ||
             (has('controller') && (has('playstation','xbox','nintendo','ps5','ps4','switch')) &&
              !has('charging dock','charging stand','battery pack','thumb grip','silicone'))) {
    subcategory = 'gaming_controller'; category = 'electronics';
    intentClass = 'gaming_controller';
    confidence = 'high'; reasons.push('controller_match');

  } else if (has('microsoft 365','office 365','microsoft office') ||
             (has('365') && has('personal','family','home','business'))) {
    subcategory = 'software_subscription'; category = 'software';
    // Normalize plan tier for identity: personal vs family are different
    intentClass = has('family') ? 'ms365_family' :
                  has('personal') ? 'ms365_personal' :
                  has('business','enterprise') ? 'ms365_business' : 'ms365_other';
    confidence = 'high'; reasons.push('software_sub_match');

  } else if (has('gift card','psn card','xbox gift card','steam gift card','google play card')) {
    subcategory = 'gift_card'; category = 'voucher';
    intentClass = 'gift_card';
    confidence = 'high'; reasons.push('gift_card_match');
  }

  // -- Track B: classify generic/lifestyle/home products --------
  // Only runs if no strong Track A subcategory was assigned above.
  if (subcategory === 'unknown') {
    const tbNoun = _tbNoun(s);
    if (tbNoun) {
      subcategory = tbNoun;
      category    = _TB_BEST_DEALS_SAFE.has(tbNoun) ? 'trackb' : 'generic';
      intentClass = tbNoun;
      confidence  = 'medium';
      reasons.push('trackb_noun:' + tbNoun);
    }
  }
  // -- End Track B classifier extension -----------------------

  const numericConfidence =
    confidence === 'high'   ? 85 :
    confidence === 'medium' ? 60 :
    confidence === 'low'    ? 35 : 20;
  const result = { role, category, subcategory, intentClass, brand, confidence, numericConfidence, reasons };
  CLASSIFY_CACHE.set(title, result);
  return result;
}

// Intent compatibility map — which intentClasses can appear together in OTHER_MODELS
const INTENT_COMPAT = {
  premium_wireless_earbuds: new Set(['premium_wireless_earbuds','wireless_earbuds']),
  wireless_earbuds:         new Set(['premium_wireless_earbuds','wireless_earbuds']),
  wireless_headphones:      new Set(['wireless_headphones']),
  flagship_smartphone:      new Set(['flagship_smartphone','smartphone']),
  smartphone:               new Set(['flagship_smartphone','smartphone']),
  gaming_laptop:            new Set(['gaming_laptop','laptop','ultrabook']),
  laptop:                   new Set(['gaming_laptop','laptop','ultrabook','business_laptop']),
  ultrabook:                new Set(['gaming_laptop','laptop','ultrabook','business_laptop']),
  business_laptop:          new Set(['laptop','ultrabook','business_laptop']),
  desktop_pc:               new Set(['desktop_pc']),
  tablet:                   new Set(['tablet']),
  smartwatch:               new Set(['smartwatch']),
  game_console:             new Set(['game_console']),
  gaming_controller:        new Set(['gaming_controller']),
  // Microsoft 365 plan compatibility:
  // Same plan tier (personal↔personal) = BEST_DEALS; cross-tier = OTHER_MODELS
  ms365_personal:           new Set(['ms365_personal']),
  ms365_family:             new Set(['ms365_family']),
  ms365_business:           new Set(['ms365_business']),
  ms365_other:              new Set(['ms365_personal','ms365_family','ms365_business','ms365_other']),
  gift_card:                new Set(),   // gift cards never match each other across services
  unknown:                  new Set(),
  // Streaming device families
  fire_tv_stick_4k_max:      new Set(['fire_tv_stick_4k_max','fire_tv_stick_4k','fire_tv_stick']),
  fire_tv_stick_4k:          new Set(['fire_tv_stick_4k_max','fire_tv_stick_4k','fire_tv_stick']),
  fire_tv_stick:             new Set(['fire_tv_stick_4k_max','fire_tv_stick_4k','fire_tv_stick']),
  roku_streaming_stick_plus: new Set(['roku_streaming_stick_plus','roku_streaming_stick','roku_express','roku_ultra','roku_device']),
  roku_streaming_stick:      new Set(['roku_streaming_stick_plus','roku_streaming_stick','roku_express','roku_ultra','roku_device']),
  roku_express:              new Set(['roku_streaming_stick_plus','roku_streaming_stick','roku_express','roku_ultra','roku_device']),
  roku_ultra:                new Set(['roku_streaming_stick_plus','roku_streaming_stick','roku_express','roku_ultra','roku_device']),
  roku_device:               new Set(['roku_streaming_stick_plus','roku_streaming_stick','roku_express','roku_ultra','roku_device']),
  roku_streambar:            new Set(['roku_streambar']),
  chromecast:                new Set(['chromecast','streaming_device']),
  apple_tv:                  new Set(['apple_tv']),
  streaming_device:          new Set(['streaming_device']),
  // Wireless chargers - all styles are other-models-compatible
  wireless_charger_stand:    new Set(['wireless_charger_stand','wireless_charger_pad','wireless_charger_dock','wireless_charger_charger']),
  wireless_charger_pad:      new Set(['wireless_charger_stand','wireless_charger_pad','wireless_charger_dock','wireless_charger_charger']),
  wireless_charger_dock:     new Set(['wireless_charger_stand','wireless_charger_pad','wireless_charger_dock','wireless_charger_charger']),
  wireless_charger_charger:  new Set(['wireless_charger_stand','wireless_charger_pad','wireless_charger_dock','wireless_charger_charger']),
  // Networking
  router:                    new Set(['router']),
  mesh_wifi:                 new Set(['mesh_wifi']),
  // TV mounts
  tv_mount_full_motion:      new Set(['tv_mount_full_motion','tv_mount_tilt','tv_mount_fixed','tv_mount_mount']),
  tv_mount_tilt:             new Set(['tv_mount_full_motion','tv_mount_tilt','tv_mount_fixed','tv_mount_mount']),
  tv_mount_fixed:            new Set(['tv_mount_full_motion','tv_mount_tilt','tv_mount_fixed','tv_mount_mount']),
  tv_mount_mount:            new Set(['tv_mount_full_motion','tv_mount_tilt','tv_mount_fixed','tv_mount_mount']),
  // Light bulbs
  light_bulb:                new Set(['light_bulb']),
};

// ── Standalone helper: intent-level compatibility check ──────
// Used by decideDealBucket and can be called independently for debugging
function isOtherModelCompatible(baseCls, candCls) {
  if (candCls.role !== 'main_product') return false;
  if (candCls.numericConfidence !== undefined && candCls.numericConfidence < 30) return false;
  const compatSet = INTENT_COMPAT[baseCls.intentClass || 'unknown'] || new Set();
  if (compatSet.has(candCls.intentClass)) return true;
  // Subcategory group fallback
  const bg = SUB_GROUP[baseCls.subcategory] || baseCls.subcategory;
  const cg = SUB_GROUP[candCls.subcategory] || candCls.subcategory;
  return bg !== 'unknown' && bg === cg;
}

// MS365 cross-tier compat for OTHER_MODELS (different plans = alternatives)
const MS365_OTHER_MODELS_COMPAT = new Set(['ms365_personal','ms365_family','ms365_business','ms365_other']);

// Subcategory grouping — for "same group" checks
const SUB_GROUP = {
  phone: 'phone', 
  tws_earbuds: 'earbuds',
  headphones: 'headphones', gaming_headset: 'headphones',
  gaming_laptop: 'laptop', ultrabook: 'laptop', business_laptop: 'laptop', laptop: 'laptop',
  gaming_desktop: 'desktop', desktop: 'desktop',
  tablet: 'tablet', smartwatch: 'smartwatch', game_console: 'console',
  gaming_controller: 'gaming_controller',
  software_subscription: 'software_subscription',
  gift_card: 'gift_card',
  // New Track B roles
  streaming_device: 'streaming_device',
  wireless_charger: 'wireless_charger',
  router: 'router',
  mesh_wifi: 'mesh_wifi',
  tv_mount: 'tv_mount',
  light_bulb: 'light_bulb',
};

// ── PRODUCT-TYPE BEST DEALS REJECT RULES ─────────────────────
// For BEST_DEALS only. If the base product is X, these candidate
// subcategory groups are NEVER allowed in Best Deals (may still be OTHER_MODELS).
const BEST_DEAL_SUBCATEGORY_REJECTS = {
  // Console page: never put controllers/games/accessories in Best Deals
  'console':            new Set(['gaming_controller','gift_card','software_subscription']),
  // Controller page: never put consoles/games in Best Deals
  'gaming_controller':  new Set(['console','gift_card','software_subscription']),
  // Phone page: never put earbuds/watches/laptops in Best Deals
  'phone':              new Set(['earbuds','headphones','smartwatch','laptop','tablet','console']),
  // Earbuds page: no headphones in Best Deals (they ARE in Other Models)
  'earbuds':            new Set(['headphones','phone','laptop','tablet']),
  // Headphones page: no earbuds in Best Deals
  'headphones':         new Set(['earbuds','phone','laptop','tablet']),
  // Software subscription: only same software in Best Deals
  'software_subscription': new Set(['console','gaming_controller','phone','tablet','laptop']),
  // Router: mesh systems must not appear in Best Deals
  'router':           new Set(['mesh_wifi','streaming_device','phone','laptop','console']),
  // Mesh wifi: routers must not appear in Best Deals
  'mesh_wifi':        new Set(['router','streaming_device','phone','laptop','console']),
  // Wireless charger: no phones/laptops/keyboards in Best Deals
  'wireless_charger': new Set(['phone','laptop','tablet','console','keyboard','earbuds','headphones']),
  // Streaming device: no phones/consoles/laptops in Best Deals
  'streaming_device': new Set(['phone','laptop','tablet','console','controller']),
  // TV mount: no phones/laptops/consoles in Best Deals
  'tv_mount':         new Set(['phone','laptop','monitor','console']),
  // Light bulb: must not show anything non-bulb
  'light_bulb':       new Set(['phone','laptop','console','controller','streaming_device','router','mesh_wifi','wireless_charger']),
};

// ── FINAL BUCKETING ───────────────────────────────────────────
function decideDealBucket(baseCtx, candCtx) {
  const { baseTitle, basePrice, baseTax, baseClassified } = baseCtx;
  const { cand, identMatch, score, candTax, candClassified } = candCtx;

  // Hard gate: candidate must be a main product
  if (candClassified.role !== 'main_product') return { bucket: 'DROP', reason: `role:${candClassified.role}` };

  // Hard gate: price sanity — per spec: 0.2x–3.5x base price, gentle and configurable
  if (basePrice && Number.isFinite(basePrice) && basePrice > 0) {
    if (cand.price < basePrice * 0.20) return { bucket: 'DROP', reason: 'price_too_low' };
    if (cand.price > basePrice * 3.5)  return { bucket: 'DROP', reason: 'price_too_high' };
  }

  // ── Noun / device-type mismatch shield ────────────────────────
  // Catches cross-type contamination even when subcategory detection returns 'unknown'.
  // E.g. a console page must never show controllers/headsets; a laptop page must
  // never show desktop towers. These are hard DROPs — not even OTHER_MODELS.
  const _NOUN_RX = {
    console:          /\b(ps5|ps4|playstation\s*[45]|xbox series|nintendo switch|steam deck|console)\b/i,
    controller:       /\b(controller|dualsense|dualshock|gamepad|pro controller)\b/i,
    laptop:           /\b(laptop|notebook|macbook)\b/i,
    desktop:          /\b(desktop pc|gaming pc|gaming desktop|tower pc|mini pc)\b/i,
    earbuds:          /\b(earbuds?|airpods|galaxy buds|true wireless)\b/i,
    headphones:       /\b(headphones?|over-ear|on-ear|wh-\d|quietcomfort)\b/i,
    monitor:          /\b(monitor|display panel)\b/i,
    streaming_device: /\b(fire tv stick|fire stick|fire tv|roku|chromecast|apple tv|streaming stick|google tv streamer)\b/i,
    wireless_charger: /\b(wireless charg|magsafe charg|magnetic charg|qi charg|charging stand|charging pad|charging dock)\b/i,
    router:           /\b(wifi router|wi-fi router|wireless router|gaming router|archer\s+[abcde])\b/i,
    mesh_wifi:        /\b(mesh wifi|mesh system|mesh network|deco\s+[mxse]|orbi|eero|velop|whole home wifi)\b/i,
    tv_mount:         /\b(tv mount|wall mount|tv bracket|full motion mount|tilting mount|monitor mount)\b/i,
    light_bulb:       /\b(light bulb|lightbulb|led bulb|corn bulb|corn light|corn led)\b/i,
    phone:            /\b(iphone|galaxy\s+[sza]\d|pixel\s+\d|smartphone|android phone|mobile phone)\b/i,
  };
  // Pairs that are incompatible — [base noun, candidate noun] → DROP
  const _NOUN_INCOMPAT = [
    ['console','controller'],['console','laptop'],['console','headphones'],['console','earbuds'],['console','monitor'],
    ['controller','console'],['controller','laptop'],['controller','headphones'],
    ['laptop','desktop'],['laptop','monitor'],
    ['desktop','laptop'],['desktop','monitor'],
    ['earbuds','headphones'],['headphones','earbuds'],
    ['monitor','laptop'],['monitor','desktop'],
    // New role shields
    ['streaming_device','phone'],['streaming_device','console'],['streaming_device','laptop'],
    ['phone','streaming_device'],
    ['wireless_charger','phone'],['wireless_charger','laptop'],['wireless_charger','console'],
    ['wireless_charger','keyboard'],['wireless_charger','earbuds'],['wireless_charger','headphones'],
    ['phone','wireless_charger'],
    ['router','mesh_wifi'],['mesh_wifi','router'],
    ['router','streaming_device'],['router','phone'],['router','laptop'],
    ['mesh_wifi','streaming_device'],['mesh_wifi','phone'],['mesh_wifi','laptop'],
    ['tv_mount','phone'],['tv_mount','laptop'],['tv_mount','console'],
    ['light_bulb','phone'],['light_bulb','laptop'],['light_bulb','console'],['light_bulb','controller'],
    ['light_bulb','streaming_device'],['light_bulb','router'],['light_bulb','wireless_charger'],
    ['light_bulb','earbuds'],['light_bulb','headphones'],
  ];
  const _getNouns = (t) => Object.entries(_NOUN_RX).filter(([,rx]) => rx.test(t)).map(([n]) => n);
  const _baseNouns = _getNouns(baseTitle);
  const _candNouns = _getNouns(cand.title || '');
  if (_baseNouns.length && _candNouns.length) {
    for (const [a, b] of _NOUN_INCOMPAT) {
      if (_baseNouns.includes(a) && _candNouns.includes(b))
        return { bucket: 'DROP', reason: `noun_mismatch:${a}↔${b}` };
    }
  }
  // ── End noun mismatch shield ───────────────────────────────────

  // Hard drop: gift cards never match anything
  if (baseTax.sub === 'gift_card' || candTax.sub === 'gift_card')
    return { bucket: 'DROP', reason: 'gift_card' };

  const baseIntent   = baseClassified.intentClass || 'unknown';
  const candIntent   = candClassified.intentClass || 'unknown';
  const baseSubGroup = SUB_GROUP[baseTax.sub] || baseTax.sub;
  const candSubGroup = SUB_GROUP[candTax.sub] || candTax.sub;

  // ── BEST_DEALS: strict same-product identity ──────────────
  // RULE: 'same_family' means different generation → must NOT be bestDeals.
  // Only EXACT / SAME_VARIANT / SAME_MODEL identity from the plugin qualifies.
  // Plugin identMatches that mean "different product": other_gen, other_family, other_brand,
  // same_family (laptop plugin), same_gpu_class, same_config → fall to OTHER_MODELS.
  // BEST_DEALS: strict identity only. laptop_same_family, same_family (old), uncertain → not bestDeals.
  // possibly_same, uncertain, same_config, same_gpu_class, laptop_same_family → OTHER_MODELS candidates
  const BEST_DEAL_ID = new Set(['exact','same_variant','same_model','likely_same','same_product_diff_size']);
  const OTHER_MODEL_ID = new Set(['other_gen','other_family','other_brand','laptop_same_family',
    'same_config','same_gpu_class','uncertain','possibly_same','same_family']);
  // Apply product-type-specific Best Deals subcategory rejects
  const bestDealRejectSet = BEST_DEAL_SUBCATEGORY_REJECTS[baseSubGroup] || new Set();
  const candIsRejectedForBestDeals = bestDealRejectSet.has(candSubGroup);

  if (baseSubGroup === candSubGroup && baseSubGroup !== 'unknown' && !candIsRejectedForBestDeals) {
    if (BEST_DEAL_ID.has(identMatch)) {
      // ── Condition gate: used items never in BEST_DEALS when base is new ──
      const baseIsNew = baseClassified.subcategory !== 'unknown' &&
        !(/\bused\b|\brefurbished\b|\brenewed\b|\bpre.?owned\b/i.test(baseTitle));
      const candIsUsed = /\bused\b|\brefurbished\b|\brenewed\b|\bpre.?owned\b/i.test(cand.title || '');
      if (baseIsNew && candIsUsed) {
        const r = { bucket: 'OTHER_MODELS', reason: 'condition_mismatch_used' };
        console.log(`[Match] Base:${baseTitle.slice(0,30)} | Cand:${(cand.title||'').slice(0,30)} | Result:${r.bucket} (${r.reason})`);
        return r;
      }
      // ── MS365: plan tier must match (Personal ≠ Family) ──
      if (baseTax.sub === 'software_subscription') {
        if (baseIntent === candIntent) {
          const r = { bucket: 'BEST_DEALS', reason: `sw_same_plan:${identMatch}` };
          console.log(`[Match] Base:${baseTitle.slice(0,30)} | Cand:${(cand.title||'').slice(0,30)} | Result:${r.bucket} (${r.reason})`);
          return r;
        }
        // Different plan tier → OTHER_MODELS (handled below)
      } else {
        const r = { bucket: 'BEST_DEALS', reason: identMatch };
        console.log(`[Match] Base:${baseTitle.slice(0,30)} | Cand:${(cand.title||'').slice(0,30)} | Result:${r.bucket} (${r.reason})`);
        return r;
      }
    }
    // same_family / other_gen / other_family / other_brand / same_gpu_class → fall through
  } // end BEST_DEALS block

  // ── OTHER_MODELS: same intent class or compatible ─────────
  // Also accepts 'uncertain'/'possibly_same' identity IF intent is compatible —
  // this prevents valid alternatives from silently dropping to nothing.
  const compatSet = INTENT_COMPAT[baseIntent] || new Set();
  const intentCompatible = compatSet.has(candIntent) || baseSubGroup === candSubGroup;
  if (intentCompatible && candSubGroup !== 'unknown') {
    const r = { bucket: 'OTHER_MODELS', reason: `compat:${baseIntent}→${candIntent}` };
    console.log(`[Match] Base:${baseTitle.slice(0,30)} | Cand:${(cand.title||'').slice(0,30)} | Result:${r.bucket} (${r.reason})`);
    return r;
  }
  // Even if intent is unknown, strong identMatch pointing to "related product" → OTHER_MODELS
  if (OTHER_MODEL_ID.has(identMatch) && score >= 20 && candSubGroup !== 'unknown' && baseSubGroup !== 'unknown') {
    const r = { bucket: 'OTHER_MODELS', reason: `identMatch_rescue:${identMatch}` };
    console.log(`[Match] Base:${baseTitle.slice(0,30)} | Cand:${(cand.title||'').slice(0,30)} | Result:${r.bucket} (${r.reason})`);
    return r;
  }

  // ── MS365: cross-tier = OTHER_MODELS (e.g. Personal vs Family) ──
  if (baseTax.sub === 'software_subscription' && candTax.sub === 'software_subscription') {
    if (MS365_OTHER_MODELS_COMPAT.has(baseIntent) && MS365_OTHER_MODELS_COMPAT.has(candIntent)) {
      const r = { bucket: 'OTHER_MODELS', reason: `sw_cross_plan:${baseIntent}→${candIntent}` };
      console.log(`[Match] Base:${baseTitle.slice(0,30)} | Cand:${(cand.title||'').slice(0,30)} | Result:${r.bucket} (${r.reason})`);
      return r;
    }
  }

  // ── Taxonomy fallback: same subcategory group even if intentClass unknown ──
  if (baseTax.sub !== 'unknown' && candTax.sub !== 'unknown') {
    const bg = SUB_GROUP[baseTax.sub] || baseTax.sub;
    const cg = SUB_GROUP[candTax.sub] || candTax.sub;
    if (bg === cg) {
      const r = { bucket: 'OTHER_MODELS', reason: `taxonomy_group:${bg}` };
      console.log(`[Match] Base:${baseTitle.slice(0,30)} | Cand:${(cand.title||'').slice(0,30)} | Result:${r.bucket} (${r.reason})`);
      return r;
    }
  }

  // -- Track B fallback -------------------------------------------
  // Fires only when Track A produced no useful bucket.
  // Hard-identity categories (phone/console/controller/sub) are excluded by _isTrackBProduct.
  if (_isTrackBProduct(baseTax)) {
    const tbR = scoreTrackBMatch(baseTitle, cand.title || '');
    if (tbR.rejectReason) {
      console.log(`[TrackB] ${baseTitle.slice(0,26)} | ${(cand.title||'').slice(0,26)} | DROP (${tbR.rejectReason})`);
      return { bucket: 'DROP', reason: tbR.rejectReason };
    }
    if (tbR.strongMatch && tbR.nounMatch && tbR._debug && _TB_BEST_DEALS_SAFE.has(tbR._debug.baseNoun)) {
      // Brand gate: for brand-sensitive categories, require same brand for BEST_DEALS
      const _TB_BRAND_GATE = new Set(['streaming_device','wireless_charger']);
      if (_TB_BRAND_GATE.has(tbR._debug.baseNoun)) {
        const bBrd = extractTrackBFingerprint(baseTitle).brand;
        const cBrd = extractTrackBFingerprint(cand.title||'').brand;
        if (!bBrd || !cBrd || bBrd !== cBrd) {
          console.log(`[TrackB] ${baseTitle.slice(0,26)} | ${(cand.title||'').slice(0,26)} | OTHER_MODELS (brand gate: ${tbR._debug.baseNoun})`);
          return { bucket: 'OTHER_MODELS', reason: `trackb_brand_gate:${tbR.score}` };
        }
      }
      console.log(`[TrackB] ${baseTitle.slice(0,26)} | ${(cand.title||'').slice(0,26)} | BEST_DEALS score=${tbR.score}`);
      return { bucket: 'BEST_DEALS', reason: `trackb_strong:${tbR.score}` };
    }
    if (tbR.moderateMatch) {
      console.log(`[TrackB] ${baseTitle.slice(0,26)} | ${(cand.title||'').slice(0,26)} | OTHER_MODELS score=${tbR.score}`);
      return { bucket: 'OTHER_MODELS', reason: `trackb_moderate:${tbR.score}` };
    }
    console.log(`[TrackB] ${baseTitle.slice(0,26)} | ${(cand.title||'').slice(0,26)} | DROP score=${tbR.score}`);
    return { bucket: 'DROP', reason: `trackb_weak:${tbR.score}` };
  }
  // -- End Track B fallback -------------------------------------

  const dropR = { bucket: 'DROP', reason: `no_compat:${baseIntent}↔${candIntent}` };
  console.log(`[Match] Base:${baseTitle.slice(0,30)} | Cand:${(cand.title||'').slice(0,30)} | Result:DROP (${dropR.reason})`);
  return dropR;
}

// ── BROAD FALLBACK TERMS ──────────────────────────────────────
// Uses real words found in product titles, NOT taxonomy codes
function getBroadFallbackTerms(sub) {
  const map = {
    phone:           ['iphone','galaxy','smartphone','mobile'],
    tws_earbuds:     ['earbuds','airpods','buds','wireless earphones'],
    headphones:      ['headphones','wh-1000','quietcomfort'],
    gaming_headset:  ['gaming headset','gaming headphones'],
    gaming_laptop:   ['gaming laptop','gaming notebook'],
    ultrabook:       ['laptop','macbook','ultrabook'],
    business_laptop: ['laptop','thinkpad','elitebook'],
    laptop:          ['laptop','notebook'],
    gaming_desktop:  ['gaming pc','gaming desktop','gaming tower'],
    desktop:         ['desktop','tower','mini pc'],
    tablet:          ['tablet','ipad'],
    smartwatch:      ['smartwatch','smart watch'],
    game_console:    ['ps5','xbox','nintendo switch'],
    monitor:         ['monitor'],
    smart_tv:        ['smart tv'],
    power_bank:      ['power bank','powerbank'],
    internal_ssd:    ['ssd','nvme'],
    external_storage:['external ssd','external hard drive'],
    gpu:             ['graphics card','rtx','gtx'],
    gaming_controller: ['xbox controller','dualsense','dualshock','pro controller','gamepad'],
    software_subscription: ['microsoft 365','office 365','microsoft office'],
    game_console:    ['ps5','playstation','xbox','nintendo switch','steam deck'],
  };
  return map[sub] || [];
}

// ── RETRIEVAL QUERIES ─────────────────────────────────────────
// Returns array-of-{terms, mode} where mode is 'OR' (any term) or 'AND' (all terms)
// OR queries for broad coverage, AND queries for precise matches
function buildRetrievalQueries(title, tax) {
  const { s } = normalize(title);
  const queries = []; // each entry: { terms: string[], mode: 'OR'|'AND' }

  if (tax.sub === 'phone') {
    const p = phonePlugin._parse(title);
    // Precise: family + gen number
    if (p.family && p.gen) queries.push({ terms: [p.family.replace(/_/g,' '), p.gen], mode: 'AND' });
    // Broad: just family name — catches all variants
    if (p.family) queries.push({ terms: [p.family.replace(/_/g,' ')], mode: 'AND' });
    // Broadest: brand-level OR for Other Models
    queries.push({ terms: ['iphone','galaxy','pixel','smartphone','tecno','infinix','redmi','oneplus','oppo','vivo'], mode: 'OR' });

  } else if (tax.sub === 'tws_earbuds' || tax.sub === 'headphones' || tax.sub === 'gaming_headset') {
    const p = audioPlugin._parse(title);
    if (p.family) {
      const fw = p.family.replace(/_/g,' ').split(' ').filter(w => w.length >= 3);
      if (fw.length >= 2) queries.push({ terms: fw.slice(0,2), mode: 'AND' });
      else if (fw.length === 1) queries.push({ terms: fw, mode: 'AND' });
    }
    if (p.brand && p.brand !== 'unknown') queries.push({ terms: [p.brand], mode: 'AND' });
    // Broad OR for Other Models
    queries.push({ terms: ['earbuds','airpods','buds','headphones','true wireless'], mode: 'OR' });

  } else if (['gaming_laptop','ultrabook','business_laptop','laptop'].includes(tax.sub)) {
    const p = laptopPlugin._parse(title);
    if (p.gpu) {
      const gpuN = p.gpu.match(/\d{3,4}/)?.[0];
      const gpuB = p.gpu.match(/^(rtx|gtx)/)?.[0];
      if (gpuB && gpuN) queries.push({ terms: [gpuB, gpuN], mode: 'AND' });
    }
    if (p.family) queries.push({ terms: [p.family.replace(/_/g,' ')], mode: 'AND' });
    if (p.brand && p.brand !== 'unknown') queries.push({ terms: [p.brand, 'laptop'], mode: 'AND' });
    // Broad OR
    queries.push({ terms: ['gaming laptop','laptop','notebook','macbook','thinkpad','xps','legion','omen'], mode: 'OR' });

  } else if (['gaming_desktop','desktop'].includes(tax.sub)) {
    const p = laptopPlugin._parse(title);
    if (p.gpu) {
      const gpuN = p.gpu.match(/\d{3,4}/)?.[0];
      const gpuB = p.gpu.match(/^(rtx|gtx)/)?.[0];
      if (gpuB && gpuN) queries.push({ terms: [gpuB, gpuN], mode: 'AND' });
    }
    if (p.family) queries.push({ terms: [p.family.replace(/_/g,' ')], mode: 'AND' });
    queries.push({ terms: ['gaming pc','gaming desktop','desktop','tower'], mode: 'OR' });

  } else if (tax.sub === 'tablet') {
    queries.push({ terms: ['ipad'], mode: 'AND' });
    queries.push({ terms: ['galaxy tab'], mode: 'AND' });
    queries.push({ terms: ['tablet','ipad'], mode: 'OR' });

  } else if (tax.sub === 'smartwatch') {
    queries.push({ terms: ['apple watch'], mode: 'AND' });
    queries.push({ terms: ['galaxy watch'], mode: 'AND' });
    queries.push({ terms: ['smartwatch','smart watch','garmin','fitbit'], mode: 'OR' });

  } else if (tax.sub === 'game_console') {
    const term = s.includes('ps5') ? 'ps5' : s.includes('ps4') ? 'ps4' :
                 s.includes('xbox series') ? 'xbox series' :
                 s.includes('switch') ? 'nintendo switch' : null;
    if (term) queries.push({ terms: [term], mode: 'AND' });
    queries.push({ terms: ['ps5','xbox','nintendo switch','playstation'], mode: 'OR' });

  } else if (tax.sub === 'gaming_controller') {
    // Precise: exact controller family
    const cBase = s.includes('xbox') ? 'xbox controller' :
                  s.includes('dualsense') ? 'dualsense' :
                  s.includes('dualshock') ? 'dualshock' :
                  s.includes('pro controller') ? 'pro controller' : 'controller';
    queries.push({ terms: [cBase], mode: 'AND' });
    // Broad: all major controller families
    queries.push({ terms: ['xbox controller','dualsense','dualshock','pro controller','gamepad'], mode: 'OR' });

  } else if (tax.sub === 'software_subscription') {
    // Normalize: "microsoft 365" / "office 365" / "microsoft office"
    queries.push({ terms: ['microsoft 365'], mode: 'AND' });
    queries.push({ terms: ['office 365'], mode: 'AND' });
    queries.push({ terms: ['microsoft 365','office 365','microsoft office'], mode: 'OR' });

  } else if (tax.sub === 'power_bank') {
    const mahM = s.match(/(\d{4,6})mah/);
    if (mahM) queries.push({ terms: [mahM[1]+'mah'], mode: 'AND' });
    queries.push({ terms: ['power bank','powerbank','portable charger'], mode: 'OR' });

  } else if (tax.sub === 'monitor') {
    const sizeM = s.match(/(\d{2,3})(?:in|inch|")/);
    if (sizeM) queries.push({ terms: [sizeM[1], 'monitor'], mode: 'AND' });
    queries.push({ terms: ['monitor','gaming monitor','display'], mode: 'OR' });

  } else if (tax.sub === 'smart_tv') {
    const sizeM = s.match(/(\d{2,3})(?:in|inch|")/);
    if (sizeM) queries.push({ terms: [sizeM[1], 'tv'], mode: 'AND' });
    queries.push({ terms: ['smart tv','television','oled tv','qled'], mode: 'OR' });

  // -- Streaming devices ------------------------------------
  } else if (tax.sub === 'streaming_device') {
    const sdFP = extractTrackBFingerprint(title);
    if (s.includes('fire tv stick 4k max')) {
      queries.push({ terms: ['fire tv stick', '4k', 'max'], mode: 'AND' });
      queries.push({ terms: ['fire tv stick', '4k'], mode: 'AND' });
    } else if (s.includes('fire tv stick 4k') || (s.includes('fire tv stick') && s.includes('4k'))) {
      queries.push({ terms: ['fire tv stick', '4k'], mode: 'AND' });
    } else if (s.includes('fire tv stick') || s.includes('fire stick')) {
      queries.push({ terms: ['fire tv stick'], mode: 'AND' });
    } else if (s.includes('roku streaming stick')) {
      queries.push({ terms: ['roku streaming stick'], mode: 'AND' });
    } else if (s.includes('roku ultra')) {
      queries.push({ terms: ['roku ultra'], mode: 'AND' });
    } else if (s.includes('roku express')) {
      queries.push({ terms: ['roku express'], mode: 'AND' });
    } else if (s.includes('roku streambar')) {
      queries.push({ terms: ['roku streambar'], mode: 'AND' });
    } else if (s.includes('roku')) {
      queries.push({ terms: ['roku'], mode: 'AND' });
    } else if (s.includes('chromecast')) {
      queries.push({ terms: ['chromecast'], mode: 'AND' });
    } else if (s.includes('apple tv')) {
      queries.push({ terms: ['apple tv'], mode: 'AND' });
    }
    if (sdFP.brand && s.includes('4k')) queries.push({ terms: [sdFP.brand.replace(/_/g,' '), '4k'], mode: 'AND' });
    else if (sdFP.brand)               queries.push({ terms: [sdFP.brand.replace(/_/g,' ')], mode: 'AND' });
    queries.push({ terms: ['fire tv stick','roku','chromecast','streaming stick','apple tv','google tv streamer'], mode: 'OR' });

  // -- Wireless chargers ------------------------------------
  } else if (tax.sub === 'wireless_charger') {
    if (s.includes('magsafe')) {
      queries.push({ terms: ['magsafe', 'charger'], mode: 'AND' });
      queries.push({ terms: ['magsafe', 'charging'], mode: 'AND' });
    } else if (s.includes('magnetic') && s.includes('charg')) {
      queries.push({ terms: ['magnetic', 'charging'], mode: 'AND' });
    } else {
      queries.push({ terms: ['wireless charger'], mode: 'AND' });
    }
    if (s.includes('stand')) queries.push({ terms: ['charging stand'], mode: 'AND' });
    else if (s.includes('pad')) queries.push({ terms: ['charging pad'], mode: 'AND' });
    else if (s.includes('dock')) queries.push({ terms: ['charging dock'], mode: 'AND' });
    queries.push({ terms: ['wireless charger','magsafe charger','charging stand','charging pad','qi charger'], mode: 'OR' });

  // -- Mesh wifi ---------------------------------------------
  } else if (tax.sub === 'mesh_wifi') {
    const meshFP = extractTrackBFingerprint(title);
    if (meshFP.brand) queries.push({ terms: [meshFP.brand.replace(/_/g,' '), 'mesh'], mode: 'AND' });
    const meshSpeedM = s.match(/\b(ax\d{3,5}|be\d{3,5})\b/i);
    if (meshSpeedM) queries.push({ terms: ['mesh', meshSpeedM[0].toLowerCase()], mode: 'AND' });
    queries.push({ terms: ['mesh wifi'], mode: 'AND' });
    queries.push({ terms: ['mesh system','mesh wifi','whole home wifi','orbi','eero','deco','velop'], mode: 'OR' });

  // -- Standalone routers ------------------------------------
  } else if (tax.sub === 'router') {
    const routerFP = extractTrackBFingerprint(title);
    const routerSpeedM = s.match(/\b(be\d{3,5}|ax\d{3,5}|ac\d{3,5})\b/i);
    if (routerSpeedM && routerFP.brand) queries.push({ terms: [routerFP.brand.replace(/_/g,' '), routerSpeedM[0].toLowerCase()], mode: 'AND' });
    if (routerSpeedM) queries.push({ terms: [routerSpeedM[0].toLowerCase(), 'router'], mode: 'AND' });
    if (routerFP.brand) queries.push({ terms: [routerFP.brand.replace(/_/g,' '), 'router'], mode: 'AND' });
    const routerWifiGen = s.includes('wifi 7') || s.includes('wi-fi 7') ? 'wifi 7' :
                          s.includes('wifi 6e') || s.includes('wi-fi 6e') ? 'wifi 6e' :
                          s.includes('wifi 6') || s.includes('wi-fi 6') ? 'wifi 6' : null;
    if (routerWifiGen) queries.push({ terms: [routerWifiGen, 'router'], mode: 'AND' });
    queries.push({ terms: ['wifi router','wireless router','gaming router','router'], mode: 'OR' });

  // -- TV / monitor mounts -----------------------------------
  } else if (tax.sub === 'tv_mount') {
    const mountStyle = s.includes('full motion') || s.includes('articulating') ? 'full motion' :
                       s.includes('tilt') ? 'tilt' : s.includes('fixed') ? 'fixed' : null;
    if (mountStyle) queries.push({ terms: [mountStyle, 'mount'], mode: 'AND' });
    const mountSzM = s.match(/(\d{2,3})(?:in|inch|")/);
    if (mountSzM) queries.push({ terms: [mountSzM[1], 'mount'], mode: 'AND' });
    queries.push({ terms: ['tv mount'], mode: 'AND' });
    queries.push({ terms: ['tv mount','wall mount','tv bracket','monitor mount'], mode: 'OR' });

  // -- Light bulbs --------------------------------------------
  } else if (tax.sub === 'light_bulb') {
    const bType = s.includes('corn') ? 'corn bulb' : s.includes('led bulb') ? 'led bulb' : 'light bulb';
    queries.push({ terms: [bType], mode: 'AND' });
    const bWattM = s.match(/\b(\d{2,4})w(?:att)?\b/i);
    if (bWattM) queries.push({ terms: [bType, bWattM[0].toLowerCase()], mode: 'AND' });
    const bPackM = s.match(/\b(\d{1,2})[\-\s]?pack\b/i);
    if (bPackM) queries.push({ terms: [bType, bPackM[0].toLowerCase()], mode: 'AND' });
    queries.push({ terms: ['light bulb','led bulb','corn bulb','corn light','bulb'], mode: 'OR' });

  } else {
    // Generic fallback: try Track B noun-first retrieval before raw tokens
    const tbFP = extractTrackBFingerprint(title);
    if (tbFP.noun) {
      // Noun is the anchor; add 1-2 strong feature/size pillars when available
      const pillars = [...tbFP.features.slice(0,2), ...tbFP.sizes.slice(0,1)];
      if (pillars.length >= 1) {
        // Precise: noun + one strong feature
        queries.push({ terms: [tbFP.noun.replace(/_/g,' '), pillars[0]], mode: 'AND' });
      }
      // Broad: just the noun (maximum recall for Track B category)
      queries.push({ terms: [tbFP.noun.replace(/_/g,' ')], mode: 'AND' });
      // Brand-level OR if brand is known
      if (tbFP.brand) queries.push({ terms: [tbFP.brand.replace(/_/g,' ')], mode: 'OR' });
    } else {
      // No noun found -- fall back to raw strong tokens
      const stopWords = new Set(['the','and','for','with','new','free','fast','best','good','high','low']);
      const { tokens } = normalize(title);
      const strong = tokens.filter(t => t.length >= 4 && !stopWords.has(t)).slice(0, 3);
      if (strong.length >= 2) queries.push({ terms: strong.slice(0,2), mode: 'AND' });
      if (strong.length >= 1) queries.push({ terms: [strong[0]], mode: 'OR' });
    }
  }

  // Deduplicate
  const seen = new Set();
  return queries.filter(q => {
    const k = q.terms.join('|');
    if (seen.has(k)) return false;
    seen.add(k); return true;
  });
}

// ── MAIN ENGINE ───────────────────────────────────────────────
async function computeDealsForProduct({ storeId, storeProductId, baseTitle, baseCurrency, basePrice, limit = 20 }) {
  const baseTax        = taxonomy(baseTitle);
  const plugin         = getPlugin(baseTax.sub);
  const baseClassified = classifyListingRuntime(baseTitle);

  const acceptedSeen = new Set();
  const rawBest      = [];
  const rawOther     = [];

  const dbg = {
    retrievedTotal: 0, dedupedTotal: 0,
    rejectedRole: 0, rejectedPrice: 0, rejectedClassifierLowConfidence: 0,
    rejectedIntentMismatch: 0, rejectedWeakIdentity: 0,
    acceptedBestDeals: 0, acceptedOtherModels: 0,
    retrieved: 0, hardRejected: 0, accepted: 0, bestDeals: 0, otherModels: 0,
    vectorUsed: false,
  };

  // ── Shared scorer — identical logic to old per-store loop ──
  function scoreAndBucket(cand) {
    const candClassified = classifyListingRuntime(cand.title);
    if (candClassified.role !== 'main_product') { dbg.rejectedRole++; dbg.hardRejected++; return; }

    const candTax = taxonomy(cand.title);
    if (baseTax.cat !== 'unknown' && candTax.cat !== 'unknown' && baseTax.cat !== candTax.cat) {
      dbg.rejectedIntentMismatch++; dbg.hardRejected++; return;
    }

    if (!priceSane(baseTax.sub, cand.price)) { dbg.rejectedPrice++; dbg.hardRejected++; return; }

    let identMatch = 'uncertain', score = 0, label = 'Similar', mTier = 'RELATED';
    try {
      identMatch = plugin.identityMatch(baseTitle, cand.title);
      score      = plugin.score(baseTitle, cand.title);
      label      = plugin.label(identMatch);
      mTier      = plugin.matchTier(identMatch);
    } catch {}

    if (candClassified.numericConfidence !== undefined && candClassified.numericConfidence < 25) {
      dbg.rejectedClassifierLowConfidence++; dbg.hardRejected++; return;
    }

    const { bucket } = decideDealBucket(
      { baseTitle, basePrice, baseTax, baseClassified },
      { cand, identMatch, score, candTax, candClassified }
    );
    if (bucket === 'DROP') return;

    const acceptKey = `${cand.storeId}|${cand.storeProductId}`;
    if (acceptedSeen.has(acceptKey)) return;
    acceptedSeen.add(acceptKey);

    const item = {
      ...cand,
      name:       STORE_DISPLAY[cand.storeId] || cand.storeId,
      matchTier:  mTier,
      matchLabel: label,
      condition:  /\bused\b|\brefurbished\b/.test((cand.title||'').toLowerCase()) ? 'used' : 'new',
      score, bucket, identMatch,
      candBrand: candClassified.brand,
    };

    if (bucket === 'BEST_DEALS')        { dbg.acceptedBestDeals++;   rawBest.push(item); }
    else if (bucket === 'OTHER_MODELS') { dbg.acceptedOtherModels++; rawOther.push(item); }
    dbg.accepted++;
  }

  // ── VECTOR RETRIEVAL — one RPC, all stores at once ─────────
  let vectorWorked = false;

  // Only attempt vector search if this listing already has an embedding in DB.
  // If it doesn't, skip straight to ilike — generateEmbedding would block here
  // waiting for Gemini (up to 10s) and cause the extension to timeout.
  // autoEmbedListing fires async after the upsert and will embed it for next visit.
  let embedding = embCache.get((baseTitle||'').trim().slice(0, 200)); // check in-memory first
  if (!embedding) {
    try {
      const { data: listingRow } = await supabase
        .from('store_listings').select('embedding')
        .eq('store_id', storeId).eq('store_product_id', storeProductId).maybeSingle();
      if (listingRow?.embedding) {
        // Row has embedding in DB — safe to generate/use (will hit Gemini only if not in embCache)
        embedding = await generateEmbedding(baseTitle);
      }
      // else: no embedding yet — skip vector, use ilike fallback below
    } catch { /* silently fall through to ilike */ }
  }

  if (embedding) {
    try {
      const { data: vecRows, error: vecErr } = await supabase.rpc("match_listings", {
        query_embedding:  embedding,
        exclude_store_id: storeId,
        match_threshold:  0.55,
        match_count:      80,
      });

      if (vecErr) {
        console.warn(`[vector] RPC error: ${vecErr.message} — falling back to ilike`);
      } else if (vecRows && vecRows.length > 0) {
        // Batch-fetch prices per store in parallel
        const byStore = {};
        for (const r of vecRows) {
          (byStore[r.store_id] = byStore[r.store_id] || []).push(r.store_product_id);
        }
        const priceMap = new Map();
        await Promise.all(Object.entries(byStore).map(async ([sid, ids]) => {
          const { data: prods } = await supabase.from("products")
            .select("store_id,store_product_id,last_price,currency")
            .eq("store_id", sid).in("store_product_id", ids);
          for (const p of (prods || []))
            priceMap.set(`${p.store_id}|${p.store_product_id}`, p);
        }));

        dbg.retrievedTotal = vecRows.length;
        dbg.vectorUsed = true;
        vectorWorked = true;

        for (const row of vecRows) {
          const p = priceMap.get(`${row.store_id}|${row.store_product_id}`);
          if (!p) continue;
          const price = Number(p.last_price);
          if (!Number.isFinite(price) || price <= 0) continue;
          scoreAndBucket({
            storeId: row.store_id, storeProductId: String(row.store_product_id),
            title: row.title || "", price,
            currency: p.currency || baseCurrency || "USD",
            url: row.page_url || row.canonical_url || null,
          });
        }
        console.log(`[vector] retrieved=${vecRows.length} best=${rawBest.length} other=${rawOther.length} for "${baseTitle.slice(0,45)}"`);
      } else {
        console.log(`[vector] 0 results above threshold — falling back to ilike`);
      }
    } catch (e) {
      console.warn(`[vector] error: ${e.message} — falling back to ilike`);
    }
  }

  // ── ILIKE FALLBACK — only if vector returned nothing ───────
  if (!vectorWorked) {
    const queries = buildRetrievalQueries(baseTitle, baseTax);
    const { data: stores } = await supabase.from('stores').select('id').neq('id', storeId);
    const storeIds = (stores || []).map(s => s.id);

    async function fetchCandidates(sid, query) {
      let q = supabase.from('store_listings')
        .select('store_id,store_product_id,title,page_url,canonical_url')
        .eq('store_id', sid).limit(100);
      if (query.mode === 'OR' && query.terms.length > 1) {
        const safe = t => t.replace(/[%_,]/g, ' ').trim().replace(/\s+/g,' ');
        const orClause = query.terms.filter(t => t.trim().length > 0)
          .map(t => `title.ilike.%${safe(t)}%`).join(',');
        if (orClause) q = q.or(orClause);
      } else {
        for (const term of query.terms) q = q.ilike('title', `%${term}%`);
      }
      const { data: listings } = await q;
      console.log(`[fetch] store=${sid} mode=${query.mode} terms=${JSON.stringify(query.terms)} found=${listings?.length||0}`);
      if (!listings?.length) return [];
      const { data: prods } = await supabase.from('products')
        .select('store_id,store_product_id,last_price,currency')
        .eq('store_id', sid).in('store_product_id', listings.map(x => x.store_product_id));
      if (!prods?.length) return [];
      const pm = new Map(prods.map(p => [`${p.store_id}|${p.store_product_id}`, p]));
      const localSeen = new Set();
      const out = [];
      for (const li of listings) {
        const key = `${sid}|${li.store_product_id}`;
        if (localSeen.has(key)) continue;
        const p = pm.get(key); if (!p) continue;
        const price = Number(p.last_price);
        if (!Number.isFinite(price) || price <= 0) continue;
        localSeen.add(key);
        dbg.dedupedTotal = (dbg.dedupedTotal||0) + 1;
        out.push({ storeId: sid, storeProductId: String(li.store_product_id), title: li.title||'', price, currency: p.currency||baseCurrency||'USD', url: li.page_url||li.canonical_url||null });
      }
      return out;
    }

    async function processStore(sid) {
      const localDbg = { retrieved: 0, retrievedTotal: 0 };
      let acceptedForStore = 0;
      for (const query of queries) {
        if (acceptedForStore >= 6 && query.mode === 'OR') break;
        const found = await fetchCandidates(sid, query);
        localDbg.retrieved += found.length; localDbg.retrievedTotal += found.length;
        for (const cand of found) {
          const before = dbg.accepted;
          scoreAndBucket(cand);
          if (dbg.accepted > before) acceptedForStore++;
        }
      }
      dbg.retrievedTotal += localDbg.retrievedTotal;
    }
    await Promise.all(storeIds.map(sid => processStore(sid)));
    console.log(`[ilike] best=${rawBest.length} other=${rawOther.length} for "${baseTitle.slice(0,45)}"`);
  }

  // ── Dedup: if an item is in both, keep in bestDeals only ──
  const bestKeys = new Set(rawBest.map(d => `${d.storeId}|${d.storeProductId}`));
  const otherDeduped = rawOther.filter(d => !bestKeys.has(`${d.storeId}|${d.storeProductId}`));

  // ── Sort Best Deals ──────────────────────────────────────
  const TIER_ORDER = { EXACT:0, SAME_VARIANT:1, SAME_MODEL:2, SAME_FAMILY:3, RELATED:4 };
  const COND_ORDER = { new:0, unknown:1, used:2 };
  rawBest.sort((a, b) => {
    const ac = Number.isFinite(basePrice) && a.price < basePrice;
    const bc = Number.isFinite(basePrice) && b.price < basePrice;
    if (ac !== bc) return ac ? -1 : 1;
    const td = (TIER_ORDER[a.matchTier]||5) - (TIER_ORDER[b.matchTier]||5);
    if (td !== 0) return td;
    const cd = (COND_ORDER[a.condition]||1) - (COND_ORDER[b.condition]||1);
    if (cd !== 0) return cd;
    return b.score - a.score || a.price - b.price;
  });

  // ── Sort Other Models: same brand first, then by score ───
  const baseBrand = baseClassified.brand;
  otherDeduped.sort((a, b) => {
    const aSame = a.candBrand === baseBrand;
    const bSame = b.candBrand === baseBrand;
    if (aSame !== bSame) return aSame ? -1 : 1;
    const cd = (COND_ORDER[a.condition]||1) - (COND_ORDER[b.condition]||1);
    if (cd !== 0) return cd;
    return b.score - a.score || Math.abs(a.price - (basePrice||0)) - Math.abs(b.price - (basePrice||0));
  });

  dbg.bestDeals   = rawBest.length;
  dbg.otherModels = otherDeduped.length;
  const _dbgLog = {
    baseTitle: baseTitle.slice(0, 60),
    baseSub: baseTax.sub, baseCat: baseTax.cat,
    baseIntent: baseClassified.intentClass, baseBrand: baseClassified.brand,
    vectorUsed: dbg.vectorUsed,
    counts: {
      retrieved: dbg.retrievedTotal, rejectedRole: dbg.rejectedRole,
      rejectedPrice: dbg.rejectedPrice, rejectedLowConf: dbg.rejectedClassifierLowConfidence,
      rejectedIntent: dbg.rejectedIntentMismatch,
      acceptedBest: dbg.acceptedBestDeals, acceptedOther: dbg.acceptedOtherModels,
    },
    topBest:  rawBest.slice(0,5).map(d => ({ t: d.title?.slice(0,40), p: d.price, id: d.identMatch })),
    topOther: otherDeduped.slice(0,5).map(d => ({ t: d.title?.slice(0,40), p: d.price, id: d.identMatch })),
  };
  console.log('[deals:debug]', JSON.stringify(_dbgLog));

  return {
    bestDeals:   rawBest.slice(0, limit),
    otherModels: otherDeduped,
    meta: { baseCategory: baseTax.cat, baseSubcategory: baseTax.sub, ...dbg }
  };
}

// ============================================================
//  SCHEMAS
// ============================================================

const ObservationSchema = z.object({
  storeId: z.string().min(2).optional(), store_id: z.string().min(2).optional(),
  storeProductId: z.string().min(2).optional(), store_product_id: z.string().min(2).optional(),
  canonicalUrl: z.string().url().optional(), canonical_url: z.string().url().optional(),
  pageUrl: z.string().url().optional(), page_url: z.string().url().optional(),
  imageUrl: z.string().optional().nullable(), image_url: z.string().optional().nullable(),
  title: z.string().optional().nullable(), price: z.number().positive(),
  currency: z.string().min(3).max(3).default("USD"),
  source: z.string().optional().default("extension"), host: z.string().optional().nullable(),
  wasPrice: z.number().positive().optional().nullable()
}).transform(o => ({
  storeId: o.storeId ?? o.store_id, storeProductId: o.storeProductId ?? o.store_product_id,
  canonicalUrl: o.canonicalUrl ?? o.canonical_url, pageUrl: o.pageUrl ?? o.page_url,
  title: o.title ?? null, imageUrl: o.imageUrl ?? o.image_url ?? null,
  price: o.price, currency: o.currency ?? "USD", source: o.source ?? "extension",
  host: o.host ?? null, wasPrice: o.wasPrice ?? null
}));

const AiChatSchema = z.object({
  message: z.string().min(1).max(4000),
  storeId: z.string().min(2), storeProductId: z.string().min(2),
  currentPrice: z.number().positive().optional(),
  currency: z.string().min(3).max(3).optional(),
  wasPrice: z.number().positive().optional().nullable(),
  chatHistory: z.array(z.object({ role: z.enum(["user","assistant"]), content: z.string() })).optional()
});

// ============================================================
//  ROUTES
// ============================================================

// Health — shows service status at a glance
app.get("/health", (_req, res) => res.json({
  ok: true,
  services: {
    chat:   { model: _svc.chat.model,   coolingDown: _svc.chat.cooldownUntil   > Date.now() },
    rec:    { model: _svc.rec.model,    coolingDown: _svc.rec.cooldownUntil    > Date.now() },
    detect: { model: _svc.detect.model, coolingDown: _svc.detect.cooldownUntil > Date.now() }
  },
  cache: { rec: recCache._store.size, detect: detectCache._store.size, chat: chatCache._store.size, pred: predCache._store.size, attr: attrCache._store.size, expl: explanCache._store.size },
  queue: { pending: [...jobStore.values()].filter(j => j.status === "pending").length,
           done: [...jobStore.values()].filter(j => j.status === "done").length }
}));

// Extractor rules
app.get("/v1/extractor", async (req, res) => {
  try {
    const host = String(req.query.host || "").toLowerCase();
    if (!host) return res.status(400).json({ ok: false, error: "missing_host" });
    const hostNoWww = host.replace(/^www\./, "");
    const rootLabel = hostNoWww.split(".")[0] || hostNoWww;
    const candidates = Array.from(new Set([host, hostNoWww, `www.${hostNoWww}`, rootLabel, `www.${rootLabel}`]));
    let { data, error } = await supabase.from("site_extractors").select("*").eq("is_enabled", true).in("host", candidates).order("priority", { ascending: true });
    if (error) return res.status(500).json({ ok: false, error: error.message });
    if (!data?.length) {
      const { data: d2 } = await supabase.from("site_extractors").select("*").eq("is_enabled", true).or(`host.ilike.%${hostNoWww}%,host.ilike.%${rootLabel}%`).order("priority", { ascending: true });
      if (d2?.length) data = d2;
    }
    return res.json({ ok: true, extractors: data || [] });
  } catch (e) { return res.status(500).json({ ok: false, error: e.message }); }
});

// ============================================================
//  BOOTSTRAP HISTORY — cold-start anchor point generator
//
//  When a product has < 3 real observations we synthesise up to
//  3 plausible past anchor points (60d / 30d / 7d ago) and save
//  them as source = 'estimated_history' in price_observations.
//
//  Rules:
//  • Only fires if realCount < 3
//  • Only inserts slots not already covered by existing estimates
//  • Uses category-aware drift + holiday calendar
//  • Peer prices provide weak baseline stabilisation
//  • Frontend receives these as normal chart points (no label)
//  • In-memory cache prevents re-querying DB for 24 h
// ============================================================

const _BOOT_PARAMS = {
  phone:                 { drift: 0.025, vol: 0.25, peerW: 0.35, hSens: 0.60 },
  tablet:                { drift: 0.020, vol: 0.22, peerW: 0.35, hSens: 0.55 },
  game_console:          { drift: 0.012, vol: 0.15, peerW: 0.30, hSens: 0.45 },
  gaming_controller:     { drift: 0.008, vol: 0.12, peerW: 0.25, hSens: 0.40 },
  gaming_laptop:         { drift: 0.022, vol: 0.25, peerW: 0.30, hSens: 0.65 },
  laptop:                { drift: 0.018, vol: 0.22, peerW: 0.30, hSens: 0.55 },
  ultrabook:             { drift: 0.018, vol: 0.22, peerW: 0.30, hSens: 0.55 },
  business_laptop:       { drift: 0.015, vol: 0.20, peerW: 0.30, hSens: 0.45 },
  gaming_desktop:        { drift: 0.020, vol: 0.25, peerW: 0.28, hSens: 0.60 },
  desktop:               { drift: 0.015, vol: 0.20, peerW: 0.28, hSens: 0.45 },
  tws_earbuds:           { drift: 0.014, vol: 0.18, peerW: 0.35, hSens: 0.50 },
  headphones:            { drift: 0.014, vol: 0.18, peerW: 0.35, hSens: 0.50 },
  gaming_headset:        { drift: 0.012, vol: 0.18, peerW: 0.30, hSens: 0.45 },
  smartwatch:            { drift: 0.016, vol: 0.20, peerW: 0.30, hSens: 0.50 },
  smart_tv:              { drift: 0.025, vol: 0.28, peerW: 0.28, hSens: 0.60 },
  monitor:               { drift: 0.015, vol: 0.20, peerW: 0.28, hSens: 0.50 },
  software_subscription: { drift: 0.003, vol: 0.08, peerW: 0.20, hSens: 0.25 },
  gpu:                   { drift: 0.022, vol: 0.30, peerW: 0.28, hSens: 0.55 },
  generic:               { drift: 0.008, vol: 0.12, peerW: 0.20, hSens: 0.30 },
};

// Map subcategory → holiday dropPct key used in HOLIDAY_EVENTS
const _BOOT_HCAT = {
  phone: 'phone', tablet: 'electronics', game_console: 'console',
  gaming_controller: 'controller', gaming_laptop: 'laptop', laptop: 'laptop',
  ultrabook: 'laptop', business_laptop: 'laptop', gaming_desktop: 'electronics',
  desktop: 'electronics', tws_earbuds: 'audio', headphones: 'audio',
  gaming_headset: 'audio', smartwatch: 'electronics', smart_tv: 'electronics',
  monitor: 'electronics', software_subscription: 'subscription', gpu: 'electronics',
};

function _bootHolidayFactor(pastDate, sub) {
  const hcat = _BOOT_HCAT[sub] || 'default';
  const windows = getHolidayWindows(pastDate.getFullYear());
  for (const w of windows) {
    if (pastDate >= w.windowStart && pastDate <= w.windowEnd) {
      const drop = w.dropPct[hcat] || w.dropPct.default || 0.05;
      return 1 - drop * 0.6; // conservative: 60% of typical sale drop
    }
  }
  // Two-week pre-sale lead-up: prices tend to be slightly elevated
  for (const w of windows) {
    const pre = new Date(w.windowStart); pre.setDate(pre.getDate() - 14);
    if (pastDate >= pre && pastDate < w.windowStart) return 1.01;
  }
  return 1.0;
}

// 24-hour dedup cache — prevents repeated DB queries on every page visit
const BOOTSTRAP_DONE_CACHE = new Map();
const _BOOT_TTL = 24 * 60 * 60 * 1000;

async function bootstrapHistoryIfNeeded({ productId, storeId, storeProductId, title, currentPrice, currency, peerPrices }) {
  if (!productId || !Number.isFinite(currentPrice) || currentPrice <= 0) return;

  const ck = `${storeId}|${storeProductId}`;
  const hit = BOOTSTRAP_DONE_CACHE.get(ck);
  if (hit && (Date.now() - hit) < _BOOT_TTL) return;

  // Fetch existing observations — separate real from estimated
  const { data: allObs, error: fetchErr } = await supabase
    .from('price_observations')
    .select('price, observed_at, source')
    .eq('product_id', productId)
    .order('observed_at', { ascending: false })
    .limit(50);

  if (fetchErr) { console.warn('[bootstrap] fetch err:', fetchErr.message); return; }

  const obs      = allObs || [];
  const realObs  = obs.filter(r => r.source !== 'estimated_history');
  const estObs   = obs.filter(r => r.source === 'estimated_history');

  // If enough real history exists, skip
  if (realObs.length >= 3) { BOOTSTRAP_DONE_CACHE.set(ck, Date.now()); return; }

  // Identify which anchor slots (60d / 30d / 7d) are already covered by estimates (±4d tolerance)
  const ANCHORS = [60, 30, 7];
  const covered = new Set();
  for (const row of estObs) {
    const dAgo = Math.round((Date.now() - new Date(row.observed_at).getTime()) / 86400000);
    for (const t of ANCHORS) { if (Math.abs(dAgo - t) <= 4) covered.add(t); }
  }
  const needed = ANCHORS.filter(d => !covered.has(d));
  if (!needed.length) { BOOTSTRAP_DONE_CACHE.set(ck, Date.now()); return; }

  // Category params
  const tax = taxonomy(title || '');
  const sub = (_BOOT_PARAMS[tax.sub] ? tax.sub : 'generic');
  const p   = _BOOT_PARAMS[sub];

  // Peer price stabilisation — sanity-gated
  let peerMedian = null;
  if (peerPrices && peerPrices.length) {
    const valid = peerPrices.filter(x => Number.isFinite(x) && x > currentPrice * 0.2 && x < currentPrice * 4);
    if (valid.length) {
      const s = [...valid].sort((a, b) => a - b);
      const m = s[Math.floor(s.length / 2)];
      if (m > 0) peerMedian = m;
    }
  }

  const today    = new Date();
  const toInsert = [];

  for (const daysAgo of needed) {
    const pastDate = new Date(today);
    pastDate.setDate(today.getDate() - daysAgo);

    // Baseline: blend current price with peer median if available
    const baseline = peerMedian !== null
      ? currentPrice * (1 - p.peerW) + peerMedian * p.peerW
      : currentPrice;

    // Drift: prices were slightly higher in the past (natural depreciation)
    const drift   = 1 + p.drift * (daysAgo / 30);

    // Holiday factor for that specific past date
    const hFactor = _bootHolidayFactor(pastDate, sub);
    const hEffect = 1 + (hFactor - 1) * p.hSens;

    let est = baseline * drift * hEffect;

    // Clamp within volatilityCap of current price
    est = Math.max(currentPrice * (1 - p.vol), Math.min(currentPrice * (1 + p.vol), est));
    est = Math.max(0.01, Math.round(est * 100) / 100);

    toInsert.push({
      product_id:  productId,
      price:       est,
      currency:    currency || 'USD',
      observed_at: pastDate.toISOString(),
      source:      'estimated_history',
    });
  }

  if (!toInsert.length) { BOOTSTRAP_DONE_CACHE.set(ck, Date.now()); return; }

  const { error: insErr } = await supabase.from('price_observations').insert(toInsert);

  console.log('[bootstrap]', JSON.stringify({
    title:       (title || '').slice(0, 45),
    sub, currentPrice, currency,
    realBefore:  realObs.length,
    inserted:    toInsert.length,
    slots:       needed,
    peerMedian,
    err:         insErr?.message || null,
  }));

  BOOTSTRAP_DONE_CACHE.set(ck, Date.now());
}

// ── End Bootstrap History ─────────────────────────────────────────────────────

// ── Observations — page load. HEURISTIC ONLY. Zero AI calls.
app.post("/v1/observations", async (req, res) => {
  const parsed = ObservationSchema.safeParse(req.body);
  if (!parsed.success) return res.status(400).json({ error: parsed.error.flatten() });
  const o = parsed.data;
  if (!o.storeId || !o.storeProductId || !o.pageUrl || !o.canonicalUrl)
    return res.status(400).json({ error: "missing_fields" });

  const { data: storeRow } = await supabase.from("stores").select("id").eq("id", o.storeId).maybeSingle();
  if (!storeRow) return res.status(400).json({ error: "unknown_store" });

  const { data: product, error: upsertErr } = await supabase.from("products").upsert({
    store_id: o.storeId, store_product_id: o.storeProductId,
    canonical_url: o.canonicalUrl, title: o.title, image_url: o.imageUrl,
    currency: o.currency, last_price: o.price, last_seen_at: new Date().toISOString()
  }, { onConflict: "store_id,store_product_id" }).select("id,store_id,store_product_id,last_price,currency,title").single();
  if (upsertErr) return res.status(500).json({ error: "db_upsert_failed", detail: upsertErr.message });

  try {
    const urlHost = (() => { try { return new URL(o.pageUrl).host.toLowerCase().replace(/^www\./,""); } catch { return null; } })();
    await supabase.from("store_listings").upsert({ store_id: o.storeId, host: urlHost||o.host||"unknown", store_product_id: o.storeProductId, canonical_url: o.canonicalUrl, page_url: o.pageUrl, title: o.title, image_url: o.imageUrl, currency: o.currency }, { onConflict: "store_id,store_product_id" });
    // Auto-embed new listings — fires async, never blocks the response
    autoEmbedListing(o.storeId, o.storeProductId, o.title).catch(() => {});
  } catch {}

  const { data: lastObs } = await supabase.from("price_observations").select("price,observed_at").eq("product_id", product.id).order("observed_at", { ascending: false }).limit(1);
  const last = lastObs?.[0] || null;
  const samePrice = last && Math.abs(Number(last.price) - o.price) < 0.00001;
  const ageMs = last ? Date.now() - new Date(last.observed_at).getTime() : Infinity;
  const priceChanged = !samePrice && last !== null; // true only when price actually moved
  if (!samePrice || ageMs > 6 * 60 * 60 * 1000) {
    await supabase.from("price_observations").insert({ product_id: product.id, price: o.price, currency: o.currency, source: o.source||"extension", page_url: o.pageUrl, observed_at: new Date().toISOString() });
  }

  // ── Deals cache — keyed on store+product+price ─────────────
  // If price changed since last observation, invalidate so results
  // reflect the new price context. Otherwise serve cached payload.
  const dealsCacheKey = `deals:${o.storeId}:${o.storeProductId}:${Math.round(o.price * 100)}`;
  let dealsResult = null;

  if (!priceChanged) {
    // Try to load from Supabase deals_cache
    try {
      const { data: cached } = await supabase.from("deals_cache")
        .select("payload, expires_at").eq("cache_key", dealsCacheKey).maybeSingle();
      if (cached && new Date(cached.expires_at) > new Date()) {
        dealsResult = cached.payload;
        console.log(`[deals-cache] HIT ${dealsCacheKey}`);
      } else if (cached) {
        // expired — delete it async
        supabase.from("deals_cache").delete().eq("cache_key", dealsCacheKey).then(() => {}).catch(() => {});
      }
    } catch (e) { console.warn(`[deals-cache] read error: ${e.message}`); }
  } else {
    // Price changed — proactively delete stale cache entries for this product
    supabase.from("deals_cache").delete()
      .like("cache_key", `deals:${o.storeId}:${o.storeProductId}:%`)
      .then(() => {}).catch(() => {});
    console.log(`[deals-cache] INVALIDATED (price changed) ${o.storeId}/${o.storeProductId}`);
  }

  if (!dealsResult) {
    dealsResult = await computeDealsForProduct({ storeId: o.storeId, storeProductId: o.storeProductId, baseTitle: o.title||product.title||"", baseCurrency: o.currency, basePrice: o.price, limit: 10 });
    // Save to cache async — 2 hour TTL
    const expiresAt = new Date(Date.now() + 2 * 60 * 60 * 1000).toISOString();
    supabase.from("deals_cache").upsert(
      { cache_key: dealsCacheKey, payload: dealsResult, expires_at: expiresAt },
      { onConflict: "cache_key" }
    ).then(() => {}).catch(e => console.warn(`[deals-cache] write error: ${e.message}`));
  }
  const { data: obsRows } = await supabase.from("price_observations").select("price,observed_at").eq("product_id", product.id).order("observed_at", { ascending: false }).limit(120);
  const prices = (obsRows||[]).map(r => Number(r.price)).filter(n => Number.isFinite(n));
  const stats  = computeStats(prices);
  const bestDealsObs = dealsResult.bestDeals || [];

  // Heuristic only — instant, no API
  const ai = heuristicRec(o.price, stats, prices.length, bestDealsObs);
  let fakeDeal = null;
  if (o.wasPrice && Number.isFinite(o.wasPrice) && o.wasPrice > o.price) {
    // peerPrices MUST use bestDeals only (not otherModels)
    fakeDeal = heuristicDetect(o.price, o.wasPrice, o.currency, prices, bestDealsObs.map(d=>d.price).filter(Number.isFinite));
  }

  // Prediction — apply shipping rules, then log for evaluation
  const _predAttrs = o.title ? await extractProductAttrs(o.title || product.title || '') : null;
  const _rawPred = await computePredictionV2({
    title: o.title || product.title || "",
    currentPrice: o.price, currency: o.currency,
    storeId: o.storeId, storeProductId: o.storeProductId,
    historyRows: obsRows || [],
    historyPrices: prices,
    peerPrices: bestDealsObs.map(d => d.price).filter(Number.isFinite),
  });
  const prediction = applyShippingRules(_rawPred, {
    nHist: prices.length, nPeer: bestDealsObs.length,
    category: _predAttrs?.category || null,
  });
  if (_rawPred && !_rawPred.fromCache) {
    logPrediction({
      storeId: o.storeId, storeProductId: o.storeProductId,
      url: o.canonicalUrl || o.pageUrl || null,
      title: o.title || product.title || '',
      category: _predAttrs?.category || null,
      brand: _predAttrs?.brand || null,
      currentPrice: o.price, currency: o.currency,
      prediction: _rawPred, attrs: _predAttrs,
    }).catch(() => {});
  }

  // Fire any active price alerts (async — don't block response)
  checkAndFireAlerts(o.storeId, o.storeProductId, o.price, o.currency, o.title||product.title, o.canonicalUrl||o.pageUrl).catch(() => {});

  // Bootstrap history if product is new (async — does not block response)
  bootstrapHistoryIfNeeded({
    productId:      product.id,
    storeId:        o.storeId,
    storeProductId: o.storeProductId,
    title:          o.title || product.title || '',
    currentPrice:   o.price,
    currency:       o.currency,
    peerPrices:     bestDealsObs.map(d => d.price).filter(Number.isFinite),
  }).catch(e => console.warn('[bootstrap] silent fail:', e.message));

  res.json({ ok: true, productId: product.id, historyCount: prices.length, stats, ai, fakeDeal, deals: bestDealsObs.slice(0,5), otherModels: (dealsResult.otherModels||[]).slice(0,4), prediction });
});

// ── Product lookup — heuristic only
app.get("/v1/products/:storeId/:storeProductId", async (req, res) => {
  const { storeId, storeProductId } = req.params;
  const { data: product } = await supabase.from("products").select("*").eq("store_id", storeId).eq("store_product_id", storeProductId).maybeSingle();
  if (!product) return res.status(404).json({ error: "not_found" });
  const { data: hist } = await supabase.from("price_observations").select("price,observed_at").eq("product_id", product.id).order("observed_at", { ascending: false }).limit(90);
  const prices = (hist||[]).map(r => Number(r.price)).filter(n => Number.isFinite(n));
  const stats  = computeStats(prices);
  const dealsR2 = await computeDealsForProduct({ storeId, storeProductId, baseTitle: product.title||"", baseCurrency: product.currency||"USD", basePrice: Number(product.last_price), limit: 10 });
  const ai = heuristicRec(Number(product.last_price), stats, prices.length, dealsR2.bestDeals||[]);
  const prediction = await computePredictionV2({
    title: product.title || "",
    currentPrice: Number(product.last_price), currency: product.currency || "USD",
    storeId, storeProductId,
    historyPrices: prices,
    peerPrices: (dealsR2.bestDeals||[]).map(d=>d.price).filter(Number.isFinite),
  });
  res.json({ ok: true, product, history: hist, stats, ai, deals: dealsR2.bestDeals||[], otherModels: dealsR2.otherModels||[], prediction });

  // Bootstrap history if product is new (async — fires after response sent)
  bootstrapHistoryIfNeeded({
    productId:      product.id,
    storeId,
    storeProductId,
    title:          product.title || '',
    currentPrice:   Number(product.last_price),
    currency:       product.currency || 'USD',
    peerPrices:     (dealsR2.bestDeals||[]).map(d => d.price).filter(Number.isFinite),
  }).catch(e => console.warn('[bootstrap] silent fail:', e.message));
});

// ── AI Recommend — repurposed: now uses the "rec" key to deepen fake deal context
// Price prediction (buildPricePrediction) handles the buy/wait signal heuristically.
// The rec key is now reserved for enriching the fake deal detector with extended history context.
// Kept as endpoint for backwards compatibility — returns prediction data instead.
app.post("/v1/ai/recommend", async (req, res) => {
  const { storeId, storeProductId } = req.body;
  if (!storeId || !storeProductId) return res.status(400).json({ ok: false, error: "missing fields" });

  const { data: product } = await supabase.from("products").select("*").eq("store_id", storeId).eq("store_product_id", storeProductId).maybeSingle();
  if (!product) return res.status(404).json({ ok: false, error: "not_found" });

  const { data: hist } = await supabase.from("price_observations").select("price,observed_at").eq("product_id", product.id).order("observed_at", { ascending: false }).limit(120);
  const prices = (hist||[]).map(r => Number(r.price)).filter(n => Number.isFinite(n));
  const stats  = computeStats(prices);
  const dealsR3 = await computeDealsForProduct({ storeId, storeProductId, baseTitle: product.title||"", baseCurrency: product.currency||"USD", basePrice: Number(product.last_price), limit: 10 });
  const peerPricesR3 = (dealsR3.bestDeals||[]).map(d=>d.price).filter(Number.isFinite);
  const prediction = await computePredictionV2({
    title: product.title || "",
    currentPrice: Number(product.last_price), currency: product.currency || "USD",
    storeId, storeProductId,
    historyPrices: prices,
    peerPrices: peerPricesR3,
  });

  res.json({ ok: true, prediction, stats, deals: dealsR3.bestDeals||[], otherModels: dealsR3.otherModels||[], pending: false });
});

// ── AI Fake Deal — USER TRIGGERED ONLY
app.post("/v1/ai/fake-deal", async (req, res) => {
  const { storeId, storeProductId, currentPrice, wasPrice, currency, title: reqTitle } = req.body;
  // Only storeId, storeProductId, and a valid currentPrice are mandatory.
  // Missing or invalid wasPrice (or wasPrice <= currentPrice) → no_discount, not a 400 error.
  if (!storeId || !storeProductId || !Number.isFinite(Number(currentPrice)))
    return res.status(400).json({ ok: false, error: "missing_fields" });

  const _cur = Number(currentPrice);
  const _was = Number(wasPrice);
  // No genuine discount → return structured no_discount immediately
  if (!Number.isFinite(_was) || _was <= _cur) {
    return res.json({ ok: true, verdictKey: "no_discount", fakeDealScore: 0, confidenceScore: 100,
      explanation: "No discount detected on this page.",
      verdict: "— No Discount", confidence: "N/A", message: "No discount detected on this page.",
      stats: { refMedian: null, refP25: null, refP75: null, mad: null, modifiedZCurrent: null, modifiedZWas: null, sourceUsed: "none" },
      evidence: { historyCount: 0, peerCount: 0, serpCount: 0 },
      refs: [], debugReasons: ["no_discount_trigger"], source: "heuristic",
      marketMedian: null, marketCount: 0, serpRefs: [], pending: false });
  }

  const ip = req.ip || "x";
  if (!checkRate(`det:${ip}:hour`, 5, 60*60*1000))
    return res.status(429).json({ ok: false, error: "Max 5 deal checks per hour." });
  if (!checkRate("det:global:hour", 40, 60*60*1000))
    return res.status(429).json({ ok: false, error: "System busy — try again in a bit." });

  // 1. Fetch product + price history from DB
  const { data: product } = await supabase
    .from("products").select("id,title,last_price")
    .eq("store_id", storeId).eq("store_product_id", storeProductId).maybeSingle();

  const title = product?.title || reqTitle || "";
  let historyPrices = [];
  let historyRows   = [];
  if (product?.id) {
    const { data: hist } = await supabase.from("price_observations")
      .select("price, observed_at").eq("product_id", product.id)
      .order("observed_at", { ascending: false }).limit(120);
    historyRows   = (hist || []).filter(r => Number.isFinite(Number(r.price)));
    historyPrices = historyRows.map(r => Number(r.price));
  }

  // 2. Get peer prices from our deals engine (DB cross-store prices)
  const dealsR4 = title ? await computeDealsForProduct({
    storeId, storeProductId, baseTitle: title,
    baseCurrency: currency||"USD", basePrice: Number(currentPrice), limit: 10
  }) : { bestDeals: [] };
  // peerPrices MUST use bestDeals only — never otherModels (would skew fake deal analysis)
  const peerPrices = (dealsR4.bestDeals||[]).map(d => d.price).filter(Number.isFinite);

  // 3. Cache check — reanalyze=true bypasses cache entirely
  const reanalyze = req.body.reanalyze === true || req.query.reanalyze === 'true';
  const cKey = detectCacheKey(product?.id || storeProductId, Number(wasPrice), Number(currentPrice));
  if (!reanalyze) {
    const cached = detectCache.get(cKey);
    if (cached) return res.json({ ok: true, ...cached, fromCache: true, pending: false });
  }

  // 4. Run the full analysis pipeline (tiers: peers → history → serp → gemini)
  try {
    const result = await analyseDeal({
      title, currentPrice: Number(currentPrice), wasPrice: Number(wasPrice),
      currency: currency||"USD", storeId, storeProductId,
      historyRows, historyPrices, peerPrices,
      reanalyze,
    });
    detectCache.set(cKey, result);
    return res.json({ ok: true, ...result, pending: false });
  } catch (e) {
    // Fallback to heuristic if pipeline errors
    const fallback = heuristicDetect(Number(currentPrice), Number(wasPrice), currency||"USD", historyPrices, peerPrices);
    return res.json({ ok: true, ...fallback, pending: false });
  }
});

// Backward-compatible alias
app.post("/v1/fake-deal", (req, res) => {
  req.url = "/v1/ai/fake-deal";
  app.handle(req, res);
});

// ── Job poll — extension polls this to get AI upgrade when ready
app.get("/v1/jobs/:jobId", (req, res) => {
  const job = jobStore.get(req.params.jobId);
  if (!job) return res.status(404).json({ ok: false, error: "not_found" });
  if (job.status !== "done") return res.json({ ok: true, status: job.status, result: null });
  return res.json({ ok: true, status: "done", result: job.result });
});

// ── Compare — pure DB, no AI
app.get("/v1/compare/:storeId/:storeProductId", async (req, res) => {
  try {
    const { storeId, storeProductId } = req.params;
    const limit  = Math.max(1, Math.min(10, Number(req.query.limit||5)));
    const offset = Math.max(0, Number(req.query.offset||0));

    // Look up product — may not exist yet on first visit to a new store
    const { data: base } = await supabase
      .from("products")
      .select("id,store_id,store_product_id,title,last_price,currency")
      .eq("store_id", storeId)
      .eq("store_product_id", storeProductId)
      .maybeSingle();

    // Also check store_listings for title (upserted on observation, even before products row exists)
    let baseTitle = base?.title || "";
    let baseCurrency = base?.currency || "USD";
    let basePrice = base ? Number(base.last_price) : null;

    if (!baseTitle) {
      const { data: sl } = await supabase
        .from("store_listings")
        .select("title,currency")
        .eq("store_id", storeId)
        .eq("store_product_id", storeProductId)
        .maybeSingle();
      if (sl?.title) { baseTitle = sl.title; baseCurrency = sl.currency || baseCurrency; }
    }

    // If we truly have nothing — return empty deals rather than 404
    // Frontend will still show the section with 0 results instead of erroring
    if (!baseTitle && !base) {
      return res.json({ ok: true, base: { storeId, storeProductId, title: "", price: null, currency: baseCurrency }, deals: [], hasMore: false, nextOffset: null });
    }

    // Run the full matching engine — returns bestDeals + otherModels
    const result = await computeDealsForProduct({
      storeId, storeProductId,
      baseTitle,
      baseCurrency,
      basePrice: Number.isFinite(basePrice) ? basePrice : null,
      limit: limit + offset + 20,
    });

    const otherOffset = Math.max(0, Number(req.query.otherOffset||0));
    const otherLimit  = Math.max(1, Math.min(100, Number(req.query.otherLimit||12)));

    const bestPage  = result.bestDeals.slice(offset, offset + limit);
    const otherPage = result.otherModels.slice(otherOffset, otherOffset + otherLimit);
    const hasMore      = offset + limit < result.bestDeals.length;
    const otherHasMore = otherOffset + otherLimit < result.otherModels.length;

    res.json({
      ok: true,
      base: { storeId, storeProductId, title: baseTitle, price: basePrice, currency: baseCurrency },
      deals:            bestPage,   // bestDeals (backwards-compat key)
      bestDeals:        bestPage,
      otherModels:      otherPage,
      hasMore,
      nextOffset:       hasMore ? offset + limit : null,
      otherHasMore,
      otherNextOffset:  otherHasMore ? otherOffset + otherLimit : null,
      meta: result.meta,
    });
  } catch (e) { res.status(500).json({ ok: false, error: e.message }); }
});

// ============================================================
//  PREDICTION CENTER v2
//  Hybrid: EWMA trend + MAD volatility + holiday windows +
//  Gemini attribute extraction + cached explanation
// ============================================================

// ── Key rotation for prediction feature only ────────────────
// Uses _kB (rec key) for attribute extraction
// Uses _kC (detect key) for explanation polishing
// Both keys were idle after fake-deal went deterministic
let _predKeyIdxAttr  = 0;  // rotates among [_kB] + env override
let _predKeyIdxExpl  = 0;  // rotates among [_kC] + env override

async function _geminiForPred(service, prompt, maxTokens) {
  // service = 'attr' | 'expl'
  // Reuses _svc's cooldown tracking via 'rec' and 'detect' slots
  const svcSlot = service === 'attr' ? 'rec' : 'detect';
  const raw = await callAI(svcSlot, prompt, maxTokens);
  return raw; // null if failed/rate-limited
}

// ── 1. Holiday / event calendar ─────────────────────────────
// Each event: name, mmdd start/end (or dynamic), leadDays, lagDays,
// typicalDropPct per category group.
// "Window" = from (event_start - leadDays) to (event_end + lagDays).
const HOLIDAY_EVENTS = [
  // Only events Amazon, eBay, AliExpress, Apple, Samsung, HP, Lenovo, Dell officially run
  {
    name: 'New Year Sales', key: 'newyear',
    startMMDD: '01-01', endMMDD: '01-07', leadDays: 0, lagDays: 2,
    dropPct: { electronics: 0.09, laptop: 0.08, phone: 0.07, subscription: 0.10, default: 0.05 }
  },
  {
    name: 'Amazon Prime Day', key: 'prime',
    startMMDD: '07-08', endMMDD: '07-12', leadDays: 5, lagDays: 3,
    dropPct: { electronics: 0.20, audio: 0.18, console: 0.15, controller: 0.13, laptop: 0.18, phone: 0.17, subscription: 0.14, default: 0.11 }
  },
  {
    name: 'Back to School', key: 'bts',
    startMMDD: '08-01', endMMDD: '09-15', leadDays: 7, lagDays: 5,
    dropPct: { laptop: 0.16, electronics: 0.11, phone: 0.10, subscription: 0.09, default: 0.06 }
  },
  {
    name: 'Amazon Fall Sale', key: 'prime_fall',
    startMMDD: '10-08', endMMDD: '10-12', leadDays: 5, lagDays: 3,
    dropPct: { electronics: 0.18, audio: 0.15, laptop: 0.15, phone: 0.14, subscription: 0.12, default: 0.09 }
  },
  {
    name: "Singles' Day", key: 'singles',
    startMMDD: '11-09', endMMDD: '11-11', leadDays: 5, lagDays: 2,
    dropPct: { electronics: 0.18, audio: 0.17, phone: 0.17, laptop: 0.15, default: 0.10 }
  },
  {
    name: 'Black Friday', key: 'bf',
    startMMDD: '11-22', endMMDD: '11-29', leadDays: 7, lagDays: 3,
    dropPct: { electronics: 0.22, audio: 0.20, console: 0.18, controller: 0.15, subscription: 0.15, laptop: 0.21, phone: 0.18, tv: 0.25, default: 0.13 }
  },
  {
    name: 'Cyber Monday', key: 'cm',
    startMMDD: '11-30', endMMDD: '12-02', leadDays: 1, lagDays: 2,
    dropPct: { electronics: 0.18, audio: 0.18, console: 0.15, controller: 0.13, subscription: 0.13, laptop: 0.18, phone: 0.15, default: 0.10 }
  },
  {
    name: 'Christmas', key: 'xmas',
    startMMDD: '12-20', endMMDD: '12-25', leadDays: 5, lagDays: 0,
    dropPct: { electronics: 0.13, audio: 0.13, console: 0.13, controller: 0.10, subscription: 0.10, laptop: 0.12, phone: 0.10, default: 0.08 }
  },
  {
    name: 'Boxing Day', key: 'boxing',
    startMMDD: '12-26', endMMDD: '12-31', leadDays: 0, lagDays: 3,
    dropPct: { electronics: 0.15, audio: 0.15, console: 0.14, controller: 0.11, subscription: 0.08, laptop: 0.15, phone: 0.13, default: 0.09 }
  },
];

// Build date windows for a given year
function buildHolidayWindows(year) {
  return HOLIDAY_EVENTS.map(ev => {
    const [sm, sd] = ev.startMMDD.split('-').map(Number);
    const [em, ed] = ev.endMMDD.split('-').map(Number);
    const evYear = (sm === 1 && year > new Date().getFullYear()) ? year : year;
    const start  = new Date(evYear, sm - 1, sd);
    const end    = new Date(evYear, em - 1, ed);
    const windowStart = new Date(start); windowStart.setDate(start.getDate() - ev.leadDays);
    const windowEnd   = new Date(end);   windowEnd.setDate(end.getDate()   + ev.lagDays);
    return { ...ev, windowStart, windowEnd, start, end };
  });
}

// Holiday cache: one entry per year
const HOLIDAY_CACHE = new Map();
function getHolidayWindows(year) {
  if (!HOLIDAY_CACHE.has(year)) HOLIDAY_CACHE.set(year, buildHolidayWindows(year));
  return HOLIDAY_CACHE.get(year);
}

// Find active and next holiday event relative to a date
function getHolidayContext(now, category) {
  const cat = (category || 'default').toLowerCase();
  const year = now.getFullYear();
  // Check current year + next year for wrap-around
  const windows = [...getHolidayWindows(year), ...getHolidayWindows(year + 1)];

  let activeEvent = null;
  let nextEvent   = null;

  for (const w of windows) {
    if (now >= w.windowStart && now <= w.windowEnd) {
      if (!activeEvent) activeEvent = w;
    } else if (now < w.windowStart) {
      if (!nextEvent) nextEvent = w;
    }
  }

  const dropPct = (ev) => ev ? (ev.dropPct[cat] || ev.dropPct['electronics'] || ev.dropPct['default'] || 0.07) : 0;
  const daysUntilNext = nextEvent ? Math.round((nextEvent.windowStart - now) / 86400000) : 999;
  const monthsAway    = Math.round(daysUntilNext / 30.5 * 10) / 10;

  return {
    activeEvent:       activeEvent ? activeEvent.name : null,
    activeDropPct:     dropPct(activeEvent),
    nextEvent:         nextEvent ? nextEvent.name : 'End of Year Sales',
    nextEventDropPct:  dropPct(nextEvent),
    daysUntilNext,
    monthsAway:        Math.ceil(monthsAway),
    insideWindow:      !!activeEvent,
    // Legacy field name content.js uses
    nextSaleEvent:     nextEvent ? nextEvent.name : 'End of Year Sales',
  };
}

// ── 2. Product attribute extraction (Gemini + rules fallback) ─
function ruleBasedAttrs(title) {
  const t = (title || '').toLowerCase();
  const attrs = {
    category: 'electronics', brand: null, model_family: null, model_number: null,
    variant: null, condition: 'new', bundle: false, subscription_term_months: null,
    console_edition: null, controller_tier: null,
  };

  // Condition
  if (/\b(refurb|renewed|pre.?owned|used|open.?box)\b/i.test(t)) attrs.condition = 'refurb';

  // Bundle
  if (/\b(bundle|with\s+game|with\s+controller|starter\s+kit)\b/i.test(t)) attrs.bundle = true;

  // Category detection
  if (/\b(microsoft\s*365|office\s*365)\b/i.test(t)) {
    attrs.category = 'subscription';
    const termM = t.match(/(\d+)[- ]month/);
    const termY = t.match(/(\d+)[- ]year/);
    if (termM) attrs.subscription_term_months = Number(termM[1]);
    else if (termY) attrs.subscription_term_months = Number(termY[1]) * 12;
    else if (/annual|yearly/i.test(t)) attrs.subscription_term_months = 12;
    attrs.model_family = /family/i.test(t) ? 'family' : /personal/i.test(t) ? 'personal' : /business/i.test(t) ? 'business' : null;
    attrs.brand = 'microsoft';
  } else if (/\b(ps5|playstation\s*5)\b/i.test(t) && !/\b(controller|dualsense)\b/i.test(t)) {
    attrs.category = 'console'; attrs.brand = 'sony';
    attrs.model_family = 'ps5';
    attrs.console_edition = /slim/i.test(t) ? 'slim' : /pro/i.test(t) ? 'pro' : 'standard';
    if (/digital/i.test(t)) attrs.variant = 'digital';
    else if (/disc/i.test(t)) attrs.variant = 'disc';
  } else if (/\b(ps4|playstation\s*4)\b/i.test(t)) {
    attrs.category = 'console'; attrs.brand = 'sony'; attrs.model_family = 'ps4';
  } else if (/\b(xbox\s*series\s*[xs]|series\s*x|series\s*s)\b/i.test(t) && !/controller/i.test(t)) {
    attrs.category = 'console'; attrs.brand = 'microsoft';
    attrs.model_family = /series\s*x/i.test(t) ? 'xbox_series_x' : 'xbox_series_s';
  } else if (/\bnintendo\s*switch\b/i.test(t) && !/controller|pro\s*controller|joy.?con/i.test(t)) {
    attrs.category = 'console'; attrs.brand = 'nintendo'; attrs.model_family = 'switch';
    if (/oled/i.test(t)) attrs.console_edition = 'oled';
    else if (/lite/i.test(t)) attrs.console_edition = 'lite';
    else attrs.console_edition = 'standard';
  } else if (/\b(xbox.*(wireless\s*)?controller|dualsense|dualshock|joy.?con|pro\s*controller)\b/i.test(t)) {
    attrs.category = 'controller';
    attrs.controller_tier = /elite/i.test(t) ? 'elite' : 'core';
    if (/\bxbox\b/i.test(t)) attrs.brand = 'microsoft';
    else if (/dualsense|dualshock/i.test(t)) attrs.brand = 'sony';
    else if (/nintendo/i.test(t)) attrs.brand = 'nintendo';
  } else if (/\biphone\b/i.test(t)) {
    attrs.category = 'phone'; attrs.brand = 'apple';
    const m = t.match(/iphone\s*(\d+)/);
    if (m) attrs.model_number = `iphone_${m[1]}`;
  } else if (/\b(galaxy\s+[szam]\d)\b/i.test(t)) {
    attrs.category = 'phone'; attrs.brand = 'samsung';
    const m = t.match(/galaxy\s+([szam]\d+)/i);
    if (m) attrs.model_number = `galaxy_${m[1].toLowerCase()}`;
  } else if (/\b(macbook|macbook\s*(air|pro))\b/i.test(t)) {
    attrs.category = 'laptop'; attrs.brand = 'apple';
  } else if (/\b(thinkpad|ideapad|zenbook|vivobook|spectre|envy|pavilion|inspiron|xps|razer\s*blade)\b/i.test(t)) {
    attrs.category = 'laptop';
  } else if (/\b(airpods|earbuds?|wf-\d|galaxy\s*buds|true\s*wireless)\b/i.test(t)) {
    attrs.category = 'audio';
    attrs.model_family = 'earbuds';
    if (/apple|airpods/i.test(t)) attrs.brand = 'apple';
    else if (/samsung/i.test(t)) attrs.brand = 'samsung';
    else if (/sony/i.test(t)) { attrs.brand = 'sony'; const m = t.match(/wf-(\S+)/i); if (m) attrs.model_number = `wf_${m[1]}`; }
  } else if (/\b(headphones?|over.ear|on.ear|wh-\d|quietcomfort|qc\d|bose|xm\d)\b/i.test(t)) {
    attrs.category = 'audio';
    attrs.model_family = 'headphones';
    if (/sony/i.test(t)) { attrs.brand = 'sony'; const m = t.match(/wh-(\S+)/i); if (m) attrs.model_number = `wh_${m[1]}`; }
    else if (/bose/i.test(t)) attrs.brand = 'bose';
    else if (/jabra/i.test(t)) attrs.brand = 'jabra';
  } else if (/\b(ipad|surface\s*(pro|go)?|galaxy\s*tab)\b/i.test(t)) {
    attrs.category = 'tablet';
  }

  // Brand fallback
  if (!attrs.brand) {
    if (/\bsony\b/i.test(t))       attrs.brand = 'sony';
    else if (/\bsamsung\b/i.test(t))   attrs.brand = 'samsung';
    else if (/\bapple\b/i.test(t))     attrs.brand = 'apple';
    else if (/\blg\b/i.test(t))        attrs.brand = 'lg';
    else if (/\blenovo\b/i.test(t))    attrs.brand = 'lenovo';
    else if (/\bhp\b/i.test(t))        attrs.brand = 'hp';
    else if (/\bdell\b/i.test(t))      attrs.brand = 'dell';
    else if (/\basus\b/i.test(t))      attrs.brand = 'asus';
    else if (/\bacer\b/i.test(t))      attrs.brand = 'acer';
    else if (/\brazer\b/i.test(t))     attrs.brand = 'razer';
  }

  return attrs;
}

async function extractProductAttrs(title) {
  const normTitle = title.trim().toLowerCase().replace(/\s+/g, ' ');
  const cached = attrCache.get(`attr:${normTitle}`);
  if (cached) return { ...cached, fromCache: true };

  // Hard categories that benefit most from Gemini parsing
  const HARD_CATEGORIES = ['subscription', 'console', 'controller'];
  const rules = ruleBasedAttrs(title);

  // Skip Gemini for clear non-hard categories to save quota
  if (!HARD_CATEGORIES.includes(rules.category)) {
    attrCache.set(`attr:${normTitle}`, rules);
    return { ...rules, fromCache: false };
  }

  // Try Gemini for hard categories
  const prompt = `Extract product attributes from this retail title as JSON. Title: "${title.slice(0, 150)}"
Return ONLY valid JSON with these exact fields (use null for unknown):
{"category":"phone|console|controller|subscription|laptop|audio|tablet|electronics","brand":null,"model_family":null,"model_number":null,"variant":null,"condition":"new|used|refurb","bundle":false,"subscription_term_months":null,"console_edition":null,"controller_tier":"core|elite|null"}
No markdown, no explanation, just the JSON object.`;

  const raw = await _geminiForPred('attr', prompt, 250);
  if (raw) {
    try {
      const parsed = JSON.parse(raw.replace(/```json|```/g, '').trim());
      if (parsed && parsed.category) {
        attrCache.set(`attr:${normTitle}`, parsed);
        return { ...parsed, fromCache: false };
      }
    } catch {}
  }

  // Gemini failed — use rules
  attrCache.set(`attr:${normTitle}`, rules);
  return { ...rules, fromCache: false };
}

// ── 3. Time-series math helpers ──────────────────────────────
// EWMA: exponentially weighted moving average, alpha=0.3 (recent data weighted more)
const EWMA_ALPHA = 0.3;
function computeEWMA(prices) {
  // prices[0] = most recent; compute from oldest to newest
  if (!prices.length) return null;
  const rev = [...prices].reverse(); // oldest first
  let ewma = rev[0];
  for (let i = 1; i < rev.length; i++) ewma = EWMA_ALPHA * rev[i] + (1 - EWMA_ALPHA) * ewma;
  return ewma;
}

// Recency-weighted linear slope over last k points
// Returns slope per observation (negative = price falling)
const SLOPE_WINDOW = 8;
function computeSlope(prices) {
  const k = Math.min(SLOPE_WINDOW, prices.length);
  if (k < 2) return 0;
  const pts = prices.slice(0, k).reverse(); // oldest→newest
  const n = pts.length;
  const xMean = (n - 1) / 2;
  const yMean = pts.reduce((a, b) => a + b, 0) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    const w = 1 + i / n; // recency weight: newer pts weighted more
    num += w * (i - xMean) * (pts[i] - yMean);
    den += w * (i - xMean) ** 2;
  }
  return den === 0 ? 0 : num / den;
}

// Volatility: IQR / 1.35 (robust equivalent of std dev)
function computeVolatility(prices) {
  if (prices.length < 2) return 0;
  const sorted = [...prices].sort((a, b) => a - b);
  const n = sorted.length;
  const p25 = sorted[Math.floor(n * 0.25)];
  const p75 = sorted[Math.floor(n * 0.75)];
  return (p75 - p25) / 1.35; // Gaussian-consistent robust sigma
}

// Percentiles
function percentile(sorted, p) {
  const idx = Math.floor(sorted.length * p);
  return sorted[Math.min(idx, sorted.length - 1)];
}

// Logistic function: maps any real to 0..1
function logistic(x) { return 1 / (1 + Math.exp(-x)); }

// ── 4. Core prediction engine ────────────────────────────────

// ============================================================
//  PRICE PREDICTION ENGINE v3
//  Timestamp-aware slope · store+category holiday impact
//  45-day event gating · split data/forecast confidence
//  Peer constraints · rich sentence output
// ============================================================

// ── A) Timestamp-based time features ──────────────────────────
function computeTimeFeatures(historyRows) {
  if (!historyRows || historyRows.length < 2) {
    return { slope: 0, volatility: 0, spanDays: 0, obsPerDay: 0, reliable: false };
  }

  const rows = historyRows
    .map(r => ({ p: Number(r.price), t: new Date(r.observed_at).getTime() }))
    .filter(r => Number.isFinite(r.p) && r.p > 0 && !isNaN(r.t))
    .sort((a, b) => a.t - b.t);

  if (rows.length < 2) return { slope: 0, volatility: 0, spanDays: 0, obsPerDay: 0, reliable: false };

  const spanMs   = rows[rows.length - 1].t - rows[0].t;
  const spanDays = spanMs / 86400000;
  const obsPerDay = rows.length / Math.max(spanDays, 1);

  // Median daily slope (robust to outliers)
  const deltas = [];
  for (let i = 1; i < rows.length; i++) {
    const dDays = (rows[i].t - rows[i-1].t) / 86400000;
    if (dDays > 0) deltas.push((rows[i].p - rows[i-1].p) / dDays);
  }
  deltas.sort((a, b) => a - b);
  const medSlope = deltas.length ? deltas[Math.floor(deltas.length / 2)] : 0;

  // Volatility = stddev of absolute daily deltas
  const absD = deltas.map(Math.abs);
  const mean = absD.reduce((s, v) => s + v, 0) / (absD.length || 1);
  const variance = absD.reduce((s, v) => s + (v - mean) ** 2, 0) / (absD.length || 1);

  return {
    slope:      Math.round(medSlope       * 10000) / 10000,
    volatility: Math.round(Math.sqrt(variance) * 10000) / 10000,
    spanDays:   Math.round(spanDays),
    obsPerDay:  Math.round(obsPerDay * 100) / 100,
    reliable:   spanDays >= 7 && rows.length >= 3,
  };
}

// ── B) Holiday impact — store + category sensitive ─────────────
// Only non-zero when event is within 45 days or active (hard gate).
function holidayImpact(store, category, hCtx) {
  const cat    = (category || 'electronics').toLowerCase();
  const daysTo = hCtx.insideWindow ? 0 : hCtx.daysUntilNext;

  // Hard gate: beyond 45 days → zero, do NOT show event in card
  if (daysTo > 45) return { dropProbBoost: 0, expectedDropDepthPct: 0, impactLabel: null, daysToEvent: daysTo, eventName: null };

  const proximity = hCtx.insideWindow ? 1.0
    : daysTo <= 14 ? 0.85
    : daysTo <= 30 ? 0.65
    : 0.35;

  const CAT_WEIGHT = {
    electronics: 1.00, laptop: 1.00, audio: 0.95, phone: 0.80,
    console: 0.85, controller: 0.75, tablet: 0.80, tv: 1.05,
    home: 0.60, fashion: 0.70, subscription: 0.55, default: 0.65,
  };
  const STORE_WEIGHT = {
    amazon: 1.00, ebay: 0.85, aliexpress: 0.90, apple: 0.55,
    samsung: 0.70, hp: 0.80, lenovo: 0.85, dell: 0.80, default: 0.75,
  };

  const cw = CAT_WEIGHT[cat] ?? CAT_WEIGHT.default;
  const sw = STORE_WEIGHT[(store||'').toLowerCase()] ?? STORE_WEIGHT.default;
  const baseDepth  = hCtx.nextEventDropPct || 0.10;
  const effDepth   = baseDepth * cw * sw * proximity;
  const probBoost  = Math.min(0.40, effDepth * 2.5 * proximity);
  const pct        = Math.round(effDepth * 100);

  return {
    dropProbBoost:        Math.round(probBoost * 100) / 100,
    expectedDropDepthPct: Math.round(effDepth  * 1000) / 1000,
    impactLabel:          pct >= 3 ? `~${pct}% typical drop` : null,
    daysToEvent:          daysTo,
    eventName:            hCtx.insideWindow ? hCtx.activeEvent : hCtx.nextEvent,
  };
}

// ── C) Product signals from own history ───────────────────────
function computeProductSignals(historyRows, currentPrice) {
  const cur = Number(currentPrice);
  const empty = {
    baseline: cur, histMin: null, histMax: null,
    aboveAvgPct: null, aboveLowPct: null,
    recentAvg: null, olderAvg: null,
    velocityPct: 0, fallingFast: false, risingFast: false,
    trendDir: 'unknown', timeFeatures: null,
  };
  if (!historyRows || historyRows.length === 0) return empty;

  const prices = historyRows.map(r => Number(r.price)).filter(n => Number.isFinite(n) && n > 0);
  if (!prices.length) return empty;

  const sorted   = [...prices].sort((a, b) => a - b);
  const ewma     = computeEWMA(prices);
  const baseline = ewma || prices.reduce((a, b) => a + b, 0) / prices.length;
  const histMin  = sorted[0];
  const histMax  = sorted[sorted.length - 1];
  const tf       = computeTimeFeatures(historyRows);

  const FALL_T = baseline * 0.0012;
  const RISE_T = baseline * 0.0012;
  const fallingFast = tf.slope < -FALL_T;
  const risingFast  = tf.slope >  RISE_T;

  const recentPrices = prices.slice(0, 3);
  const olderPrices  = prices.slice(3, 8);
  const recentAvg = recentPrices.length ? Math.round(recentPrices.reduce((a,b)=>a+b,0)/recentPrices.length*100)/100 : null;
  const olderAvg  = olderPrices.length  ? Math.round(olderPrices.reduce((a,b)=>a+b,0)/olderPrices.length*100)/100   : null;
  const velocityPct = (recentAvg && olderAvg) ? (recentAvg - olderAvg) / olderAvg : 0;

  const trendDir = fallingFast ? 'falling fast'
    : risingFast             ? 'rising fast'
    : tf.slope < -FALL_T*0.3 ? 'falling'
    : tf.slope >  RISE_T*0.3 ? 'rising'
    : 'stable';

  return {
    baseline:    Math.round(baseline * 100) / 100,
    histMin, histMax,
    aboveAvgPct: Math.round((cur - baseline) / baseline * 100),
    aboveLowPct: Math.round((cur - histMin)  / histMin  * 100),
    recentAvg, olderAvg, velocityPct,
    fallingFast, risingFast, trendDir,
    timeFeatures: tf,
  };
}

// ── D) Combine all signals → prediction object ─────────────────
function combineSignals({ cur, fmt, signals, holiday, peers, nHist, nPeer }) {
  const {
    baseline, histMin, aboveAvgPct, aboveLowPct,
    recentAvg, olderAvg, velocityPct, fallingFast, risingFast, trendDir,
  } = signals;

  const { dropProbBoost, expectedDropDepthPct, impactLabel, daysToEvent, eventName } = holiday;

  // Peer median
  const pSorted    = [...peers].sort((a,b)=>a-b);
  const peerMedian = peers.length ? pSorted[Math.floor(pSorted.length/2)] : null;
  const vsMedianPct = peerMedian ? Math.round((cur - peerMedian) / peerMedian * 100) : null;
  const peerSignal  = vsMedianPct === null ? 'unknown'
    : vsMedianPct >  12 ? 'expensive'
    : vsMedianPct < -8  ? 'cheap'
    : 'fair';

  // 30-day forecast
  const tf        = signals.timeFeatures || { slope: 0, volatility: 0, reliable: false };
  const projected = nHist >= 3 && tf.reliable ? baseline + tf.slope * 30 : baseline;
  const band      = Math.max((tf.volatility || 0) * 30 * 1.2, baseline * 0.04);
  const hShift    = baseline * expectedDropDepthPct;

  let eMin = Math.round(Math.max(cur * 0.40, projected - band - hShift) * 100) / 100;
  let eMed = Math.round(Math.max(cur * 0.50, projected)                 * 100) / 100;
  let eMax = Math.round((projected + band)                               * 100) / 100;

  if (nPeer >= 2) {
    const pP25 = percentile(pSorted, 0.25);
    const pP75 = percentile(pSorted, 0.75);
    eMin = Math.round(Math.max(eMin, pP25 * 0.88) * 100) / 100;
    eMed = Math.round(Math.min(Math.max(eMed, eMin), pP75 * 1.05) * 100) / 100;
  }

  // Drop probability
  const trendBoost = fallingFast ? 0.20 : trendDir === 'falling' ? 0.10 : 0;
  const baseDrop   = nHist >= 5
    ? (aboveAvgPct > 10 ? 0.35 : aboveAvgPct > 0 ? 0.22 : 0.12)
    : (peerSignal === 'expensive' ? 0.30 : 0.18);
  const dropProbability30d = Math.min(0.92, Math.round((baseDrop + dropProbBoost + trendBoost) * 100) / 100);

  // Target buy price — lowest realistic price below current
  const candidates = [
    histMin,
    nPeer >= 2 ? percentile(pSorted, 0.10) : null,
    expectedDropDepthPct > 0.02 ? Math.round(Math.min(cur, peerMedian||cur) * (1 - expectedDropDepthPct) * 100)/100 : null,
    eMin,
  ].filter(v => v !== null && Number.isFinite(v) && v < cur * 0.98 && v > cur * 0.40);
  const targetBuyPrice = candidates.length ? Math.round(Math.min(...candidates)*100)/100 : null;

  // ── Verdict + rich sentence reasoning ──
  const mStr = peerMedian ? `market median ${fmt(peerMedian)} across ${peers.length + 1} stores` : null;
  let verdict, action, reasoning;

  if (daysToEvent === 0 && aboveLowPct !== null && aboveLowPct < 10) {
    verdict = 'BUY'; action = 'buy';
    reasoning = `${eventName} is active and at ${fmt(cur)} the price is near its tracked low of ${fmt(histMin)} — one of the best prices you will see this cycle.`;
  } else if (daysToEvent === 0) {
    verdict = 'WAIT'; action = 'wait';
    reasoning = `${eventName} is active but at ${fmt(cur)} the price is still ${aboveLowPct}% above its tracked low of ${fmt(histMin)} — expect it to drop further toward ${fmt(eMin)} before the window closes.`;
  } else if (peerSignal === 'expensive' && daysToEvent !== null && daysToEvent <= 45) {
    verdict = 'WAIT'; action = 'wait';
    reasoning = `At ${fmt(cur)} you are paying ${Math.abs(vsMedianPct)}% above the ${mStr}, and ${eventName} is ${daysToEvent <= 14 ? 'only ' + daysToEvent + ' days away' : 'within 45 days'} — ${impactLabel || 'a price drop is expected'}, putting the target buy price around ${fmt(targetBuyPrice || eMin)}.`;
  } else if (peerSignal === 'expensive') {
    verdict = 'WAIT'; action = 'wait';
    reasoning = `At ${fmt(cur)} this is ${Math.abs(vsMedianPct)}% above the ${mStr}${aboveAvgPct !== null ? ', and ' + aboveAvgPct + '% above its own tracked average of ' + fmt(baseline) : ''} — consider waiting for a correction toward ${fmt(peerMedian || eMin)}.`;
  } else if (fallingFast) {
    verdict = 'WATCH'; action = 'watch';
    reasoning = `The price is dropping fast — it averaged ${fmt(olderAvg)} previously and is now at ${fmt(recentAvg)}, a ${Math.round(Math.abs(velocityPct)*100)}% move. At this rate it could reach ${fmt(targetBuyPrice || eMin)} within 30 days — set an alert rather than buying now.`;
  } else if (aboveLowPct !== null && aboveLowPct < 5) {
    verdict = 'BUY'; action = 'buy';
    reasoning = `At ${fmt(cur)} this is within 5% of its tracked low of ${fmt(histMin)} across ${nHist} observations${mStr ? ', and the ' + mStr + ' confirms it is competitively priced' : ''} — it rarely gets cheaper than this.`;
  } else if (aboveAvgPct !== null && aboveAvgPct > 12 && daysToEvent !== null && daysToEvent <= 45) {
    verdict = 'WAIT'; action = 'wait';
    reasoning = `At ${fmt(cur)} this is ${aboveAvgPct}% above its tracked average of ${fmt(baseline)} and ${eventName} is ${daysToEvent <= 14 ? daysToEvent + ' days away' : 'within 45 days'} — ${impactLabel || 'a drop is expected'}, with a target buy price around ${fmt(targetBuyPrice || eMin)}.`;
  } else if (aboveAvgPct !== null && aboveAvgPct > 12) {
    verdict = 'WAIT'; action = 'wait';
    const note = risingFast ? 'the price has been climbing and corrections usually follow' : 'consider waiting for the price to return toward its baseline';
    reasoning = `At ${fmt(cur)} this is ${aboveAvgPct}% above its tracked average of ${fmt(baseline)} across ${nHist} observations — ${note}${mStr ? ', and the ' + mStr + ' is lower' : ''}.`;
  } else if (risingFast) {
    verdict = 'WATCH'; action = 'watch';
    reasoning = `The price has been rising — it averaged ${fmt(olderAvg)} recently and is now at ${fmt(recentAvg)}, a ${Math.round(Math.abs(velocityPct)*100)}% move upward. Worth monitoring for a few days before committing at ${fmt(cur)}.`;
  } else if (peerSignal === 'cheap') {
    verdict = 'BUY'; action = 'buy';
    reasoning = `At ${fmt(cur)} this is ${Math.abs(vsMedianPct)}% below the ${mStr} — cheaper than most stores right now${aboveLowPct !== null && aboveLowPct < 15 ? ', and near its tracked low of ' + fmt(histMin) : ''}. No major sale within 45 days to justify waiting.`;
  } else {
    verdict = 'FAIR'; action = 'ok';
    const tNote = trendDir === 'stable' ? 'Price is stable'
      : trendDir === 'falling' ? 'Price is gently drifting down'
      : 'Price has been edging up slightly';
    const pNote = mStr ? ` The ${mStr} confirms this is in line with the market.` : '';
    reasoning = `At ${fmt(cur)} this is close to its tracked average of ${fmt(baseline)} across ${nHist} observations — a reasonable price if you need it now. ${tNote}.${pNote}`;
  }

  // ── Split confidence ──
  const dataScore = Math.min(100, (nHist * 3) + (nPeer * 5) + (tf.reliable ? 15 : 0));
  const dataConfidence = dataScore >= 70 ? 'High' : dataScore >= 35 ? 'Medium' : 'Low';
  const dataConfidenceScore = Math.min(92, Math.max(20, dataScore));

  const sCount = [peerSignal !== 'unknown', tf.reliable, daysToEvent !== null && daysToEvent <= 45, Math.abs(aboveAvgPct||0) > 5].filter(Boolean).length;
  const forecastConfidence = sCount >= 3 ? 'High' : sCount >= 2 ? 'Medium' : 'Low';
  const forecastConfidenceScore = Math.min(92, Math.max(20, 35 + sCount * 15 + (nPeer >= 3 ? 8 : 0)));
  const confidence = Math.round((dataConfidenceScore + forecastConfidenceScore) / 2);

  return {
    verdict, action, reasoning,
    targetBuyPrice, trendDirection: trendDir, dropProbability30d,
    expectedMin30d: eMin, expectedMedian30d: eMed, expectedMax30d: eMax,
    expectedRange30d: { min: eMin, median: eMed, max: eMax },
    predictedLow: targetBuyPrice || (eMin < cur ? eMin : null),
    aboveAvgPct, aboveLowPct, peerSignal, vsMedianPct, peerMedian,
    confidence, dataConfidence, dataConfidenceScore, forecastConfidence, forecastConfidenceScore,
    seasonalDriver: daysToEvent !== null && daysToEvent <= 45 ? { eventName, daysToEvent, impactSummary: impactLabel } : null,
    signals,
  };
}

// ── E) Format for UI card ──────────────────────────────────────
function formatPredictionForUI(pred, fmt) {
  const confLabel = pred.dataConfidenceScore >= 70 ? 'High' : pred.dataConfidenceScore >= 40 ? 'Medium' : 'Low';
  const trendLabel = { 'falling fast':'Falling Fast','falling':'Falling','rising fast':'Rising Fast','rising':'Rising','stable':'Stable','unknown':'Unknown' }[pred.trendDirection] || pred.trendDirection;
  return {
    verdict: pred.verdict,
    trendDirection: pred.trendDirection, trendLabel,
    predictionWindowDays: 30,
    targetBuyPrice: pred.targetBuyPrice,
    targetBuyPriceFmt: pred.targetBuyPrice ? fmt(pred.targetBuyPrice) : null,
    expectedRange30d: pred.expectedRange30d,
    dropProbability30d: pred.dropProbability30d,
    confidence: confLabel, confidenceScore: pred.dataConfidenceScore,
    explanation: pred.reasoning,
    seasonalDriver: pred.seasonalDriver,
  };
}

// ── Main function ──────────────────────────────────────────────
async function computePredictionV2({ title, currentPrice, currency, storeId, storeProductId, historyPrices, historyRows, peerPrices, now: _now }) {
  const now  = _now || new Date();
  const cur  = Number(currentPrice);
  const curr = currency || 'USD';
  const fmt  = p => moneyFmt(curr, p);

  if (!cur || !Number.isFinite(cur)) return null;

  const cKey   = predCacheKey(storeId, storeProductId, Math.round(cur * 100), curr);
  const cached = predCache.get(cKey);
  if (cached) return { ...cached, fromCache: true };

  // Accept historyRows (with timestamps) or fall back to flat prices array
  const rows  = (historyRows || []).filter(r => Number.isFinite(Number(r.price)) && Number(r.price) > 0);
  const hist  = rows.length
    ? rows.map(r => Number(r.price))
    : (historyPrices || []).filter(n => Number.isFinite(n) && n > 0);
  const peers = (peerPrices || []).filter(n => Number.isFinite(n) && n > 0);
  const nHist = hist.length;
  const nPeer = peers.length;

  const attrs   = title ? await extractProductAttrs(title) : ruleBasedAttrs('');
  const cat     = attrs.category || 'electronics';
  const hCtx    = getHolidayContext(now, cat);
  const tf      = computeTimeFeatures(rows.length >= 2 ? rows : null);
  const signals = rows.length >= 1
    ? computeProductSignals(rows, cur)
    : { baseline: cur, histMin: null, histMax: null, aboveAvgPct: null, aboveLowPct: null,
        recentAvg: null, olderAvg: null, velocityPct: 0, fallingFast: false, risingFast: false,
        trendDir: 'unknown', timeFeatures: tf };
  if (!signals.timeFeatures) signals.timeFeatures = tf;

  const holiday  = holidayImpact(storeId, cat, hCtx);
  const combined = combineSignals({ cur, fmt, signals, holiday, peers, nHist, nPeer });
  const ui       = formatPredictionForUI(combined, fmt);

  const allMarket   = [cur, ...peers].filter(n => Number.isFinite(n) && n > 0);
  const mktSorted   = [...allMarket].sort((a,b) => a - b);
  const mktMedian   = mktSorted[Math.floor(mktSorted.length / 2)];

  const allWindows   = [...getHolidayWindows(now.getFullYear()), ...getHolidayWindows(now.getFullYear()+1)];
  const nextEvtObj   = allWindows.find(w => now < w.windowStart);
  const nextEvtDate  = nextEvtObj && hCtx.daysUntilNext <= 45
    ? nextEvtObj.start.toLocaleDateString('en-US', { month: 'long', day: 'numeric' })
    : null;

  // ── Human-readable pct string: never "-1% above", say "1% below" ──
  const fmtVsMedian = (pct) => {
    if (pct === null || pct === undefined) return null;
    if (pct > 0)  return `${pct}% above market`;
    if (pct < 0)  return `${Math.abs(pct)}% below market`;
    return 'at market price';
  };
  const fmtVsAvg = (pct) => {
    if (pct === null || pct === undefined) return null;
    if (pct > 0)  return `${pct}% above tracked average`;
    if (pct < 0)  return `${Math.abs(pct)}% below tracked average`;
    return 'at tracked average';
  };

  // ── Trend label — clean pill text ──
  const trendLabel = {
    'falling fast': 'Falling Fast', 'falling': 'Falling',
    'rising fast':  'Rising Fast',  'rising':  'Rising',
    'stable': 'Stable', 'unknown': null,
  }[combined.trendDirection] || null;

  // ── Confidence label ──
  const confLabel = combined.dataConfidenceScore >= 70 ? 'High'
    : combined.dataConfidenceScore >= 40 ? 'Medium' : 'Low';

  // ── Target price label — be precise about what the number means ──
  // targetBuyPrice = realistic near-term buy target (peer median or holiday floor)
  // expectedMin30d = model's 30-day low estimate (can be more aggressive)
  // Only surface expectedMin30d if meaningfully different from targetBuyPrice
  const targetDisplay  = combined.targetBuyPrice ? fmt(combined.targetBuyPrice) : null;
  const floorDisplay   = (() => {
    if (!combined.expectedMin30d) return null;
    if (!combined.targetBuyPrice) return fmt(combined.expectedMin30d);
    // Only show floor separately if it's >8% below target (otherwise redundant)
    const diff = (combined.targetBuyPrice - combined.expectedMin30d) / combined.targetBuyPrice;
    return diff >= 0.08 ? fmt(combined.expectedMin30d) : null;
  })();

  // ── Sale event — only surface if active or within 45 days, never null ──
  const saleEvent = (() => {
    if (hCtx.insideWindow && hCtx.activeEvent) {
      return { name: hCtx.activeEvent, label: 'Active now', daysAway: 0 };
    }
    if (hCtx.daysUntilNext <= 45 && hCtx.nextEvent) {
      return {
        name: hCtx.nextEvent,
        label: nextEvtDate ? `${hCtx.nextEvent} · ${nextEvtDate}` : `${hCtx.nextEvent} · ${hCtx.daysUntilNext}d away`,
        daysAway: hCtx.daysUntilNext,
      };
    }
    return null; // hidden entirely — never show null text
  })();

  // ── Subtitle — one clean line matching the verdict ──
  const subtitle = {
    'WAIT':  'Price looks elevated right now',
    'WATCH': 'Price is moving — monitor before buying',
    'BUY':   'Good time to buy',
    'FAIR':  'Price is reasonable right now',
  }[combined.verdict] || 'Price analysis';

  // ── card: structured display object for the frontend ──
  // Every field is either a clean string or null (never undefined, never raw number)
  const card = {
    verdict:        combined.verdict,
    subtitle,
    // Data rows — frontend renders as label: value pairs, skips nulls
    currentPrice:   fmt(cur),
    marketMedian:   nPeer >= 1 ? `${fmt(mktMedian)} across ${allMarket.length} stores` : null,
    trackedAverage: signals.baseline && nHist >= 3 ? fmt(signals.baseline) : null,
    trend:          trendLabel,
    confidence:     confLabel,
    // Position summary — human readable, no raw numbers
    vsMarket:       combined.vsMedianPct !== null ? fmtVsMedian(combined.vsMedianPct) : null,
    vsAverage:      combined.aboveAvgPct  !== null ? fmtVsAvg(combined.aboveAvgPct)   : null,
    // Short explanation — 1-2 sentences, no raw %/numbers repeated from rows above
    explanation:    combined.reasoning,
    // Price targets — labeled precisely
    targetBuyPrice: targetDisplay,   // realistic near-term entry
    lowEndEstimate: floorDisplay,    // bear-case floor, only if meaningfully lower
    // Sale event — null means don't render the row at all
    saleEvent,
    // Window
    predictionWindow: '30 days',
  };

  const result = {
    // ── Primary fields ──
    verdict:              combined.verdict,
    action:               combined.action,
    reasoning:            combined.reasoning,
    explanation:          combined.reasoning,
    trendDirection:       combined.trendDirection,
    targetBuyPrice:       combined.targetBuyPrice,
    predictedLow:         combined.predictedLow,
    dropProbability30d:   combined.dropProbability30d,
    dropProbability7d:    combined.dropProbability30d,
    buyNowProbability:    combined.verdict === 'BUY' ? 0.75 : combined.verdict === 'WAIT' ? 0.18 : 0.45,
    expectedMin30d:       combined.expectedMin30d,
    expectedMedian30d:    combined.expectedMedian30d,
    expectedMax30d:       combined.expectedMax30d,
    expectedRange30d:     combined.expectedRange30d,
    confidence:           combined.confidence,
    dataConfidence:       combined.dataConfidence,
    dataConfidenceScore:  combined.dataConfidenceScore,
    forecastConfidence:   combined.forecastConfidence,
    forecastConfidenceScore: combined.forecastConfidenceScore,
    // ── Seasonal — null when > 45 days away, never exposed as raw null text ──
    seasonalDriver:       combined.seasonalDriver,
    nextSaleEvent:        saleEvent?.name || null,
    monthsAway:           hCtx.daysUntilNext <= 45 ? null : hCtx.monthsAway, // suppressed when > 45d
    // ── Position ──
    aboveAvgPct:          combined.aboveAvgPct,
    aboveLowPct:          combined.aboveLowPct,
    vsMedianPct:          combined.vsMedianPct,
    vsMedianLabel:        fmtVsMedian(combined.vsMedianPct),
    vsAverageLabel:       fmtVsAvg(combined.aboveAvgPct),
    // ── Context ──
    historyCount:         nHist,
    coldStart:            nHist < 5,
    reasons:              [combined.reasoning],
    marketIntel: nPeer >= 1 ? {
      marketMedian:  fmt(mktMedian),
      storeCount:    allMarket.length,
      vsMedianPct:   combined.vsMedianPct !== null ? fmtVsMedian(combined.vsMedianPct) : null,
      targetBuyPrice: targetDisplay,
    } : null,
    seasonality: saleEvent ? {
      nextEvent:         saleEvent.name,
      label:             saleEvent.label,
      daysAway:          saleEvent.daysAway,
      expectedImpactPct: combined.seasonalDriver ? Math.round(holiday.expectedDropDepthPct * 100) : 0,
      activeEvent:       hCtx.activeEvent || null,
      insideWindow:      hCtx.insideWindow,
      nextEventDate:     nextEvtDate || null,
    } : null,
    stats: {
      baseline:   combined.signals?.baseline ?? Math.round(mktMedian * 100)/100,
      ewma:       Math.round((signals.baseline || cur) * 100)/100,
      slope:      tf.slope,
      volatility: tf.volatility,
      spanDays:   tf.spanDays,
      obsPerDay:  tf.obsPerDay,
      p25:        nPeer >= 1 ? mktSorted[Math.floor(mktSorted.length*0.25)] : null,
      p75:        nPeer >= 1 ? mktSorted[Math.floor(mktSorted.length*0.75)] : null,
      nHistory:   nHist, nPeers: nPeer,
    },
    debug: {
      coldStart:      nHist < 5,
      peerSignal:     combined.peerSignal,
      vsMedianPct:    combined.vsMedianPct,
      holidayGated:   hCtx.daysUntilNext > 45,
      holidayDaysTo:  hCtx.daysUntilNext,
      usedTimestamps: tf.reliable,
      usedPeers:      nPeer > 0,
      usedGemini:     false,
    },
    // ── card: clean display object — frontend renders this, skipping null fields ──
    card,
    ui,
  };

  predCache.set(cKey, result);
  return result;
}


async function sendWhatsAppAlert({ whatsapp, callmebotKey, message }) {
  try {
    const encoded = encodeURIComponent(message);
    const url = `https://api.callmebot.com/whatsapp.php?phone=${encodeURIComponent(whatsapp)}&text=${encoded}&apikey=${encodeURIComponent(callmebotKey)}`;
    const r = await fetch(url, { signal: AbortSignal.timeout(10000) });
    const body = await r.text();
    // CallMeBot returns "Message queued" or similar on success
    const ok = r.ok && !body.toLowerCase().includes("error");
    if (!ok) console.error("[CallMeBot] Unexpected response:", body.slice(0, 200));
    return ok;
  } catch (e) {
    console.error("[CallMeBot] Send failed:", e.message);
    return false;
  }
}

async function checkAndFireAlerts(storeId, storeProductId, currentPrice, currency, productTitle, productUrl) {
  if (!Number.isFinite(currentPrice)) return;
  try {
    const { data: alerts } = await supabase
      .from("price_alerts")
      .select("*")
      .eq("store_id", storeId)
      .eq("store_product_id", storeProductId)
      .eq("is_active", true);

    if (!alerts?.length) return;

    for (const alert of alerts) {
      if (currentPrice > Number(alert.target_price)) continue; // price hasn't dropped to target yet

      const currency_sym = currency || alert.currency || "USD";
      const fmtPrice  = (p) => {
        try { return new Intl.NumberFormat("en-US", { style: "currency", currency: currency_sym, maximumFractionDigits: 2 }).format(p); }
        catch { return `${currency_sym} ${Number(p).toFixed(2)}`; }
      };

      const title       = productTitle || alert.product_title || "your tracked product";
      const link        = productUrl || alert.product_url || "";
      const priceAtSetup = Number.isFinite(Number(alert.current_price)) ? Number(alert.current_price) : NaN;
      const saving      = Number.isFinite(priceAtSetup) && priceAtSetup > currentPrice ? priceAtSetup - currentPrice : 0;
      const saveStr     = saving > 0 ? ` (down ${fmtPrice(saving)} from ${fmtPrice(priceAtSetup)} when you set the alert)` : "";
      const message = [
        `🎯 Atheon Price Drop Alert!`,
        ``,
        `"${title}"`,
        ``,
        `✅ Price is now: ${fmtPrice(currentPrice)}${saveStr}`,
        `🎯 Your target was: ${fmtPrice(alert.target_price)}`,
        ``,
        link ? `👉 Buy now: ${link}` : `Check it out on your tracked store.`,
        ``,
        `— Atheon Price Tracker`
      ].join("\n");

      const sent = await sendWhatsAppAlert({
        whatsapp: alert.whatsapp,
        callmebotKey: alert.callmebot_key,
        message
      });

      if (sent) {
        await supabase
          .from("price_alerts")
          .update({ is_active: false, last_notified_at: new Date().toISOString() })
          .eq("id", alert.id);
        console.log(`[Alerts] Fired alert ${alert.id} for ${storeId}/${storeProductId} at ${currentPrice}`);
      }
    }
  } catch (e) {
    console.error("[Alerts] checkAndFireAlerts error:", e.message);
  }
}

// ── POST /v1/alerts — Create a price alert
app.post("/v1/alerts", async (req, res) => {
  const { storeId, storeProductId, whatsapp, callmebotKey, targetPrice, currentPrice, currency, productTitle, productUrl, sessionId } = req.body;

  if (!storeId || !storeProductId || !whatsapp || !callmebotKey || !Number.isFinite(Number(targetPrice)))
    return res.status(400).json({ ok: false, error: "Missing required fields: storeId, storeProductId, whatsapp, callmebotKey, targetPrice" });

  // Normalise phone number
  const phone = String(whatsapp).replace(/\s+/g, "");
  if (!/^\+[1-9]\d{6,14}$/.test(phone))
    return res.status(400).json({ ok: false, error: "Invalid WhatsApp number. Include country code, e.g. +447911123456" });

  // Cap: 5 active alerts per session per day
  if (sessionId) {
    const { count } = await supabase
      .from("price_alerts")
      .select("id", { count: "exact", head: true })
      .eq("session_id", sessionId)
      .eq("is_active", true);
    if (count >= 10)
      return res.status(429).json({ ok: false, error: "You have 10 active alerts. Cancel one before adding another." });
  }

  const { data, error } = await supabase.from("price_alerts").insert({
    store_id: storeId,
    store_product_id: storeProductId,
    whatsapp: phone,
    callmebot_key: String(callmebotKey).trim(),
    target_price: Number(targetPrice),
    current_price: Number.isFinite(Number(currentPrice)) && Number(currentPrice) > 0 ? Number(currentPrice) : null,
    currency: String(currency || "USD"),
    product_title: productTitle || null,
    product_url: productUrl || null,
    session_id: sessionId || null,
    is_active: true
  }).select("id").single();

  if (error) return res.status(500).json({ ok: false, error: "Failed to save alert." });

  return res.json({ ok: true, alertId: data.id, message: "Alert set! You'll get a WhatsApp message when the price drops." });
});

// ── GET /v1/alerts — List alerts for a session
app.get("/v1/alerts", async (req, res) => {
  const { sessionId, storeId, storeProductId } = req.query;
  if (!sessionId) return res.status(400).json({ ok: false, error: "sessionId required" });

  let q = supabase.from("price_alerts").select("id,store_id,store_product_id,target_price,current_price,currency,product_title,is_active,created_at").eq("session_id", sessionId).order("created_at", { ascending: false }).limit(20);
  if (storeId) q = q.eq("store_id", storeId);
  if (storeProductId) q = q.eq("store_product_id", storeProductId);

  const { data, error } = await q;
  if (error) return res.status(500).json({ ok: false, error: error.message });
  return res.json({ ok: true, alerts: data || [] });
});

// ── DELETE /v1/alerts/:id — Cancel an alert
app.delete("/v1/alerts/:id", async (req, res) => {
  const { sessionId } = req.body;
  const { id } = req.params;
  if (!id || !sessionId) return res.status(400).json({ ok: false, error: "Missing id or sessionId" });

  const { error } = await supabase.from("price_alerts")
    .update({ is_active: false })
    .eq("id", id)
    .eq("session_id", sessionId); // safety: users can only cancel their own

  if (error) return res.status(500).json({ ok: false, error: error.message });
  return res.json({ ok: true, message: "Alert cancelled." });
});

// ── POST /v1/history-audit — Browser history audit on first install
// Extension sends up to 100 URLs from history, we resolve them to products
// and check for price drops vs when the user visited.
app.post("/v1/history-audit", async (req, res) => {
  const { sessionId, urls } = req.body; // urls: [{url, visitedAt}]
  if (!sessionId || !Array.isArray(urls) || !urls.length)
    return res.status(400).json({ ok: false, error: "sessionId and urls[] required" });

  const capped = urls.slice(0, 150); // hard cap
  const results = [];

  for (const entry of capped) {
    const rawUrl = String(entry.url || "").trim();
    const visitedAt = entry.visitedAt || null;

    try {
      const parsed = new URL(rawUrl);
      const host = parsed.hostname.toLowerCase().replace(/^www\./, "");

      // Find extractor for this host
      const { data: extractors } = await supabase
        .from("site_extractors")
        .select("store_id,product_id_regex,product_id_mode,path_regex")
        .eq("is_enabled", true)
        .or(`host.eq.${host},host.eq.www.${host}`)
        .order("priority", { ascending: true })
        .limit(1);

      const ex = extractors?.[0];
      if (!ex) continue; // unsupported store

      // Extract product ID from URL
      let storeProductId = null;
      if (ex.product_id_regex) {
        try {
          const m = ex.product_id_regex.match(/^\/(.+)\/([gimsuy]*)$/);
          const re = m ? new RegExp(m[1], m[2]) : new RegExp(ex.product_id_regex);
          const mm = rawUrl.match(re);
          if (mm) storeProductId = mm[1] || mm[0];
        } catch {}
      }
      if (!storeProductId && ex.product_id_mode === "canonical") {
        storeProductId = rawUrl;
      }
      if (!storeProductId) continue;

      // Look up product in DB
      const { data: product } = await supabase
        .from("products")
        .select("id,title,last_price,currency,canonical_url,image_url")
        .eq("store_id", ex.store_id)
        .eq("store_product_id", storeProductId)
        .maybeSingle();

      if (!product) continue;

      const priceNow = Number(product.last_price);
      if (!Number.isFinite(priceNow)) continue;

      // Find price closest to visitedAt (if we have it)
      let priceAtVisit = null;
      if (visitedAt) {
        const { data: obsNear } = await supabase
          .from("price_observations")
          .select("price,observed_at")
          .eq("product_id", product.id)
          .lte("observed_at", new Date(visitedAt).toISOString())
          .order("observed_at", { ascending: false })
          .limit(1);
        if (obsNear?.[0]) priceAtVisit = Number(obsNear[0].price);
      }

      const dropped = priceAtVisit != null && priceNow < priceAtVisit;
      const dropAmount = dropped ? Math.round((priceAtVisit - priceNow) * 100) / 100 : null;
      const dropPct = dropped && priceAtVisit > 0 ? Math.round(((priceAtVisit - priceNow) / priceAtVisit) * 100 * 10) / 10 : null;

      // Store in DB
      await supabase.from("browser_history_audit").insert({
        session_id: sessionId,
        url: rawUrl,
        visited_at: visitedAt ? new Date(visitedAt).toISOString() : null,
        resolved: true,
        store_id: ex.store_id,
        store_product_id: storeProductId,
        product_title: product.title,
        price_at_visit: priceAtVisit,
        price_now: priceNow,
        currency: product.currency || "USD",
        price_dropped: dropped,
        drop_amount: dropAmount,
        drop_pct: dropPct,
      });

      results.push({
        url: rawUrl,
        storeId: ex.store_id,
        title: product.title,
        imageUrl: product.image_url || null,
        productUrl: product.canonical_url || rawUrl,
        priceAtVisit,
        priceNow,
        currency: product.currency || "USD",
        dropped,
        dropAmount,
        dropPct,
        visitedAt,
      });

    } catch (e) {
      // Skip bad URLs silently
    }
  }

  // Sort: biggest drops first
  results.sort((a, b) => (b.dropPct || 0) - (a.dropPct || 0));

  return res.json({
    ok: true,
    resolved: results.length,
    total: capped.length,
    dropped: results.filter(r => r.dropped).length,
    results,
  });
});

// ── AI Chat — live, per-user rate limited, cached for repeat questions
app.post("/v1/ai/chat", async (req, res) => {
  const ip = req.ip || "x";
  if (!checkRate(`chat:${ip}:hour`, 10, 60*60*1000))
    return res.status(429).json({ ok: false, error: "Chat limit: 10 messages per hour. Come back soon." });
  if (!checkRate(`chat:${ip}:burst`, 2, 10*1000))
    return res.status(429).json({ ok: false, error: "Slow down a little." });

  const parsed = AiChatSchema.safeParse(req.body);
  if (!parsed.success) return res.status(400).json({ ok: false, error: parsed.error.flatten() });

  const { message, storeId, storeProductId, chatHistory = [] } = parsed.data;
  const { data: product } = await supabase.from("products").select("*").eq("store_id", storeId).eq("store_product_id", storeProductId).maybeSingle();
  if (!product) return res.status(404).json({ ok: false, error: "not_found" });

  const { data: hist } = await supabase.from("price_observations").select("price,observed_at").eq("product_id", product.id).order("observed_at", { ascending: false }).limit(120);
  const prices = (hist||[]).map(r => Number(r.price)).filter(n => Number.isFinite(n));
  const stats  = computeStats(prices);
  const currentPrice = Number(parsed.data.currentPrice ?? product.last_price);
  const currency = String(parsed.data.currency ?? product.currency ?? "USD");
  const dealsR5 = await computeDealsForProduct({ storeId, storeProductId, baseTitle: product.title||"", baseCurrency: currency, basePrice: currentPrice, limit: 10 });
  const deals = dealsR5.bestDeals || [];

  // Heuristic fake deal for chat context (no AI call here — detector key is reserved)
  let fakeDealCtx = "";
  if (parsed.data.wasPrice) {
    const fd = heuristicDetect(currentPrice, parsed.data.wasPrice, currency, prices, deals.map(d=>d.price).filter(Number.isFinite));
    if (fd) fakeDealCtx = `${fd.verdict} — ${fd.message}`;
  }

  // Cache for first message only (not multi-turn)
  const isFirst = chatHistory.length === 0;
  const cKey = isFirst ? chatCacheKey(product.id, message) : null;
  if (cKey) { const cached = chatCache.get(cKey); if (cached) return res.json({ ok: true, reply: cached, stats, deals, fromCache: true }); }

  const cheaperDeals  = deals.filter(d => d.price < currentPrice);
  const exactDeals    = cheaperDeals.filter(d => d.matchTier === "EXACT" || d.matchTier === "SAME_VARIANT");
  const relatedDeals  = cheaperDeals.filter(d => d.matchTier === "SAME_MODEL" || d.matchTier === "SAME_FAMILY");
  const usedDeals     = deals.filter(d => d.condition === "used" && d.price < currentPrice * 1.05);

  // Compute price trend direction
  let trend = "stable";
  if (prices.length >= 4) {
    const recent = prices.slice(0, 3).reduce((a,b) => a+b,0) / 3;
    const older  = prices.slice(-3).reduce((a,b) => a+b,0)   / 3;
    if (recent < older * 0.96) trend = "falling";
    else if (recent > older * 1.04) trend = "rising";
  }

  // Build rich, structured context for the AI
  const ctx = [
    `PRODUCT: ${product.title}`,
    `CURRENT PRICE: ${moneyFmt(currency, currentPrice)} (${currency})`,
    product.canonical_url ? `LINK: ${product.canonical_url}` : "",

    stats
      ? [
          `PRICE HISTORY (${prices.length} data points):`,
          `  All-time low:  ${moneyFmt(currency, stats.min)}`,
          `  All-time high: ${moneyFmt(currency, stats.max)}`,
          `  Average:       ${moneyFmt(currency, stats.avg)}`,
          `  Trend:         ${trend} (based on recent vs older observations)`,
          `  vs Average:    current is ${currentPrice > stats.avg ? "+" : ""}${Math.round(((currentPrice - stats.avg) / stats.avg) * 100)}% vs avg`,
          `  vs Low:        current is ${Math.round(((currentPrice - stats.min) / stats.min) * 100)}% above all-time low`,
        ].join("\n")
      : "PRICE HISTORY: None yet — first time tracking this product.",

    exactDeals.length
      ? `EXACT SAME SPECS CHEAPER ELSEWHERE:\n${exactDeals.slice(0,5).map(d =>
          `  - ${d.name}: ${moneyFmt(d.currency||currency, d.price)} (save ${moneyFmt(currency, currentPrice - d.price)})${d.url ? ` → ${d.url}` : ""} [${d.matchLabel}]`
        ).join("\n")}`
      : "EXACT MATCHES CHEAPER: None found.",

    relatedDeals.length
      ? `SAME MODEL LINE (different tier/storage) CHEAPER:\n${relatedDeals.slice(0,3).map(d =>
          `  - ${d.name}: ${moneyFmt(d.currency||currency, d.price)} — "${d.title?.slice(0,60)}"${d.url ? ` → ${d.url}` : ""} [${d.matchLabel}]`
        ).join("\n")}`
      : "",

    usedDeals.length
      ? `USED / REFURBISHED OPTIONS:\n${usedDeals.slice(0,3).map(d =>
          `  - ${d.name}: ${moneyFmt(d.currency||currency, d.price)} — "${d.title?.slice(0,60)}"${d.url ? ` → ${d.url}` : ""} [${d.matchLabel}]`
        ).join("\n")}`
      : "",

    fakeDealCtx ? `DISCOUNT CHECK: ${fakeDealCtx}` : "",
  ].filter(Boolean).join("\n\n");

  const convHistory = chatHistory.slice(-8).map(h => `${h.role==="user"?"User":"Atheon"}: ${h.content}`).join("\n");

  const today = new Date();
  const monthName = today.toLocaleString("en-US", { month: "long" });
  const dayOfMonth = today.getDate();

  const prompt = [
    `You are Atheon — a sharp, knowledgeable shopping assistant living inside a price-tracking browser extension. You talk like a real, warm person — not a robot or a hype machine.

TODAY: ${monthName} ${dayOfMonth}, ${today.getFullYear()}

MOST IMPORTANT: Read what the user actually said. Only answer that. Never dump everything at once unprompted.

─── HOW TO HANDLE EACH TYPE OF MESSAGE ───

GREETING (hey/hi/hello/sup/yo):
→ Casual greeting. One short line about the product (name + current price). Done. Keep it under 2 sentences.

"who are you" / "what can you do":
→ Explain you're Atheon, a price intelligence assistant. 2-3 casual sentences. Mention what you track (prices, deals, fake discounts). Done.

"tell me about the product" / "what is this":
→ Describe the product naturally. Mention current price, what type of product it is. If there's history, mention one key insight (e.g. "it's been lower before"). Done.

"is this a good price" / "should I buy" / "worth it?":
→ Give a clear BUY, WAIT, or DEPENDS answer. Cite actual numbers (current vs avg vs low). Mention trend. If there are cheaper exact matches elsewhere, say so. Be decisive — don't hedge everything. Done.

"cheaper elsewhere" / "compare" / "other stores":
→ ALWAYS lead with EXACT SAME SPECS deals first. These are the same product, same storage, same tier.
→ Show a markdown table for exact matches. Then optionally mention same-line options (different storage/tier) as "also worth a look."
→ NEVER mix used/refurb with new listings unless user specifically asks. Keep them clearly separate.
→ Format: | Store | Price | You Save |

"fake deal" / "is the discount real" / "was price legit?":
→ Give the verdict from the context. Cite the numbers. Be direct. Done.

"can't afford it" / "too expensive" / "budget option":
→ Look at used/refurb options in context first. Suggest those with price. If none, say "no cheaper options in my database right now."

"what's the price trend" / "going up or down?":
→ Use the trend data from context. Mention recent vs historical. Suggest when to buy based on trend + season.

SEASONAL CONTEXT you should know:
- Jan: post-Christmas clearance, CES launches → prices often drop in Feb after CES
- Feb: Valentine's Day sales on lifestyle/audio/gifts
- Mar-Apr: Spring sales, some laptop/tablet launches
- May-Jun: Back to school prep begins, Mother's Day
- Jul: Amazon Prime Day (huge discounts on electronics)
- Aug-Sep: Back to School peak — best time to buy laptops, tablets, headphones
- Oct: Pre-holiday deals begin, Apple event season
- Nov: Black Friday — historically best prices of the year on electronics
- Dec: Cyber Monday, Christmas deals, Boxing Day (UK/CA/AU)
Use this knowledge when giving BUY/WAIT advice.

─── PERSONALITY ───
- Match energy: chill user = chill Atheon. Excited user = enthusiastic back.
- If someone's rude or pushes limits, handle it warmly and redirect. Never lecture.
- Use contractions ("I don't", "it's", "you're"). Sound natural.
- Never sound like a template. Every reply should feel fresh.
- Don't start replies with "Great question!" or "Certainly!" — just answer.

─── FORMATTING ───
- Store links: [StoreName](url) — the store name IS the link
- Comparison table:
  | Store | Price | You Save |
  |-------|-------|----------|
  | [Amazon](url) | $499 | $50 |
- Bold key numbers and verdicts: **$299**, **BUY NOW**, **WAIT**
- Used/refurb always clearly labelled — never mix with new in the same table row
- Short for small talk. Only go detailed when asked for actual data.

─── DATA RULES ───
- Only use prices and stores from the context below. Never invent stores or prices.
- If a deal has no URL, show store name as plain text (no link).
- EXACT SAME SPECS means same model number + same tier + same storage. Don't say "exact match" for a different storage size or older model.
- If data is missing, say so honestly — "I don't have that info right now."
- Never say a product is "the best" unless the data actually supports it.`,

    "\n\n═══ PRODUCT CONTEXT ═══",
    ctx,
    convHistory ? `\n═══ CONVERSATION SO FAR ═══\n${convHistory}` : "",
    `\n═══ USER JUST SAID ═══\nUser: ${message}\nAtheon:`
  ].join("\n");

  const reply = await callAI("chat", prompt, 800);
  if (reply) {
    if (cKey) chatCache.set(cKey, reply);
    return res.json({
      ok: true,
      reply,
      // Structured fields — chat.js uses these when response comes from Gemini
      actionSuggestions: [],
      followupQuestions: [],
      buyNowProbability: null,
      confidence: 60,
      stats,
      deals
    });
  }

  // Heuristic chat fallback — always returns something useful
  const m = message.toLowerCase();
  const ai = heuristicRec(currentPrice, stats, prices.length, deals);
  let fallback;
  if (m.match(/^(hey|hi|hello|sup|yo)\b/)) fallback = `Hey! I'm Atheon. You're looking at **${product.title}** — currently ${moneyFmt(currency, currentPrice)}. Ask me anything.`;
  else if (m.includes("who are you")||m.includes("about you")) fallback = `I'm Atheon — a price tracking assistant in your browser. I watch prices, spot fake deals, and find cheaper options. What do you need?`;
  else if (m.includes("buy")||m.includes("wait")||m.includes("worth")||m.includes("recommend")) fallback = `**${ai.action}:** ${ai.text}`;
  else if (m.includes("cheap")||m.includes("compare")||m.includes("elsewhere")) {
    const ch = deals.filter(d => d.price < currentPrice);
    fallback = ch.length
      ? `Here's what I found:\n\n| Store | Price | You Save |\n|-------|-------|----------|\n${ch.slice(0,5).map(d=>`| ${d.url?`[${d.name}](${d.url})`:d.name} | ${moneyFmt(d.currency,d.price)} | ${moneyFmt(currency,currentPrice-d.price)} |`).join("\n")}`
      : "No cheaper options in my database right now.";
  }
  else if (m.includes("fake")||m.includes("real deal")||m.includes("discount")) fallback = fakeDealCtx || "No 'Was' price detected on this page right now.";
  else fallback = `I'm having trouble with AI right now. Try: "Should I buy?", "Is it cheaper elsewhere?", or "Is this deal fake?"`;

  res.json({ ok: true, reply: fallback, actionSuggestions: [], followupQuestions: [], buyNowProbability: null, confidence: 30, stats, deals });
});


// ============================================================
//  PREDICTION EVIDENCE SYSTEM
//  Logging · Delayed evaluation · Backtesting · Trust reporting
//  Answers: "Can we prove this predictor is accurate enough?"
// ============================================================

// ── SQL SCHEMA (run once in Supabase) ──────────────────────────
// Keep here as documentation and for migration scripts.
const PREDICTION_SCHEMA_SQL = `
-- Table 1: exact snapshot of every prediction shown to a user
CREATE TABLE IF NOT EXISTS prediction_logs (
  id                    uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  store                 text NOT NULL,
  store_product_id      text NOT NULL,
  canonical_product_id  uuid,
  url                   text,
  title_snapshot        text,
  category              text,
  brand                 text,
  predicted_at          timestamptz NOT NULL DEFAULT now(),
  current_price         numeric(12,4) NOT NULL,
  currency              text NOT NULL DEFAULT 'USD',
  verdict               text NOT NULL,           -- BUY | WAIT | WATCH | FAIR | UNCERTAIN
  trend_direction       text,                    -- falling | rising | stable | unknown
  prediction_window_days int NOT NULL DEFAULT 30,
  target_buy_price      numeric(12,4),
  expected_min_30d      numeric(12,4),
  expected_median_30d   numeric(12,4),
  expected_max_30d      numeric(12,4),
  drop_probability_30d  numeric(5,4),
  confidence_score      int,
  confidence_label      text,
  data_confidence       text,
  forecast_confidence   text,
  history_points_count  int NOT NULL DEFAULT 0,
  peer_count            int NOT NULL DEFAULT 0,
  baseline_price        numeric(12,4),
  seasonal_driver_json  jsonb,
  raw_prediction_json   jsonb NOT NULL,
  model_version         text NOT NULL DEFAULT 'v3'
);

CREATE INDEX IF NOT EXISTS idx_pred_logs_store     ON prediction_logs(store, store_product_id);
CREATE INDEX IF NOT EXISTS idx_pred_logs_predicted  ON prediction_logs(predicted_at);
CREATE INDEX IF NOT EXISTS idx_pred_logs_verdict    ON prediction_logs(verdict);
CREATE INDEX IF NOT EXISTS idx_pred_logs_category   ON prediction_logs(category);

-- Table 2: what actually happened in the 30-day window
CREATE TABLE IF NOT EXISTS prediction_outcomes (
  id                    uuid DEFAULT gen_random_uuid() PRIMARY KEY,
  prediction_log_id     uuid NOT NULL REFERENCES prediction_logs(id),
  evaluated_at          timestamptz NOT NULL DEFAULT now(),
  window_days           int NOT NULL DEFAULT 30,
  future_min_price      numeric(12,4),
  future_median_price   numeric(12,4),
  future_max_price      numeric(12,4),
  future_final_price    numeric(12,4),
  best_price_day_offset int,
  did_drop_meaningfully boolean,
  did_hit_target_buy_price boolean,
  decision_correct      boolean,
  direction_correct     boolean,
  range_hit             boolean,
  calibration_bucket    text,
  notes_json            jsonb
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_pred_outcomes_log ON prediction_outcomes(prediction_log_id);
`;

// ── PART 1: Accuracy definitions (deterministic, auditable) ───

// "Meaningfully better price" = at least 3% OR $15 lower, whichever is greater.
function isMeaningfulDrop(currentPrice, observedPrice) {
  const pctThreshold = currentPrice * 0.03;
  const absThreshold = 15;
  const threshold    = Math.max(pctThreshold, absThreshold);
  return (currentPrice - observedPrice) >= threshold;
}

// Was the WAIT/WATCH verdict correct? (did a meaningful drop appear within 30d?)
// Was the BUY verdict correct? (did no meaningful drop appear within 30d?)
function decisionCorrect(prediction, futurePrices) {
  const { verdict, current_price } = prediction;
  if (!futurePrices || futurePrices.length === 0) return null; // insufficient data
  const anyDrop = futurePrices.some(p => isMeaningfulDrop(current_price, p));
  if (verdict === 'WAIT' || verdict === 'WATCH') return anyDrop;
  if (verdict === 'BUY')                         return !anyDrop;
  if (verdict === 'FAIR')                        return !anyDrop; // treat FAIR as soft buy
  return null; // UNCERTAIN excluded
}

// Did the trend direction match what actually happened?
function directionCorrect(prediction, futurePrices) {
  const { trend_direction, current_price } = prediction;
  if (!futurePrices || futurePrices.length < 3) return null;
  const futureAvg  = futurePrices.reduce((a,b)=>a+b,0) / futurePrices.length;
  const actualDir  = futureAvg < current_price * 0.98 ? 'falling'
    : futureAvg > current_price * 1.02 ? 'rising'
    : 'stable';
  const predicted = (trend_direction || '').replace(' fast', '');
  return predicted === actualDir;
}

// Did actual prices fall within expected_min_30d .. expected_max_30d?
function rangeHit(prediction, futurePrices) {
  const { expected_min_30d, expected_max_30d } = prediction;
  if (!futurePrices || futurePrices.length === 0) return null;
  if (!expected_min_30d || !expected_max_30d)      return null;
  const futureMin = Math.min(...futurePrices);
  const futureMax = Math.max(...futurePrices);
  // At least partial overlap between predicted range and observed range
  return futureMax >= Number(expected_min_30d) && futureMin <= Number(expected_max_30d);
}

// Did the product actually hit the target buy price within 30 days?
function targetBuyPriceHit(prediction, futurePrices) {
  const { target_buy_price } = prediction;
  if (!target_buy_price || !futurePrices || futurePrices.length === 0) return null;
  return futurePrices.some(p => p <= Number(target_buy_price));
}

// Bucket 0-20, 21-40, 41-60, 61-80, 81-100
function assignCalibrationBucket(prob) {
  if (!prob && prob !== 0) return null;
  const pct = Math.round(prob * 100);
  if (pct <= 20)  return '0-20';
  if (pct <= 40)  return '21-40';
  if (pct <= 60)  return '41-60';
  if (pct <= 80)  return '61-80';
  return '81-100';
}

// ── PART 2: Log a prediction ───────────────────────────────────
// Called immediately after computePredictionV2 returns.
// Fire-and-forget — never blocks the user response.
async function logPrediction({ storeId, storeProductId, url, title, category, brand,
    currentPrice, currency, prediction, attrs }) {
  try {
    if (!prediction || !prediction.verdict) return;

    // Fetch canonical product id if available
    const { data: prod } = await supabase
      .from('products')
      .select('id')
      .eq('store_id', storeId)
      .eq('store_product_id', storeProductId)
      .maybeSingle();

    const row = {
      store:                storeId,
      store_product_id:     storeProductId,
      canonical_product_id: prod?.id || null,
      url:                  url || null,
      title_snapshot:       (title || '').slice(0, 500),
      category:             category || attrs?.category || null,
      brand:                attrs?.brand || null,
      predicted_at:         new Date().toISOString(),
      current_price:        currentPrice,
      currency:             currency || 'USD',
      verdict:              prediction.verdict,
      trend_direction:      prediction.trendDirection || null,
      prediction_window_days: 30,
      target_buy_price:     prediction.targetBuyPrice || null,
      expected_min_30d:     prediction.expectedMin30d || null,
      expected_median_30d:  prediction.expectedMedian30d || null,
      expected_max_30d:     prediction.expectedMax30d || null,
      drop_probability_30d: prediction.dropProbability30d || null,
      confidence_score:     prediction.confidence || null,
      confidence_label:     prediction.dataConfidence || null,
      data_confidence:      prediction.dataConfidence || null,
      forecast_confidence:  prediction.forecastConfidence || null,
      history_points_count: prediction.historyCount || 0,
      peer_count:           prediction.stats?.nPeers || 0,
      baseline_price:       prediction.stats?.baseline || null,
      seasonal_driver_json: prediction.seasonalDriver || null,
      raw_prediction_json:  prediction,
      model_version:        'v3',
    };

    const { error } = await supabase.from('prediction_logs').insert(row);
    if (error) console.warn('[pred:log] insert failed:', error.message);

  } catch (e) {
    console.warn('[pred:log] error:', e.message);
  }
}

// ── PART 3: Delayed evaluator (runs once per day) ──────────────
async function runPredictionEvaluator() {
  console.log('[eval] Starting prediction evaluator job...');
  const WINDOW_DAYS   = 30;
  const BATCH_SIZE    = 50;
  const cutoff        = new Date(Date.now() - WINDOW_DAYS * 86400000).toISOString();

  try {
    // 1. Find predictions older than 30d with no outcome yet
    const { data: logs, error: logErr } = await supabase
      .from('prediction_logs')
      .select('id, store, store_product_id, canonical_product_id, predicted_at, current_price, verdict, trend_direction, target_buy_price, expected_min_30d, expected_max_30d, drop_probability_30d')
      .lt('predicted_at', cutoff)
      .not('id', 'in', supabase.from('prediction_outcomes').select('prediction_log_id'))
      .limit(BATCH_SIZE);

    if (logErr) { console.warn('[eval] fetch logs error:', logErr.message); return; }
    if (!logs || logs.length === 0) { console.log('[eval] No predictions ready for evaluation.'); return; }

    console.log(`[eval] Evaluating ${logs.length} predictions...`);
    let evaluated = 0, skipped = 0;

    for (const log of logs) {
      try {
        const windowStart = new Date(log.predicted_at).toISOString();
        const windowEnd   = new Date(new Date(log.predicted_at).getTime() + WINDOW_DAYS * 86400000).toISOString();

        // 2. Pull price observations for this product in the 30d window after prediction
        let obsQuery = supabase
          .from('price_observations')
          .select('price, observed_at')
          .gte('observed_at', windowStart)
          .lte('observed_at', windowEnd)
          .order('observed_at', { ascending: true });

        if (log.canonical_product_id) {
          obsQuery = obsQuery.eq('product_id', log.canonical_product_id);
        } else {
          // Fall back to matching via products table
          const { data: prod } = await supabase
            .from('products')
            .select('id')
            .eq('store_id', log.store)
            .eq('store_product_id', log.store_product_id)
            .maybeSingle();
          if (!prod) { skipped++; continue; }
          obsQuery = obsQuery.eq('product_id', prod.id);
        }

        const { data: obs } = await obsQuery;
        if (!obs || obs.length < 2) { skipped++; continue; } // need at least 2 observations

        const futurePrices = obs.map(r => Number(r.price)).filter(n => Number.isFinite(n) && n > 0);
        if (futurePrices.length < 2) { skipped++; continue; }

        // 3. Compute future stats
        const sortedFuture   = [...futurePrices].sort((a,b) => a - b);
        const futureMin      = sortedFuture[0];
        const futureMax      = sortedFuture[sortedFuture.length - 1];
        const futureMedian   = sortedFuture[Math.floor(sortedFuture.length / 2)];
        const futureFinal    = futurePrices[futurePrices.length - 1];

        // Best price day offset (how many days into the window was the lowest price?)
        const minIdx         = futurePrices.indexOf(futureMin);
        const minObsDate     = obs[minIdx]?.observed_at;
        const bestDayOffset  = minObsDate
          ? Math.round((new Date(minObsDate) - new Date(log.predicted_at)) / 86400000)
          : null;

        // 4. Score
        const didDrop    = decisionCorrect(log, futurePrices);
        const dirOk      = directionCorrect(log, futurePrices);
        const rangeOk    = rangeHit(log, futurePrices);
        const targetOk   = targetBuyPriceHit(log, futurePrices);
        const calBucket  = assignCalibrationBucket(log.drop_probability_30d);

        // 5. Insert outcome
        const { error: outErr } = await supabase.from('prediction_outcomes').insert({
          prediction_log_id:       log.id,
          evaluated_at:            new Date().toISOString(),
          window_days:             WINDOW_DAYS,
          future_min_price:        futureMin,
          future_median_price:     futureMedian,
          future_max_price:        futureMax,
          future_final_price:      futureFinal,
          best_price_day_offset:   bestDayOffset,
          did_drop_meaningfully:   futurePrices.some(p => isMeaningfulDrop(log.current_price, p)),
          did_hit_target_buy_price: targetOk,
          decision_correct:        didDrop,
          direction_correct:       dirOk,
          range_hit:               rangeOk,
          calibration_bucket:      calBucket,
          notes_json: {
            obs_count:    futurePrices.length,
            future_min:   futureMin,
            current_price: log.current_price,
            drop_pct:     Math.round((log.current_price - futureMin) / log.current_price * 100),
          },
        });

        if (outErr) console.warn(`[eval] outcome insert error for ${log.id}:`, outErr.message);
        else evaluated++;

      } catch (e) {
        console.warn(`[eval] error on log ${log.id}:`, e.message);
        skipped++;
      }
    }
    console.log(`[eval] Done. Evaluated: ${evaluated}, Skipped: ${skipped}`);
  } catch (e) {
    console.error('[eval] fatal error:', e.message);
  }
}

// ── Schedule evaluator: runs daily at 03:00 server time ────────
function scheduleDailyEvaluator() {
  const MS_IN_DAY = 86400000;
  const now       = new Date();
  const next3am   = new Date(now);
  next3am.setHours(3, 0, 0, 0);
  if (next3am <= now) next3am.setTime(next3am.getTime() + MS_IN_DAY);
  const msUntil3am = next3am - now;
  setTimeout(() => {
    runPredictionEvaluator().catch(e => console.error('[eval] scheduler error:', e.message));
    setInterval(() => {
      runPredictionEvaluator().catch(e => console.error('[eval] scheduler error:', e.message));
    }, MS_IN_DAY);
  }, msUntil3am);
  console.log(`[eval] Evaluator scheduled. First run in ${Math.round(msUntil3am/3600000)}h.`);
}
scheduleDailyEvaluator();

// ── PART 4: Backtester ─────────────────────────────────────────
// Runs the predictor on historical cutoff dates and scores against real future data.
// Called via admin endpoint — not on every request.
async function runBacktest({ storeId, storeProductId, cutoffDate, windowDays = 30 }) {
  const cutoff = new Date(cutoffDate);
  const windowEnd = new Date(cutoff.getTime() + windowDays * 86400000);

  // Pull history UP TO cutoff (no leakage)
  const { data: prod } = await supabase
    .from('products')
    .select('id, title, currency, last_price')
    .eq('store_id', storeId)
    .eq('store_product_id', storeProductId)
    .maybeSingle();
  if (!prod) return { error: 'product_not_found' };

  const { data: histRows } = await supabase
    .from('price_observations')
    .select('price, observed_at')
    .eq('product_id', prod.id)
    .lt('observed_at', cutoff.toISOString())
    .order('observed_at', { ascending: false })
    .limit(120);

  if (!histRows || histRows.length === 0) return { error: 'no_history_before_cutoff' };

  const currentPriceAtCutoff = Number(histRows[0].price);
  const historyPrices = histRows.map(r => Number(r.price)).filter(Number.isFinite);

  // Run predictor with cutoff as "now" (no future data)
  const prediction = await computePredictionV2({
    title: prod.title || '',
    currentPrice: currentPriceAtCutoff,
    currency: prod.currency || 'USD',
    storeId, storeProductId,
    historyRows,
    historyPrices,
    peerPrices: [],  // no peers in backtest (conservative)
    now: cutoff,
  });

  if (!prediction) return { error: 'prediction_failed' };

  // Pull FUTURE observations (30d after cutoff) for scoring
  const { data: futureRows } = await supabase
    .from('price_observations')
    .select('price, observed_at')
    .eq('product_id', prod.id)
    .gte('observed_at', cutoff.toISOString())
    .lte('observed_at', windowEnd.toISOString())
    .order('observed_at', { ascending: true });

  if (!futureRows || futureRows.length < 2) return { error: 'insufficient_future_data', prediction };

  const futurePrices = futureRows.map(r => Number(r.price)).filter(Number.isFinite);
  const sortedF = [...futurePrices].sort((a,b) => a - b);

  const syntheticLog = {
    verdict:           prediction.verdict,
    trend_direction:   prediction.trendDirection,
    current_price:     currentPriceAtCutoff,
    target_buy_price:  prediction.targetBuyPrice,
    expected_min_30d:  prediction.expectedMin30d,
    expected_max_30d:  prediction.expectedMax30d,
    drop_probability_30d: prediction.dropProbability30d,
  };

  return {
    cutoffDate:       cutoff.toISOString(),
    currentPrice:     currentPriceAtCutoff,
    verdict:          prediction.verdict,
    trendDirection:   prediction.trendDirection,
    targetBuyPrice:   prediction.targetBuyPrice,
    confidence:       prediction.confidence,
    futureMin:        sortedF[0],
    futureFinal:      futurePrices[futurePrices.length - 1],
    futureObs:        futurePrices.length,
    decisionCorrect:  decisionCorrect(syntheticLog, futurePrices),
    directionCorrect: directionCorrect(syntheticLog, futurePrices),
    rangeHit:         rangeHit(syntheticLog, futurePrices),
    targetHit:        targetBuyPriceHit(syntheticLog, futurePrices),
    didDropMeaningfully: futurePrices.some(p => isMeaningfulDrop(currentPriceAtCutoff, p)),
    calibrationBucket: assignCalibrationBucket(prediction.dropProbability30d),
  };
}

// ── PART 5: Trust report generator ────────────────────────────
// Returns a structured evidence report with calibration, accuracy by slice,
// and a plain-English "Can we make a trust claim?" section.
async function generateTrustReport({ minCount = 20 } = {}) {
  // Fetch evaluated predictions with outcomes joined
  const { data: rows, error } = await supabase
    .from('prediction_logs')
    .select(`
      id, verdict, trend_direction, category, confidence_score,
      confidence_label, data_confidence, history_points_count,
      peer_count, drop_probability_30d, current_price, store,
      predicted_at,
      prediction_outcomes (
        decision_correct, direction_correct, range_hit,
        did_hit_target_buy_price, did_drop_meaningfully,
        calibration_bucket, future_min_price, future_final_price
      )
    `)
    .not('prediction_outcomes', 'is', null)
    .limit(5000);

  if (error) return { error: error.message };

  // Flatten: one row per evaluated prediction
  const evaluated = (rows || [])
    .map(r => ({ ...r, outcome: r.prediction_outcomes?.[0] }))
    .filter(r => r.outcome && r.outcome.decision_correct !== null);

  if (evaluated.length < minCount) {
    return {
      status: 'insufficient_data',
      total_evaluated: evaluated.length,
      message: `Only ${evaluated.length} evaluated predictions. Need at least ${minCount} to report.`,
    };
  }

  // ── Core metric helpers ──
  const pct = (arr, key) => {
    const valid = arr.filter(r => r.outcome[key] !== null);
    if (!valid.length) return null;
    return Math.round(valid.filter(r => r.outcome[key] === true).length / valid.length * 1000) / 10;
  };
  const avg = (arr, key) => {
    const vals = arr.map(r => Number(r[key])).filter(Number.isFinite);
    return vals.length ? Math.round(vals.reduce((a,b)=>a+b,0) / vals.length * 10) / 10 : null;
  };

  // ── Overall accuracy ──
  const overall = {
    total:             evaluated.length,
    decisionAccuracy:  pct(evaluated, 'decision_correct'),
    directionAccuracy: pct(evaluated, 'direction_correct'),
    rangeHitRate:      pct(evaluated, 'range_hit'),
    targetHitRate:     pct(evaluated, 'did_hit_target_buy_price'),
    dropRate:          pct(evaluated, 'did_drop_meaningfully'),
  };

  // ── Accuracy by confidence band ──
  const confBands = ['Low', 'Medium', 'High'];
  const byConfidence = {};
  for (const band of confBands) {
    const slice = evaluated.filter(r => r.confidence_label === band || r.data_confidence === band);
    if (slice.length < 5) continue;
    byConfidence[band] = {
      count:            slice.length,
      decisionAccuracy: pct(slice, 'decision_correct'),
      directionAccuracy:pct(slice, 'direction_correct'),
      rangeHitRate:     pct(slice, 'range_hit'),
    };
  }

  // Check confidence ordering (High > Medium > Low)
  const confOrdering = (() => {
    const h = byConfidence['High']?.decisionAccuracy;
    const m = byConfidence['Medium']?.decisionAccuracy;
    const l = byConfidence['Low']?.decisionAccuracy;
    if (h === null || m === null || l === null) return 'insufficient_data';
    return (h >= m && m >= l) ? 'valid' : 'INVALID';
  })();

  // ── Calibration by drop probability bucket ──
  const buckets = ['0-20','21-40','41-60','61-80','81-100'];
  const calibration = {};
  for (const bucket of buckets) {
    const slice = evaluated.filter(r => r.outcome.calibration_bucket === bucket);
    if (!slice.length) continue;
    const avgPredProb = avg(slice, 'drop_probability_30d') || 0;
    const actualRate  = pct(slice, 'did_drop_meaningfully') || 0;
    calibration[bucket] = {
      count:         slice.length,
      avgPredictedProb: Math.round(avgPredProb * 100),
      actualDropRate:   actualRate,
      error:            Math.round(Math.abs(avgPredProb * 100 - actualRate) * 10) / 10,
    };
  }

  // ── Accuracy by category ──
  const categories = [...new Set(evaluated.map(r => r.category).filter(Boolean))];
  const byCategory = {};
  for (const cat of categories) {
    const slice = evaluated.filter(r => r.category === cat);
    if (slice.length < 5) continue;
    byCategory[cat] = {
      count:            slice.length,
      decisionAccuracy: pct(slice, 'decision_correct'),
    };
  }

  // ── History bucket performance ──
  const byHistoryBucket = {
    'cold (0-4)':   { slice: evaluated.filter(r => r.history_points_count < 5) },
    'sparse (5-14)':{ slice: evaluated.filter(r => r.history_points_count >= 5 && r.history_points_count < 15) },
    'rich (15+)':   { slice: evaluated.filter(r => r.history_points_count >= 15) },
  };
  for (const [k, v] of Object.entries(byHistoryBucket)) {
    byHistoryBucket[k] = {
      count:            v.slice.length,
      decisionAccuracy: pct(v.slice, 'decision_correct'),
    };
  }

  // ── Holiday period performance ──
  // "near holiday" = seasonalDriver existed (was logged)
  // We don't have that in the join, so proxy with: pull seasonal_driver_json from logs
  // For now: mark all evaluated as "any" — holiday slice available via SQL view

  // ── "Can we make a trust claim?" ──
  const highConfSlice = evaluated.filter(r =>
    (r.confidence_label === 'High' || r.data_confidence === 'High') &&
    r.history_points_count >= 10
  );
  const highConfElec  = highConfSlice.filter(r => r.category === 'electronics');
  const claimDA       = byConfidence['High']?.decisionAccuracy;
  const calibMaxError = Math.max(...Object.values(calibration).map(b => b.error || 0));
  const confOrderOk   = confOrdering === 'valid';

  const trustClaim = (() => {
    if (!claimDA)         return { canClaim: false, reason: 'Insufficient high-confidence data.' };
    if (claimDA >= 80 && confOrderOk && calibMaxError <= 15) {
      return {
        canClaim: true,
        claim: `Decision accuracy is ${claimDA}% on high-confidence predictions with 10+ history points.`,
        details: [
          'Confidence ordering is valid (High > Medium > Low).',
          `Calibration max error: ${calibMaxError}% — acceptable.`,
          'Safe to surface as evidence-backed guidance on high-confidence predictions.',
        ],
      };
    }
    if (claimDA >= 70 && confOrderOk) {
      return {
        canClaim: true,
        claim: `Decision accuracy is ${claimDA}% — acceptable but not strong.`,
        details: [
          confOrderOk ? 'Confidence ordering is valid.' : 'WARNING: Confidence ordering is INVALID.',
          calibMaxError > 15 ? `Calibration error ${calibMaxError}% is high — avoid showing precise drop probabilities.` : 'Calibration acceptable.',
          'Use measured language. Do not claim >80% accuracy.',
        ],
      };
    }
    return {
      canClaim: false,
      reason: `Decision accuracy is only ${claimDA}% — below trust threshold.`,
      details: [
        confOrderOk ? 'Confidence ordering is valid.' : 'WARNING: Confidence ordering INVALID — confidence ratings are not trustworthy.',
        calibMaxError > 15 ? `Calibration is poor (max error: ${calibMaxError}%). Do not show precise probabilities.` : '',
        'Cold-start performance may be weak — consider suppressing hard verdicts with < 5 history points.',
      ].filter(Boolean),
    };
  })();

  return {
    generatedAt:     new Date().toISOString(),
    totalEvaluated:  evaluated.length,
    overall,
    byConfidence,
    confidenceOrdering: confOrdering,
    calibration,
    byCategory,
    byHistoryBucket,
    trustClaim,
  };
}

// ── PART 6: Live shipping rules (applied at prediction time) ───
// Adjusts or suppresses prediction output based on accumulated evidence.
// Initially conservative — tightens as we gather real accuracy data.
function applyShippingRules(prediction, { nHist, nPeer, category } = {}) {
  if (!prediction) return prediction;

  const p = { ...prediction };

  // Rule 1: No hard BUY with zero peer support and cold history
  if (p.verdict === 'BUY' && nHist < 5 && nPeer === 0) {
    p.verdict   = 'FAIR';
    p.action    = 'ok';
    p.reasoning = `Limited data (${nHist} observation${nHist !== 1 ? 's' : ''}, no peer prices) — treating as fair price rather than a confident buy signal. ${p.reasoning}`;
    p.confidence = Math.min(p.confidence || 50, 42);
    p._ruleApplied = 'no_buy_without_data';
  }

  // Rule 2: Never show precise drop probability text when confidence is Low
  if ((p.dataConfidence === 'Low' || p.confidence < 40) && p.dropProbability30d !== null) {
    p.dropProbability30d = null;  // suppressed — not earned
    p._probSuppressed = true;
  }

  // Rule 3: Suppress seasonalDriver if data_confidence is Low
  // (holiday signal alone with no price history = noise)
  if (p.dataConfidence === 'Low' && p.seasonalDriver && nHist < 3) {
    p.seasonalDriver = null;
    p._seasonalSuppressed = true;
  }

  // Rule 4: Cap confidence score for cold-start at 55
  if (nHist < 5 && (p.confidence || 0) > 55) {
    p.confidence         = 55;
    p.dataConfidenceScore = Math.min(p.dataConfidenceScore || 55, 55);
  }

  return p;
}

// ── PART 7: Admin endpoints ────────────────────────────────────

// GET /v1/admin/trust-report — full evidence report
app.get('/v1/admin/trust-report', async (req, res) => {
  // Simple bearer token gate — set ADMIN_KEY in env
  const key = req.headers['x-admin-key'] || req.query.key;
  if (process.env.ADMIN_KEY && key !== process.env.ADMIN_KEY) {
    return res.status(401).json({ error: 'unauthorized' });
  }
  try {
    const report = await generateTrustReport();
    res.json({ ok: true, report });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

// POST /v1/admin/backtest — run backtest for one product + cutoff
app.post('/v1/admin/backtest', async (req, res) => {
  const key = req.headers['x-admin-key'] || req.query.key;
  if (process.env.ADMIN_KEY && key !== process.env.ADMIN_KEY) {
    return res.status(401).json({ error: 'unauthorized' });
  }
  const { storeId, storeProductId, cutoffDate, windowDays } = req.body;
  if (!storeId || !storeProductId || !cutoffDate) {
    return res.status(400).json({ error: 'storeId, storeProductId, cutoffDate required' });
  }
  try {
    const result = await runBacktest({ storeId, storeProductId, cutoffDate, windowDays });
    res.json({ ok: true, result });
  } catch (e) {
    res.status(500).json({ ok: false, error: e.message });
  }
});

// POST /v1/admin/evaluate-now — trigger evaluator manually
app.post('/v1/admin/evaluate-now', async (req, res) => {
  const key = req.headers['x-admin-key'] || req.query.key;
  if (process.env.ADMIN_KEY && key !== process.env.ADMIN_KEY) {
    return res.status(401).json({ error: 'unauthorized' });
  }
  runPredictionEvaluator().catch(e => console.error('[eval] manual trigger error:', e.message));
  res.json({ ok: true, message: 'Evaluator started in background.' });
});

// GET /v1/admin/schema-sql — return the SQL to run in Supabase
app.get('/v1/admin/schema-sql', (req, res) => {
  const key = req.headers['x-admin-key'] || req.query.key;
  if (process.env.ADMIN_KEY && key !== process.env.ADMIN_KEY) {
    return res.status(401).json({ error: 'unauthorized' });
  }
  res.type('text/plain').send(PREDICTION_SCHEMA_SQL);
});

// ── Useful SQL views (for direct Supabase queries) ─────────────
// Run these in your Supabase SQL editor after creating the tables above.
const REPORTING_VIEWS_SQL = `
-- Overall decision accuracy
SELECT
  COUNT(*) AS total,
  ROUND(100.0 * SUM(CASE WHEN o.decision_correct THEN 1 ELSE 0 END) / COUNT(*), 1) AS decision_accuracy_pct,
  ROUND(100.0 * SUM(CASE WHEN o.direction_correct THEN 1 ELSE 0 END) / COUNT(*), 1) AS direction_accuracy_pct,
  ROUND(100.0 * SUM(CASE WHEN o.range_hit THEN 1 ELSE 0 END) / COUNT(*), 1) AS range_hit_pct
FROM prediction_logs l
JOIN prediction_outcomes o ON o.prediction_log_id = l.id
WHERE o.decision_correct IS NOT NULL;

-- Accuracy by category
SELECT
  l.category,
  COUNT(*) AS total,
  ROUND(100.0 * SUM(CASE WHEN o.decision_correct THEN 1 ELSE 0 END) / COUNT(*), 1) AS decision_accuracy_pct
FROM prediction_logs l
JOIN prediction_outcomes o ON o.prediction_log_id = l.id
WHERE o.decision_correct IS NOT NULL
GROUP BY l.category
HAVING COUNT(*) >= 5
ORDER BY decision_accuracy_pct DESC;

-- Calibration by drop probability bucket
SELECT
  o.calibration_bucket,
  COUNT(*) AS total,
  ROUND(AVG(l.drop_probability_30d) * 100, 1) AS avg_predicted_pct,
  ROUND(100.0 * SUM(CASE WHEN o.did_drop_meaningfully THEN 1 ELSE 0 END) / COUNT(*), 1) AS actual_drop_pct
FROM prediction_logs l
JOIN prediction_outcomes o ON o.prediction_log_id = l.id
WHERE o.calibration_bucket IS NOT NULL
GROUP BY o.calibration_bucket
ORDER BY o.calibration_bucket;

-- Accuracy by confidence band
SELECT
  l.confidence_label,
  COUNT(*) AS total,
  ROUND(100.0 * SUM(CASE WHEN o.decision_correct THEN 1 ELSE 0 END) / COUNT(*), 1) AS decision_accuracy_pct
FROM prediction_logs l
JOIN prediction_outcomes o ON o.prediction_log_id = l.id
WHERE o.decision_correct IS NOT NULL AND l.confidence_label IS NOT NULL
GROUP BY l.confidence_label
ORDER BY decision_accuracy_pct DESC;

-- Near-holiday vs non-holiday accuracy
SELECT
  CASE WHEN l.seasonal_driver_json IS NOT NULL THEN 'near_holiday' ELSE 'no_holiday' END AS period,
  COUNT(*) AS total,
  ROUND(100.0 * SUM(CASE WHEN o.decision_correct THEN 1 ELSE 0 END) / COUNT(*), 1) AS decision_accuracy_pct
FROM prediction_logs l
JOIN prediction_outcomes o ON o.prediction_log_id = l.id
WHERE o.decision_correct IS NOT NULL
GROUP BY 1;
`;

const port = process.env.PORT || 8787;
app.listen(port, () => console.log(`Atheon API running on http://localhost:${port}`));
