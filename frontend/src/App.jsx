import { useEffect, useState, useCallback } from "react";
import Globe from "./Globe";
import {
  runQuery, generateNDVIMap, generateNDWIMap,
  runChangeDetection, semanticSearch, splitDateRange, defaultDateWindow,
} from "./api";
import {
  ArrowRight, Menu, X, Leaf, Droplets, Activity,
  MapPin, Satellite, BarChart3, CheckCircle2, AlertCircle,
  Search, Sparkles, Database, Brain, Layers3, Globe2,
  Target, Eye, Loader2, TrendingDown, TrendingUp, Minus,
  FlaskConical, Info, ChevronDown, ChevronUp,
} from "lucide-react";

const ANALYSIS_META = {
  ndvi: { label: "Vegetation Health", icon: null, color: "#4ade80", description: "NDVI — Normalized Difference Vegetation Index", triggerMap: "ndvi" },
  ndwi: { label: "Water Bodies", icon: null, color: "#38bdf8", description: "NDWI — Normalized Difference Water Index", triggerMap: "ndwi" },
  water_extent: { label: "Water Extent", icon: null, color: "#38bdf8", description: "Water extent analysis via NDWI", triggerMap: "ndwi" },
  change: { label: "Change Detection", icon: null, color: "#fb923c", description: "Temporal NDVI change detection", triggerMap: "change" },
  water_extent_change: { label: "Water Extent Change", icon: null, color: "#a78bfa", description: "Water extent change over time", triggerMap: "change" },
  discovery: { label: "Scene Discovery", icon: null, color: "#e2e8f0", description: "Satellite scene discovery & metadata", triggerMap: null },
};

const EXAMPLE_QUERIES = [
  { text: "Analyze vegetation health around Bengaluru" },
  { text: "Find lakes around Bengaluru" },
  { text: "Compare lake water extent since last year" },
];

function formatLabel(key) {
  if (!key) return "Unknown";
  return String(key).replaceAll("_", " ").replace(/\b\w/g, (l) => l.toUpperCase());
}
function formatValue(value) {
  if (value === null || value === undefined) return "—";
  if (typeof value === "number")
    return Number.isInteger(value) ? value.toLocaleString() : value.toLocaleString(undefined, { maximumFractionDigits: 4 });
  return String(value);
}
function getScalarMetrics(metrics) {
  if (!metrics) return [];
  return Object.entries(metrics).filter(([, v]) => typeof v !== "object" || v === null);
}

function QueryInterpretationCard({ result }) {
  const sq = result?.structured_query;
  const geo = result?.geographic_result;
  const meta = ANALYSIS_META[sq?.analysis] || ANALYSIS_META.discovery;
  return (
    <div className="qi-card">
      <div className="qi-header"><FlaskConical size={16} /><span>QUERY INTERPRETATION</span></div>
      <div className="qi-row">
        <div className="qi-item">
          <span><MapPin size={12} /> LOCATION</span>
          <strong>{geo?.name || sq?.location || "Not specified"}</strong>
          {geo?.latitude && <small>{Number(geo.latitude).toFixed(4)}, {Number(geo.longitude).toFixed(4)}</small>}
        </div>
        <div className="qi-item">
          <span><Brain size={12} /> INTENT</span>
          <strong>{meta.label}</strong>
          <small>{meta.description}</small>
        </div>
        <div className="qi-item">
          <span><Layers3 size={12} /> ANALYSIS</span>
          <div className="qi-badge" style={{ background: `${meta.color}22`, borderColor: `${meta.color}55`, color: meta.color }}>
            {sq?.analysis?.toUpperCase()}
          </div>
        </div>
        <div className="qi-item">
          <span><Eye size={12} /> STATUS</span>
          {result?.validation_status === "passed"
            ? <div className="qi-status ok"><CheckCircle2 size={13} /> Ready</div>
            : <div className="qi-status warn"><AlertCircle size={13} /> {result?.validation_status || "—"}</div>}
        </div>
      </div>
    </div>
  );
}

function SatelliteMetadata({ scene }) {
  if (!scene) return null;
  return (
    <div className="scene-card">
      <div className="scene-card-title">SATELLITE OBSERVATION</div>
      <div className="scene-grid">
        <div><span>SCENE ID</span><strong className="scene-id-val">{scene.scene_id || "—"}</strong></div>
        <div><span>ACQUIRED</span><strong>{scene.datetime ? new Date(scene.datetime).toLocaleString() : "—"}</strong></div>
        <div><span>CLOUD COVER</span><strong>{scene.cloud_cover != null ? `${scene.cloud_cover}%` : "—"}</strong></div>
        <div><span>VALID PIXELS</span><strong>{scene.valid_pixel_fraction != null ? `${(scene.valid_pixel_fraction * 100).toFixed(1)}%` : "—"}</strong></div>
      </div>
    </div>
  );
}

function MetricGrid({ metrics }) {
  const entries = getScalarMetrics(metrics);
  if (!entries.length) return null;
  return (
    <div className="metrics">
      {entries.map(([key, value]) => (
        <div className="metric metric-highlight" key={key}>
          <span>{formatLabel(key)}</span>
          <strong>{formatValue(value)}</strong>
        </div>
      ))}
    </div>
  );
}

function NDVIResults({ analysisResult }) {
  return (
    <div className="analysis-result-block ndvi-block">
      <div className="arb-header">
        <Leaf size={18} color="#4ade80" />
        <div><h3>Vegetation Health (NDVI)</h3><p>Normalized Difference Vegetation Index — Sentinel-2 L2A</p></div>
      </div>
      <MetricGrid metrics={analysisResult?.metrics} />
      <div className="ndvi-legend">
        <span style={{ color: "#ef4444" }}>–1.0 Bare / Water</span>
        <div className="ndvi-bar" />
        <span style={{ color: "#4ade80" }}>+1.0 Dense Vegetation</span>
      </div>
    </div>
  );
}

function NDWIResults({ analysisResult, label }) {
  return (
    <div className="analysis-result-block ndwi-block">
      <div className="arb-header">
        <Droplets size={18} color="#38bdf8" />
        <div><h3>{label || "Water Bodies (NDWI)"}</h3><p>Normalized Difference Water Index — Sentinel-2 L2A</p></div>
      </div>
      <MetricGrid metrics={analysisResult?.metrics} />
      <div className="ndwi-legend">
        <span style={{ color: "#92400e" }}>–1.0 Dry / Vegetation</span>
        <div className="ndwi-bar" />
        <span style={{ color: "#38bdf8" }}>+1.0 Open Water</span>
      </div>
    </div>
  );
}

function WaterExtentResults({ metrics }) {
  if (!metrics) return null;
  const extent = metrics.water_extent_km2 ?? metrics.water_area_km2 ?? null;
  const fraction = metrics.water_pixel_fraction ?? metrics.water_fraction ?? null;
  if (extent == null && fraction == null) return null;
  return (
    <div className="water-extent-block">
      <div className="web-label">WATER EXTENT</div>
      <div className="web-grid">
        {extent != null && <div className="web-stat"><strong>{Number(extent).toFixed(2)} km²</strong><span>Water Area</span></div>}
        {fraction != null && <div className="web-stat"><strong>{(Number(fraction) * 100).toFixed(1)}%</strong><span>Water Fraction</span></div>}
      </div>
    </div>
  );
}

function ChangeDetectionResults({ changeData }) {
  if (!changeData?.result) return null;
  const r = changeData.result;
  const delta = r.mean_ndvi_change ?? null;
  const trendColor = delta == null ? "#e2e8f0" : delta > 0.01 ? "#4ade80" : delta < -0.01 ? "#f87171" : "#e2e8f0";
  const TrendIcon = delta == null ? Minus : delta > 0.01 ? TrendingUp : delta < -0.01 ? TrendingDown : Minus;
  return (
    <div className="change-detection-card" id="change-detection-result">
      <div className="change-detection-header">
        <div>
          <div className="section-label">CHANGE DETECTION</div>
          <h3>Vegetation Change Analysis</h3>
          <p>Comparison between two satellite observation periods.</p>
        </div>
        <div className="change-status"><CheckCircle2 size={14} /> ANALYSIS COMPLETE</div>
      </div>
      <div className="change-scenes">
        <div><span>FIRST SCENE</span><strong>{r.first_scene_id || "—"}</strong></div>
        <div><span>SECOND SCENE</span><strong>{r.second_scene_id || "—"}</strong></div>
      </div>
      <div className="change-result-grid">
        <div className="change-result-item"><span>FIRST NDVI MEAN</span><strong>{r.first_ndvi_mean?.toFixed(4) ?? "—"}</strong></div>
        <div className="change-result-item"><span>SECOND NDVI MEAN</span><strong>{r.second_ndvi_mean?.toFixed(4) ?? "—"}</strong></div>
        <div className="change-result-item">
          <span>MEAN NDVI CHANGE</span>
          <strong style={{ color: trendColor }}><TrendIcon size={14} style={{ verticalAlign: "middle" }} /> {delta?.toFixed(4) ?? "—"}</strong>
        </div>
        <div className="change-result-item"><span>MINIMUM CHANGE</span><strong>{r.minimum_change?.toFixed(4) ?? "—"}</strong></div>
        <div className="change-result-item"><span>MAXIMUM CHANGE</span><strong>{r.maximum_change?.toFixed(4) ?? "—"}</strong></div>
      </div>
    </div>
  );
}

function MapDisplay({ mapUrl, mapType, mapStats }) {
  if (!mapUrl) return null;
  const isNDVI = mapType === "ndvi";
  return (
    <div className="map-display" id="satellite-map">
      <div className="map-display-header">
        <div className="map-title-group">
          <div className="map-title-row">
            {isNDVI ? <Leaf size={18} /> : <Droplets size={18} />}
            <div>
              <h3>{isNDVI ? "Vegetation Health" : "Water & Moisture"}</h3>
              <span>{mapType.toUpperCase()} SATELLITE ANALYSIS</span>
            </div>
          </div>
          {mapStats && (
            <div className="map-stats-row">
              {Object.entries(mapStats).filter(([, v]) => typeof v === "number").map(([k, v]) => (
                <span key={k} className="map-stat-pill">{formatLabel(k)}: {Number(v).toFixed(4)}</span>
              ))}
            </div>
          )}
        </div>
        <a className="map-open-button" href={mapUrl} target="_blank" rel="noreferrer">Open full map <ArrowRight size={15} /></a>
      </div>
      <div className="map-image-wrapper">
        <img src={mapUrl} alt={`${mapType} satellite analysis`} />
      </div>
    </div>
  );
}

function ExplanationPanel({ explanation }) {
  if (!explanation) return null;
  return (
    <div className="explanation">
      <div className="explanation-label"><Brain size={14} /> AI-GROUNDED EXPLANATION</div>
      <p>{explanation}</p>
    </div>
  );
}

function AutoMapStatus({ loading, error, mapType }) {
  if (!loading && !error) return null;
  return (
    <div className={`auto-map-notice ${error ? "error" : "loading"}`}>
      {loading && <><Loader2 size={14} className="spin" /> Auto-generating {mapType?.toUpperCase()} map…</>}
      {error && <><AlertCircle size={14} /><span>Map unavailable: {error}</span><small> (Text results shown below.)</small></>}
    </div>
  );
}

function SemanticSearchResults({ searchResults, onError }) {
  if (!searchResults) return null;
  return (
    <section id="semantic-search-results" style={{ padding: "70px 6vw", background: "#07120d", color: "#eef8f1" }}>
      <div style={{ maxWidth: "1200px", margin: "0 auto" }}>
        <div className="section-label">REMOTECLIP · FAISS SEARCH</div>
        <div style={{ display: "flex", justifyContent: "space-between", gap: "20px", alignItems: "end", flexWrap: "wrap" }}>
          <div>
            <h2 style={{ marginBottom: "8px" }}>Semantic Satellite Results</h2>
            <p style={{ opacity: 0.72, margin: 0 }}>"{searchResults.visual_query}" · {searchResults.candidates.length} candidate tile(s)</p>
          </div>
          <div style={{ opacity: 0.6, fontSize: "12px" }}>QUERY ID · {searchResults.query_id}</div>
        </div>
        {searchResults.candidates.length === 0
          ? <p style={{ marginTop: "30px", opacity: 0.7 }}>No matching satellite tiles found.</p>
          : (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: "20px", marginTop: "30px" }}>
              {searchResults.candidates.map((candidate, index) => (
                <article key={candidate.tile_id} style={{ overflow: "hidden", borderRadius: "18px", border: "1px solid rgba(150,210,170,0.18)", background: "rgba(255,255,255,0.035)" }}>
                  {candidate.image_url ? (
                    <img src={candidate.image_url} alt={`Tile ${candidate.tile_id}`} style={{ width: "100%", aspectRatio: "1/1", objectFit: "cover", display: "block" }} onError={() => onError(`Cannot display tile ${candidate.tile_id}.`)} />
                  ) : (
                    <div style={{ width: "100%", aspectRatio: "1/1", background: "rgba(255,255,255,0.05)", display: "flex", alignItems: "center", justifyContent: "center", color: "rgba(255,255,255,0.2)" }}>
                      <span>No image preview</span>
                    </div>
                  )}
                  <div style={{ padding: "18px" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", gap: "12px" }}>
                      <strong>#{index + 1} · {candidate.tile_id}</strong>
                      <strong>{Number(candidate.score).toFixed(4)}</strong>
                    </div>
                    <div style={{ marginTop: "12px", fontSize: "12px", opacity: 0.68, lineHeight: 1.6 }}>
                      <div>Cosine similarity</div>
                      <div>BBOX: {candidate.bbox.map((v) => Number(v).toFixed(5)).join(", ")}</div>
                    </div>
                  </div>
                </article>
              ))}
            </div>
          )}
      </div>
    </section>
  );
}

function App() {
  const [query, setQuery] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);
  const [activeNav, setActiveNav] = useState("home");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [mapLoading, setMapLoading] = useState(false);
  const [mapError, setMapError] = useState("");
  const [mapUrl, setMapUrl] = useState("");
  const [mapType, setMapType] = useState("");
  const [mapStats, setMapStats] = useState(null);
  const [changeLoading, setChangeLoading] = useState(false);
  const [changeError, setChangeError] = useState("");
  const [changeResult, setChangeResult] = useState(null);
  const [searchResults, setSearchResults] = useState(null);
  const [searchLoading, setSearchLoading] = useState(false);
  const [showAdditional, setShowAdditional] = useState(false);

  useEffect(() => {
    const sections = ["home", "explore", "analysis", "about"];
    const handleScroll = () => {
      const pos = window.scrollY + 180;
      let current = "home";
      sections.forEach((id) => {
        const el = document.getElementById(id);
        if (el && pos >= el.offsetTop) current = id;
      });
      setActiveNav(current);
    };
    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  const scrollToSection = useCallback((id) => {
    setMenuOpen(false);
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  }, []);

  useEffect(() => {
    if (!result) return;
    const analysis = result.structured_query?.analysis;
    const geo = result.geographic_result;
    if (!geo?.latitude || !geo?.longitude) return;
    if (result.validation_status !== "passed") return;
    const startDate = result.structured_query?.start_date || defaultDateWindow().startDate;
    const endDate = result.structured_query?.end_date || defaultDateWindow().endDate;
    setMapUrl(""); setMapType(""); setMapStats(null); setMapError("");
    setChangeResult(null); setChangeError("");
    if (analysis === "ndvi") autoGenerateMap("ndvi", geo, startDate, endDate);
    else if (analysis === "ndwi" || analysis === "water_extent") autoGenerateMap("ndwi", geo, startDate, endDate);
    else if (analysis === "change" || analysis === "water_extent_change") autoRunChangeDetection(geo, startDate, endDate);
  }, [result]);

  const autoGenerateMap = async (type, geo, startDate, endDate) => {
    setMapLoading(true); setMapType(type); setMapError("");
    try {
      const fn = type === "ndvi" ? generateNDVIMap : generateNDWIMap;
      const data = await fn({ latitude: geo.latitude, longitude: geo.longitude, startDate, endDate });
      setMapUrl(data.map_url);
      setMapStats(data.statistics || null);
      setTimeout(() => document.getElementById("satellite-map")?.scrollIntoView({ behavior: "smooth", block: "center" }), 200);
    } catch (err) {
      console.error("Auto map error:", err);
      setMapError(err.message || "Map generation failed.");
    } finally {
      setMapLoading(false);
    }
  };

  const autoRunChangeDetection = async (geo, startDate, endDate) => {
    setChangeLoading(true); setChangeError("");
    try {
      const { firstStart, firstEnd, secondStart, secondEnd } = splitDateRange(startDate, endDate);
      const data = await runChangeDetection({ latitude: geo.latitude, longitude: geo.longitude, firstStartDate: firstStart, firstEndDate: firstEnd, secondStartDate: secondStart, secondEndDate: secondEnd });
      setChangeResult(data);
      setTimeout(() => document.getElementById("change-detection-result")?.scrollIntoView({ behavior: "smooth", block: "center" }), 200);
    } catch (err) {
      console.error("Auto change error:", err);
      setChangeError(err.message || "Change detection failed.");
    } finally {
      setChangeLoading(false);
    }
  };

  const handleManualMap = async (type) => {
    const geo = result?.geographic_result;
    if (!geo?.latitude || !geo?.longitude) { setError("No geographic location available."); return; }
    const startDate = result.structured_query?.start_date || defaultDateWindow().startDate;
    const endDate = result.structured_query?.end_date || defaultDateWindow().endDate;
    await autoGenerateMap(type, geo, startDate, endDate);
  };

  const handleManualChangeDetection = async () => {
    const geo = result?.geographic_result;
    if (!geo?.latitude || !geo?.longitude) { setError("No geographic location available."); return; }
    const startDate = result.structured_query?.start_date || defaultDateWindow().startDate;
    const endDate = result.structured_query?.end_date || defaultDateWindow().endDate;
    await autoRunChangeDetection(geo, startDate, endDate);
  };

  const handleQuery = async () => {
    if (!query.trim() || loading) return;
    setLoading(true); setError(""); setResult(null);
    setMapUrl(""); setMapType(""); setMapStats(null); setMapError("");
    setChangeResult(null); setChangeError(""); setSearchResults(null); setShowAdditional(false);
    try {
      const data = await runQuery(query);
      setResult(data);
      setTimeout(() => document.getElementById("geoquery-result")?.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
    } catch (err) {
      setError(err.message || "Unable to connect to the GeoQuery backend.");
    } finally {
      setLoading(false);
    }
  };

  const handleSemanticSearch = async () => {
    if (!query.trim() || searchLoading) return;
    setSearchLoading(true); setError(""); setSearchResults(null);
    try {
      const data = await semanticSearch({ visualQuery: query.trim(), location: result?.geographic_result?.name || null });
      const candidates = (data.candidates || []).map((c) => ({
        ...c,
        image_url: c.image_url?.startsWith("http")
          ? c.image_url
          : c.image_url
          ? `http://127.0.0.1:8000${c.image_url}`
          : `https://placehold.co/400x400/0a1f14/4ade80?text=TILE+${c.tile_id.split('_').pop()}`
      }));
      setSearchResults({ ...data, candidates });
      setTimeout(() => document.getElementById("semantic-search-results")?.scrollIntoView({ behavior: "smooth", block: "start" }), 100);
    } catch (err) {
      setError(err.message || "Semantic search failed.");
    } finally {
      setSearchLoading(false);
    }
  };

  const selectExample = (text) => { setQuery(text); scrollToSection("home"); };
  const selectAnalysis = (type) => {
    const m = { ndvi: "Analyze vegetation health around Bengaluru", ndwi: "Find lakes around Bengaluru", change: "Show vegetation changes around Bengaluru since last year" };
    setQuery(m[type] || "Analyze vegetation health around Bengaluru");
    scrollToSection("home");
  };

  const analysisType = result?.structured_query?.analysis || result?.analysis_result?.analysis || "unknown";
  const validationPassed = result?.validation_status === "passed";
  const showNDVI = validationPassed && analysisType === "ndvi";
  const showNDWI = validationPassed && (analysisType === "ndwi" || analysisType === "water_extent");
  const showWaterExtent = validationPassed && (analysisType === "water_extent" || analysisType === "water_extent_change");
  const showChange = validationPassed && (analysisType === "change" || analysisType === "water_extent_change");
  const showDiscovery = validationPassed && analysisType === "discovery";
  const hasGeoCoords = result?.geographic_result?.latitude != null && result?.geographic_result?.longitude != null;

  return (
    <div className="app">
      <header className="navbar">
        <div className="brand" onClick={() => scrollToSection("home")}>
          <div className="brand-symbol"><div className="brand-core"></div></div>
          <div className="brand-text">
            <div className="brand-name">Geo<span>Query</span></div>
            <div className="brand-subtitle">EXPLORE · ANALYZE · UNDERSTAND</div>
          </div>
        </div>
        <nav className={`nav-links ${menuOpen ? "open" : ""}`}>
          <button className={activeNav === "home" ? "nav-active" : ""} onClick={() => scrollToSection("home")}>Home</button>
          <button className={activeNav === "explore" ? "nav-active" : ""} onClick={() => scrollToSection("explore")}>Explore</button>
          <button className={activeNav === "analysis" ? "nav-active" : ""} onClick={() => scrollToSection("analysis")}>Analysis</button>
          <button className={activeNav === "about" ? "nav-active" : ""} onClick={() => scrollToSection("about")}>About</button>
        </nav>
        <button className="get-started" onClick={() => scrollToSection("home")}>Get Started <ArrowRight size={16} /></button>
        <button className="menu-button" onClick={() => setMenuOpen(!menuOpen)}>{menuOpen ? <X size={23} /> : <Menu size={23} />}</button>
      </header>

      <main id="home" className="hero">
        <div className="background-glow"></div>
        <section className="hero-content">
          <div className="eyebrow"><span className="eyebrow-line"></span>SATELLITE INSIGHTS · REAL IMPACT</div>
          <h1>Ask the Earth.<br /><span>See the Change.</span></h1>
          <p className="description">GeoQuery uses AI and satellite imagery to help you explore our planet — from vegetation and water bodies to environmental changes.</p>
          <div className="search-container">
            <Search size={20} className="search-icon" />
            <input type="text" value={query} placeholder="Ask GeoQuery — e.g. 'Find lakes around Bengaluru'" onChange={(e) => setQuery(e.target.value)} onKeyDown={(e) => { if (e.key === "Enter") handleQuery(); }} />
            <button className="search-button" onClick={handleQuery} disabled={loading || !query.trim()}>
              {loading ? <Loader2 size={20} className="spin" /> : <ArrowRight size={20} />}
            </button>
          </div>
          <button type="button" onClick={handleSemanticSearch} disabled={searchLoading || !query.trim()} style={{ marginTop: "12px", width: "100%", padding: "12px 18px", borderRadius: "12px", border: "1px solid rgba(120,180,140,0.35)", background: "rgba(20,50,35,0.65)", color: "#dff5e5", cursor: (searchLoading || !query.trim()) ? "not-allowed" : "pointer", opacity: (searchLoading || !query.trim()) ? 0.55 : 1, fontWeight: 600, letterSpacing: "0.02em", fontSize: "14px" }}>
            {searchLoading ? "Searching Satellite Tiles…" : "Search Satellite Tiles with RemoteCLIP"}
          </button>
          <div className="examples">
            <p>TRY AN EXAMPLE</p>
            <div className="example-list">
              {EXAMPLE_QUERIES.map((example, index) => (
                <button className="example" key={index} onClick={() => selectExample(example.text)}>
                  <Search size={14} /><span>{example.text}</span>
                </button>
              ))}
            </div>
          </div>
        </section>
        <section className="earth-area">
          <div className="orbit orbit-1"></div><div className="orbit orbit-2"></div>
          <div className="earth-halo"></div>
          <div className="earth"><Globe /></div>
          <div className="data-point point-1"></div><div className="data-point point-2"></div><div className="data-point point-3"></div>
        </section>
        <div className="side-text"><div className="side-line"></div><p>A<br />CLEANER<br />GREENER<br />BRIGHTER<br />TOMORROW</p></div>
        <div className="stats">
          <div className="stat"><strong>AI</strong><span>GEOSPATIAL INTELLIGENCE</span></div>
          <div className="divider"></div>
          <div className="stat"><strong>NDVI</strong><span>VEGETATION ANALYSIS</span></div>
          <div className="divider"></div>
          <div className="stat"><strong>NDWI</strong><span>WATER ANALYSIS</span></div>
        </div>
        <div className="scroll" onClick={() => scrollToSection("explore")}>
          <div className="scroll-circle">↓</div>
          <span>SCROLL<br />TO EXPLORE</span>
        </div>
      </main>

      {error && (
        <div className="query-error">
          <AlertCircle size={18} /><span>{error}</span>
          <button onClick={() => setError("")}><X size={16} /></button>
        </div>
      )}

      {loading && (
        <div className="global-loading">
          <Loader2 size={32} className="spin" />
          <div>
            <strong>Analyzing your query…</strong>
            <p>Running the AI pipeline: parsing → geocoding → satellite retrieval → analysis</p>
          </div>
        </div>
      )}

      <SemanticSearchResults searchResults={searchResults} onError={setError} />

      {result && (
        <section id="geoquery-result" className="result-section">
          <div className="result-header">
            <div><div className="section-label">GEOQUERY RESULT</div><h2>Satellite Intelligence</h2></div>
            <div className={`validation-badge ${validationPassed ? "passed" : "rejected"}`}>
              {validationPassed ? <CheckCircle2 size={15} /> : <AlertCircle size={15} />}
              {result.validation_status || "unknown"}
            </div>
          </div>
          <div className="original-query"><span>YOUR QUERY</span><p>"{result.query}"</p></div>

          <QueryInterpretationCard result={result} />

          {(mapLoading || mapError) && <AutoMapStatus loading={mapLoading} error={mapError} mapType={mapType} />}
          {(changeLoading || changeError) && <AutoMapStatus loading={changeLoading} error={changeError} mapType="change-detection" />}

          {showNDVI && <NDVIResults analysisResult={result.analysis_result} />}
          {showNDWI && <NDWIResults analysisResult={result.analysis_result} label={analysisType === "water_extent" ? "Water Extent (NDWI)" : "Water Bodies (NDWI)"} />}
          {showWaterExtent && <WaterExtentResults metrics={result.analysis_result?.metrics} />}

          {showDiscovery && result.analysis_result?.metrics && (
            <div className="analysis-result-block discovery-block">
              <div className="arb-header">
                <Satellite size={18} color="#e2e8f0" />
                <div><h3>Scene Discovery</h3><p>Sentinel-2 satellite scene found for the requested area</p></div>
              </div>
              <MetricGrid metrics={result.analysis_result.metrics} />
            </div>
          )}

          {result.analysis_result?.status === "no_data" && (
            <div className="no-data-block">
              <AlertCircle size={18} />
              <div><strong>No satellite data available</strong><p>{result.analysis_result.message || "No suitable scene found for the requested area and time period."}</p></div>
            </div>
          )}
          {result.analysis_result?.status === "error" && (
            <div className="no-data-block error">
              <AlertCircle size={18} />
              <div><strong>Analysis error</strong><p>{result.analysis_result.message}</p></div>
            </div>
          )}

          <SatelliteMetadata scene={result.analysis_result?.scene} />
          <MapDisplay mapUrl={mapUrl} mapType={mapType} mapStats={mapStats} />

          {showChange && (
            <>
              {changeLoading && <div className="map-loading"><span className="loading-spinner"></span>Running change detection analysis…</div>}
              {changeResult && <ChangeDetectionResults changeData={changeResult} />}
            </>
          )}

          <ExplanationPanel explanation={result.explanation} />

          {hasGeoCoords && (
            <div className="additional-analysis">
              <button className="additional-toggle" onClick={() => setShowAdditional(!showAdditional)}>
                <Info size={15} /><span>Additional Analysis</span>{showAdditional ? <ChevronUp size={15} /> : <ChevronDown size={15} />}
              </button>
              {showAdditional && (
                <div className="additional-panel">
                  <p className="additional-note">Run additional spectral analyses beyond the primary query intent.</p>
                  <div className="map-buttons">
                    <button onClick={() => handleManualMap("ndvi")} disabled={mapLoading}><Leaf size={16} />{mapLoading && mapType === "ndvi" ? "Generating…" : "Generate NDVI Map"}</button>
                    <button onClick={() => handleManualMap("ndwi")} disabled={mapLoading}><Droplets size={16} />{mapLoading && mapType === "ndwi" ? "Generating…" : "Generate NDWI Map"}</button>
                    <button onClick={handleManualChangeDetection} disabled={changeLoading}><Activity size={16} />{changeLoading ? "Analyzing Change…" : "Detect Change"}</button>
                  </div>
                </div>
              )}
            </div>
          )}

          <details className="technical-details">
            <summary><span>VIEW PIPELINE DETAILS</span><ArrowRight size={14} /></summary>
            <div className="technical-grid">
              <div><span>01 · QUERY</span><strong>{result.query || "—"}</strong></div>
              <div><span>QUERY ID</span><strong>{result.query_id || "—"}</strong></div>
              <div><span>02 · VISUAL QUERY</span><strong>{result.structured_query?.visual_query || "—"}</strong></div>
              <div><span>ANALYSIS TYPE</span><strong>{formatLabel(analysisType)}</strong></div>
              <div><span>03 · LOCATION</span><strong>{result.geographic_result?.name || "—"}</strong></div>
              <div><span>COORDINATES</span><strong>{result.geographic_result?.latitude != null ? `${formatValue(result.geographic_result.latitude)}, ${formatValue(result.geographic_result.longitude)}` : "—"}</strong></div>
              <div><span>04 · VALIDATION</span><strong className={validationPassed ? "pipeline-success" : ""}>{result.validation_status || "—"}</strong></div>
              <div><span>VALIDATION REASON</span><strong>{result.validation_reason || "—"}</strong></div>
              <div><span>05 · SATELLITE SCENE</span><strong className="scene-id">{result.analysis_result?.scene?.scene_id || "—"}</strong></div>
              <div><span>ACQUIRED</span><strong>{result.analysis_result?.scene?.datetime ? new Date(result.analysis_result.scene.datetime).toLocaleString() : "—"}</strong></div>
              <div><span>CLOUD COVER</span><strong>{result.analysis_result?.scene?.cloud_cover != null ? `${result.analysis_result.scene.cloud_cover}%` : "—"}</strong></div>
              <div><span>VALID PIXELS</span><strong>{result.analysis_result?.scene?.valid_pixel_fraction != null ? `${(result.analysis_result.scene.valid_pixel_fraction * 100).toFixed(1)}%` : "—"}</strong></div>
              <div><span>06 · SPECTRAL ANALYSIS</span><strong>{formatLabel(analysisType)}</strong></div>
              {getScalarMetrics(result.analysis_result?.metrics).map(([key, value]) => (
                <div key={`pipeline-${key}`}><span>{formatLabel(key)}</span><strong>{formatValue(value)}</strong></div>
              ))}
              <div><span>07 · MAP</span><strong className={mapUrl ? "pipeline-success" : ""}>{mapUrl ? "GENERATED" : mapLoading ? "GENERATING…" : "—"}</strong></div>
              <div><span>08 · CHANGE DETECTION</span><strong className={changeResult?.result ? "pipeline-success" : ""}>{changeResult?.result ? "COMPLETE" : changeLoading ? "RUNNING…" : "—"}</strong></div>
              <div className="pipeline-source"><span>DATA SOURCE</span><strong>{result.evidence?.source || "Sentinel-2 L2A via Planetary Computer STAC"}</strong></div>
            </div>
          </details>
        </section>
      )}

      <section id="explore" className="product-section explore-section">
        <div className="section-inner">
          <div className="section-label">01 · EXPLORE</div>
          <div className="section-heading-row">
            <div><h2>Explore the Earth.</h2><p className="section-intro">Start with a question. GeoQuery translates natural language into a geographic and satellite analysis.</p></div>
            <div className="section-number">01</div>
          </div>
          <div className="explore-grid">
            <div className="explore-card large">
              <div className="explore-icon"><Sparkles size={22} /></div>
              <span className="card-index">01</span>
              <h3>Ask naturally.</h3>
              <p>Describe what you want to find using ordinary language. No complex GIS commands are required.</p>
              <button onClick={() => selectExample("Analyze vegetation health around Bengaluru")}>Try a query <ArrowRight size={15} /></button>
            </div>
            <div className="explore-card">
              <div className="explore-icon"><MapPin size={22} /></div>
              <span className="card-index">02</span>
              <h3>Locate precisely.</h3>
              <p>GeoQuery grounds the request to a geographic location before analysis.</p>
            </div>
            <div className="explore-card">
              <div className="explore-icon"><Satellite size={22} /></div>
              <span className="card-index">03</span>
              <h3>Discover satellite scenes.</h3>
              <p>Relevant satellite observations become the evidence behind the analysis.</p>
            </div>
          </div>
          <div className="explore-strip">
            <div><Globe2 size={18} /><span>GEOGRAPHIC GROUNDING</span></div>
            <div><Brain size={18} /><span>AI QUERY UNDERSTANDING</span></div>
            <div><Database size={18} /><span>SATELLITE EVIDENCE</span></div>
            <div><Target size={18} /><span>ANALYSIS READY</span></div>
          </div>
        </div>
      </section>

      <section id="analysis" className="product-section analysis-section">
        <div className="section-inner">
          <div className="section-label">02 · ANALYSIS</div>
          <div className="section-heading-row">
            <div><h2>Understand the Change.</h2><p className="section-intro">Turn satellite observations into measurable environmental signals.</p></div>
            <div className="section-number">02</div>
          </div>
          <div className="analysis-grid">
            <div className="analysis-card">
              <div className="analysis-card-top"><div className="analysis-icon"><Leaf size={23} /></div><span>SPECTRAL INDEX</span></div>
              <h3>NDVI</h3>
              <p>Measure vegetation health and identify areas where plant cover is strong, weak, or changing.</p>
              <div className="analysis-meta"><span>VEGETATION</span><span>−1 → +1</span></div>
              <button onClick={() => selectAnalysis("ndvi")}>Explore NDVI <ArrowRight size={15} /></button>
            </div>
            <div className="analysis-card">
              <div className="analysis-card-top"><div className="analysis-icon"><Droplets size={23} /></div><span>SPECTRAL INDEX</span></div>
              <h3>NDWI</h3>
              <p>Highlight water and moisture patterns using spectral information from satellite imagery.</p>
              <div className="analysis-meta"><span>WATER</span><span>−1 → +1</span></div>
              <button onClick={() => selectAnalysis("ndwi")}>Explore NDWI <ArrowRight size={15} /></button>
            </div>
            <div className="analysis-card featured">
              <div className="analysis-card-top"><div className="analysis-icon"><Activity size={23} /></div><span>TEMPORAL ANALYSIS</span></div>
              <h3>Change Detection</h3>
              <p>Compare two satellite observation periods and quantify how vegetation signals changed over time.</p>
              <div className="analysis-meta"><span>TIME SERIES</span><span>Δ NDVI</span></div>
              <button onClick={() => selectAnalysis("change")}>Detect Change <ArrowRight size={15} /></button>
            </div>
          </div>
          <div className="analysis-flow">
            <div className="flow-title">ANALYSIS PIPELINE</div>
            <div className="flow-steps">
              <div className="flow-step"><span>01</span><Search size={17} /><strong>Query</strong></div>
              <div className="flow-line"></div>
              <div className="flow-step"><span>02</span><MapPin size={17} /><strong>Location</strong></div>
              <div className="flow-line"></div>
              <div className="flow-step"><span>03</span><Satellite size={17} /><strong>Satellite</strong></div>
              <div className="flow-line"></div>
              <div className="flow-step"><span>04</span><Layers3 size={17} /><strong>Analysis</strong></div>
              <div className="flow-line"></div>
              <div className="flow-step"><span>05</span><Eye size={17} /><strong>Insight</strong></div>
            </div>
          </div>
        </div>
      </section>

      <section id="about" className="product-section about-section">
        <div className="section-inner">
          <div className="section-label">03 · ABOUT GEOQUERY</div>
          <div className="about-main">
            <div className="about-copy">
              <h2>Intelligence<br />from above.</h2>
              <p>GeoQuery is designed to make geospatial intelligence easier to access. Instead of navigating complicated satellite-data workflows, users can ask questions in natural language and receive measurable satellite-derived insights.</p>
              <button className="about-action" onClick={() => scrollToSection("home")}>Ask GeoQuery <ArrowRight size={16} /></button>
            </div>
            <div className="about-visual">
              <div className="about-orbit"></div>
              <div className="about-core"><Globe2 size={34} /><span>GEO</span><strong>QUERY</strong></div>
              <div className="about-node node-a">AI</div>
              <div className="about-node node-b">GIS</div>
              <div className="about-node node-c">EO</div>
            </div>
          </div>
          <div className="technology-section">
            <div className="technology-heading"><span>UNDER THE HOOD</span><h3>From question to satellite insight.</h3></div>
            <div className="technology-grid">
              <div className="technology-item"><Brain size={19} /><div><span>AI</span><strong>Natural Language Understanding</strong><p>Converts user questions into structured analysis intent.</p></div></div>
              <div className="technology-item"><MapPin size={19} /><div><span>GEO</span><strong>Geographic Grounding</strong><p>Connects a requested place with usable coordinates.</p></div></div>
              <div className="technology-item"><Satellite size={19} /><div><span>EO</span><strong>Earth Observation</strong><p>Uses satellite observations as evidence for analysis.</p></div></div>
              <div className="technology-item"><BarChart3 size={19} /><div><span>DATA</span><strong>Quantitative Analysis</strong><p>Produces measurable metrics, maps, and change indicators.</p></div></div>
            </div>
          </div>
          <div className="about-footer">
            <span>GEOQUERY</span><span>EXPLORE · ANALYZE · UNDERSTAND</span><span>SATELLITE INTELLIGENCE</span>
          </div>
        </div>
      </section>

    </div>
  );
}

export default App;
