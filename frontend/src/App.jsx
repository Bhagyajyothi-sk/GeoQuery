import { useEffect, useState } from "react";
import Globe from "./Globe";

import {
  ArrowRight,
  Menu,
  X,
  Leaf,
  Droplets,
  Activity,
  MapPin,
  Satellite,
  BarChart3,
  CheckCircle2,
  AlertCircle,
  Search,
  Sparkles,
  Database,
  Brain,
  Layers3,
  Globe2,
  Target,
  Eye,
} from "lucide-react";

const API_BASE = "http://127.0.0.1:8000";

function App() {
  const [query, setQuery] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);
  const [activeNav, setActiveNav] = useState("home");

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [mapLoading, setMapLoading] = useState(false);
  const [mapUrl, setMapUrl] = useState("");
  const [mapType, setMapType] = useState("");

  const [changeLoading, setChangeLoading] = useState(false);
  const [changeResult, setChangeResult] = useState(null);

  const [searchResults, setSearchResults] = useState(null);
  const [searchLoading, setSearchLoading] = useState(false);

  const examples = [
    {
      icon: <Leaf size={14} />,
      text: "Vegetation loss near Bengaluru",
    },
    {
      icon: <Droplets size={14} />,
      text: "Water bodies in Karnataka",
    },
    {
      icon: <Activity size={14} />,
      text: "What changed in this area?",
    },
  ];

  /*
   * ---------------------------------------------------------
   * NAVIGATION OBSERVER
   * ---------------------------------------------------------
   */

  useEffect(() => {
    const sections = ["home", "explore", "analysis", "about"];

    const handleScroll = () => {
      const scrollPosition = window.scrollY + 180;

      let current = "home";

      sections.forEach((id) => {
        const element = document.getElementById(id);

        if (element && scrollPosition >= element.offsetTop) {
          current = id;
        }
      });

      setActiveNav(current);
    };

    window.addEventListener("scroll", handleScroll);

    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, []);

  /*
   * ---------------------------------------------------------
   * SMOOTH NAVIGATION
   * ---------------------------------------------------------
   */

  const scrollToSection = (id) => {
    setMenuOpen(false);

    document.getElementById(id)?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  };

  /*
   * ---------------------------------------------------------
   * MAIN GEOQUERY REQUEST
   * ---------------------------------------------------------
   */

  const handleSemanticSearch = async () => {
    if (!query.trim() || searchLoading) return;

    setSearchLoading(true);
    setError("");
    setSearchResults(null);

    try {
      const response = await fetch(`${API_BASE}/api/search`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          visual_query: query.trim(),
          location: result?.geographic_result?.name || null,
          top_k: 5,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data?.detail || "Semantic satellite search failed.");
      }

      const candidates = (data.candidates || []).map((candidate) => ({
        ...candidate,
        image_url: candidate.image_url?.startsWith("http")
          ? candidate.image_url
          : `${API_BASE}${candidate.image_url || ""}`,
      }));

      setSearchResults({ ...data, candidates });

      setTimeout(() => {
        document.getElementById("semantic-search-results")?.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }, 100);
    } catch (err) {
      console.error("Semantic search error:", err);
      setError(err.message || "Unable to run semantic satellite search.");
    } finally {
      setSearchLoading(false);
    }
  };

  const handleQuery = async () => {
    if (!query.trim() || loading) return;

    setLoading(true);
    setError("");
    setMapUrl("");
    setMapType("");
    setChangeResult(null);

    try {
      console.log("GeoQuery:", query);

      const response = await fetch(`${API_BASE}/api/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: query.trim(),
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data?.detail || "GeoQuery backend returned an error."
        );
      }

      console.log("Backend response:", data);

      setResult(data);

      setTimeout(() => {
        document
          .getElementById("geoquery-result")
          ?.scrollIntoView({
            behavior: "smooth",
            block: "start",
          });
      }, 100);
    } catch (err) {
      console.error("Backend connection error:", err);

      setError(
        err.message ||
          "Unable to connect to the GeoQuery backend."
      );
    } finally {
      setLoading(false);
    }
  };

  /*
   * ---------------------------------------------------------
   * MAP REQUEST
   * ---------------------------------------------------------
   */

  const handleMapRequest = async (type) => {
    if (!result?.geographic_result) {
      setError(
        "No geographic location is available for this result."
      );
      return;
    }

    const geo = result.geographic_result;

    if (
      geo.latitude === null ||
      geo.longitude === null ||
      geo.latitude === undefined ||
      geo.longitude === undefined
    ) {
      setError(
        "This result does not contain valid coordinates."
      );
      return;
    }

    setMapLoading(true);
    setError("");
    setMapUrl("");
    setMapType(type);

    const startDate =
      result.structured_query?.start_date ||
      "2025-01-01";

    const endDate =
      result.structured_query?.end_date ||
      new Date().toISOString().slice(0, 10);

    try {
      let endpoint = "";
      let body = {};

      if (type === "ndvi") {
        endpoint = "/api/maps/ndvi";

        body = {
          latitude: geo.latitude,
          longitude: geo.longitude,
          start_date: startDate,
          end_date: endDate,
        };
      }

      if (type === "ndwi") {
        endpoint = "/api/maps/ndwi";

        body = {
          latitude: geo.latitude,
          longitude: geo.longitude,
          start_date: startDate,
          end_date: endDate,
        };
      }

      if (!endpoint) {
        throw new Error("Unknown map analysis type.");
      }

      const response = await fetch(
        `${API_BASE}${endpoint}`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify(body),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data?.detail ||
            "Map rendering failed."
        );
      }

      if (!data.map_url) {
        throw new Error(
          "The backend completed the request but did not return a map URL."
        );
      }

      const finalMapUrl = data.map_url.startsWith("http")
        ? data.map_url
        : `${API_BASE}${data.map_url}`;

      setMapUrl(finalMapUrl);

      setTimeout(() => {
        document
          .getElementById("satellite-map")
          ?.scrollIntoView({
            behavior: "smooth",
            block: "center",
          });
      }, 100);
    } catch (err) {
      console.error("Map request error:", err);

      setError(
        err.message ||
          "Unable to generate the satellite map."
      );
    } finally {
      setMapLoading(false);
    }
  };

  /*
   * ---------------------------------------------------------
   * CHANGE DETECTION
   * ---------------------------------------------------------
   */

  const handleChangeDetection = async () => {
    if (!result?.geographic_result) {
      setError(
        "No geographic location is available for this result."
      );
      return;
    }

    const geo = result.geographic_result;

    if (
      geo.latitude === null ||
      geo.longitude === null ||
      geo.latitude === undefined ||
      geo.longitude === undefined
    ) {
      setError(
        "This result does not contain valid coordinates."
      );
      return;
    }

    setChangeLoading(true);
    setError("");
    setChangeResult(null);

    const startDate =
      result.structured_query?.start_date ||
      "2025-01-01";

    const endDate =
      result.structured_query?.end_date ||
      new Date().toISOString().slice(0, 10);

    try {
      const start = new Date(startDate);
      const end = new Date(endDate);

      const midpoint = new Date(
        start.getTime() +
          (end.getTime() - start.getTime()) / 2
      );

      const firstEnd = midpoint
        .toISOString()
        .slice(0, 10);

      const secondStart = new Date(
        midpoint.getTime() +
          24 * 60 * 60 * 1000
      )
        .toISOString()
        .slice(0, 10);

      const response = await fetch(
        `${API_BASE}/api/maps/change-detection`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            latitude: geo.latitude,
            longitude: geo.longitude,
            first_start_date: startDate,
            first_end_date: firstEnd,
            second_start_date: secondStart,
            second_end_date: endDate,
            max_cloud_cover: 20,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data?.detail ||
            "Change detection failed."
        );
      }

      console.log(
        "CHANGE DETECTION RESPONSE:",
        JSON.stringify(data, null, 2)
      );

      setChangeResult(data);

      setTimeout(() => {
        document
          .getElementById("change-detection-result")
          ?.scrollIntoView({
            behavior: "smooth",
            block: "center",
          });
      }, 100);
    } catch (err) {
      console.error(
        "Change detection error:",
        err
      );

      setError(
        err.message ||
          "Unable to run change detection."
      );
    } finally {
      setChangeLoading(false);
    }
  };

  /*
   * ---------------------------------------------------------
   * HELPERS
   * ---------------------------------------------------------
   */

  const formatLabel = (key) => {
    if (!key) return "Unknown";

    return String(key)
      .replaceAll("_", " ")
      .replace(/\b\w/g, (letter) =>
        letter.toUpperCase()
      );
  };

  const formatValue = (value) => {
    if (value === null || value === undefined) {
      return "—";
    }

    if (typeof value === "number") {
      return Number.isInteger(value)
        ? value.toLocaleString()
        : value.toLocaleString(undefined, {
            maximumFractionDigits: 6,
          });
    }

    return String(value);
  };

  const getMetrics = () => {
    const metrics =
      result?.analysis_result?.metrics;

    if (!metrics) return [];

    return Object.entries(metrics).filter(
      ([, value]) =>
        typeof value !== "object" ||
        value === null
    );
  };

  const getChangeMetrics = () => {
    return (
      result?.analysis_result?.metrics
        ?.change_metrics || {}
    );
  };

  const analysisType =
    result?.structured_query?.analysis ||
    result?.analysis_result?.analysis ||
    "unknown";

  const validationPassed =
    result?.validation_status === "passed";

  const selectExample = (text) => {
    setQuery(text);
    scrollToSection("home");
  };

  const selectAnalysis = (type) => {
    if (type === "ndvi") {
      setQuery("Vegetation health near Bengaluru");
    }

    if (type === "ndwi") {
      setQuery("Water bodies in Karnataka");
    }

    if (type === "change") {
      setQuery("Vegetation loss near Bengaluru");
    }

    scrollToSection("home");
  };

  /*
   * ---------------------------------------------------------
   * UI
   * ---------------------------------------------------------
   */

  return (
    <div className="app">

      {/* =====================================================
          NAVBAR
      ===================================================== */}

      <header className="navbar">

        <div
          className="brand"
          onClick={() => scrollToSection("home")}
        >
          <div className="brand-symbol">
            <div className="brand-core"></div>
          </div>

          <div className="brand-text">
            <div className="brand-name">
              Geo<span>Query</span>
            </div>

            <div className="brand-subtitle">
              EXPLORE · ANALYZE · UNDERSTAND
            </div>
          </div>
        </div>

        <nav
          className={`nav-links ${
            menuOpen ? "open" : ""
          }`}
        >

          <button
            className={
              activeNav === "home"
                ? "nav-active"
                : ""
            }
            onClick={() =>
              scrollToSection("home")
            }
          >
            Home
          </button>

          <button
            className={
              activeNav === "explore"
                ? "nav-active"
                : ""
            }
            onClick={() =>
              scrollToSection("explore")
            }
          >
            Explore
          </button>

          <button
            className={
              activeNav === "analysis"
                ? "nav-active"
                : ""
            }
            onClick={() =>
              scrollToSection("analysis")
            }
          >
            Analysis
          </button>

          <button
            className={
              activeNav === "about"
                ? "nav-active"
                : ""
            }
            onClick={() =>
              scrollToSection("about")
            }
          >
            About
          </button>

        </nav>

        <button
          className="get-started"
          onClick={() =>
            scrollToSection("home")
          }
        >
          Get Started
          <ArrowRight size={16} />
        </button>

        <button
          className="menu-button"
          onClick={() =>
            setMenuOpen(!menuOpen)
          }
        >
          {menuOpen ? (
            <X size={23} />
          ) : (
            <Menu size={23} />
          )}
        </button>

      </header>

      {/* =====================================================
          HERO / HOME
      ===================================================== */}

      <main
        id="home"
        className="hero"
      >

        <div className="background-glow"></div>

        <section className="hero-content">

          <div className="eyebrow">
            <span className="eyebrow-line"></span>
            SATELLITE INSIGHTS · REAL IMPACT
          </div>

          <h1>
            Ask the Earth.
            <br />
            <span>See the Change.</span>
          </h1>

          <p className="description">
            GeoQuery uses AI and satellite imagery
            to help you explore our planet — from
            vegetation and water bodies to
            environmental changes.
          </p>

          {/* SEARCH */}

          <div className="search-container">

            <Search
              size={20}
              className="search-icon"
            />

            <input
              type="text"
              value={query}
              placeholder="Ask GeoQuery..."
              onChange={(e) =>
                setQuery(e.target.value)
              }
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  handleQuery();
                }
              }}
            />

            <button
              className="search-button"
              onClick={handleQuery}
              disabled={loading || searchLoading}
              title="Run full GeoQuery analysis"
            >
              {loading ? (
                <span className="loading-dot">
                  ...
                </span>
              ) : (
                <ArrowRight size={20} />
              )}
            </button>

          </div>

          <button
            type="button"
            onClick={handleSemanticSearch}
            disabled={searchLoading || !query.trim()}
            style={{
              marginTop: "12px",
              width: "100%",
              padding: "12px 18px",
              borderRadius: "12px",
              border: "1px solid rgba(120, 180, 140, 0.35)",
              background: "rgba(20, 50, 35, 0.65)",
              color: "#dff5e5",
              cursor: searchLoading || !query.trim() ? "not-allowed" : "pointer",
              opacity: searchLoading || !query.trim() ? 0.55 : 1,
              fontWeight: 600,
              letterSpacing: "0.02em",
            }}
          >
            {searchLoading ? "Searching Satellite Tiles..." : "Search Satellite Tiles with RemoteCLIP"}
          </button>

          {/* EXAMPLES */}

          <div className="examples">

            <p>TRY AN EXAMPLE</p>

            <div className="example-list">

              {examples.map(
                (example, index) => (
                  <button
                    className="example"
                    key={index}
                    onClick={() =>
                      selectExample(
                        example.text
                      )
                    }
                  >
                    {example.icon}

                    <span>
                      {example.text}
                    </span>
                  </button>
                )
              )}

            </div>

          </div>

        </section>

        {/* EARTH */}

        <section className="earth-area">

          <div className="orbit orbit-1"></div>
          <div className="orbit orbit-2"></div>

          <div className="earth-halo"></div>

          <div className="earth">
            <Globe />
          </div>

          <div className="data-point point-1"></div>
          <div className="data-point point-2"></div>
          <div className="data-point point-3"></div>

        </section>

        {/* SIDE TEXT */}

        <div className="side-text">

          <div className="side-line"></div>

          <p>
            A<br />
            CLEANER<br />
            GREENER<br />
            BRIGHTER<br />
            TOMORROW
          </p>

        </div>

        {/* STATS */}

        <div className="stats">

          <div className="stat">
            <strong>AI</strong>
            <span>GEOSPATIAL INTELLIGENCE</span>
          </div>

          <div className="divider"></div>

          <div className="stat">
            <strong>NDVI</strong>
            <span>VEGETATION ANALYSIS</span>
          </div>

          <div className="divider"></div>

          <div className="stat">
            <strong>NDWI</strong>
            <span>WATER ANALYSIS</span>
          </div>

        </div>

        <div
          className="scroll"
          onClick={() =>
            scrollToSection("explore")
          }
        >
          <div className="scroll-circle">
            ↓
          </div>

          <span>
            SCROLL
            <br />
            TO EXPLORE
          </span>
        </div>

      </main>

      {/* =====================================================
          ERROR
      ===================================================== */}

      {error && (
        <div className="query-error">

          <AlertCircle size={18} />

          <span>{error}</span>

          <button
            onClick={() => setError("")}
          >
            <X size={16} />
          </button>

        </div>
      )}

      {searchResults && (
        <section
          id="semantic-search-results"
          style={{
            padding: "70px 6vw",
            background: "#07120d",
            color: "#eef8f1",
          }}
        >
          <div style={{ maxWidth: "1200px", margin: "0 auto" }}>
            <div className="section-label">REMOTECLIP · FAISS SEARCH</div>
            <div style={{ display: "flex", justifyContent: "space-between", gap: "20px", alignItems: "end", flexWrap: "wrap" }}>
              <div>
                <h2 style={{ marginBottom: "8px" }}>Semantic Satellite Results</h2>
                <p style={{ opacity: 0.72, margin: 0 }}>
                  “{searchResults.visual_query}” · {searchResults.candidates.length} candidate tile(s)
                </p>
              </div>
              <div style={{ opacity: 0.6, fontSize: "12px" }}>QUERY ID · {searchResults.query_id}</div>
            </div>

            {searchResults.candidates.length === 0 ? (
              <p style={{ marginTop: "30px", opacity: 0.7 }}>No matching satellite tiles found.</p>
            ) : (
              <div style={{
                display: "grid",
                gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))",
                gap: "20px",
                marginTop: "30px",
              }}>
                {searchResults.candidates.map((candidate, index) => (
                  <article
                    key={candidate.tile_id}
                    style={{
                      overflow: "hidden",
                      borderRadius: "18px",
                      border: "1px solid rgba(150, 210, 170, 0.18)",
                      background: "rgba(255,255,255,0.035)",
                    }}
                  >
                    {candidate.image_url && (
                      <img
                        src={candidate.image_url}
                        alt={`Satellite tile ${candidate.tile_id}`}
                        style={{ width: "100%", aspectRatio: "1 / 1", objectFit: "cover", display: "block" }}
                        onError={() => setError(`Could not display satellite tile ${candidate.tile_id}.`)}
                      />
                    )}
                    <div style={{ padding: "18px" }}>
                      <div style={{ display: "flex", justifyContent: "space-between", gap: "12px" }}>
                        <strong>#{index + 1} · {candidate.tile_id}</strong>
                        <strong>{Number(candidate.score).toFixed(4)}</strong>
                      </div>
                      <div style={{ marginTop: "12px", fontSize: "12px", opacity: 0.68, lineHeight: 1.6 }}>
                        <div>Cosine similarity</div>
                        <div>BBOX: {candidate.bbox.map((value) => Number(value).toFixed(5)).join(", ")}</div>
                      </div>
                    </div>
                  </article>
                ))}
              </div>
            )}
          </div>
        </section>
      )}

      {/* =====================================================
          GEOQUERY RESULT
      ===================================================== */}

      {result && (

        <section
          id="geoquery-result"
          className="result-section"
        >

          <div className="result-header">

            <div>
              <div className="section-label">
                GEOQUERY RESULT
              </div>

              <h2>
                Satellite Intelligence
              </h2>
            </div>

            <div
              className={`validation-badge ${
                validationPassed
                  ? "passed"
                  : "rejected"
              }`}
            >

              {validationPassed ? (
                <CheckCircle2 size={15} />
              ) : (
                <AlertCircle size={15} />
              )}

              {result.validation_status ||
                "unknown"}

            </div>

          </div>

          {/* ORIGINAL QUERY */}

          <div className="original-query">

            <span>YOUR QUERY</span>

            <p>
              “{result.query}”
            </p>

          </div>

          {/* LOCATION / ANALYSIS / SCENE */}

          <div className="result-info">

            <div>
              <span>
                <MapPin size={12} />
                LOCATION
              </span>

              <strong>
                {result.geographic_result
                  ?.name ||
                  "Not specified"}
              </strong>
            </div>

            <div>
              <span>
                <BarChart3 size={12} />
                ANALYSIS
              </span>

              <strong>
                {formatLabel(
                  analysisType
                )}
              </strong>
            </div>

            <div>
              <span>
                <Satellite size={12} />
                SCENE
              </span>

              <strong>
                {result.analysis_result
                  ?.scene?.scene_id ||
                  "Not available"}
              </strong>
            </div>

          </div>

          {/* METRICS */}

          {getMetrics().length > 0 && (

            <div className="metrics">

              {getMetrics().map(
                ([key, value]) => (

                  <div
                    className="metric"
                    key={key}
                  >

                    <span>
                      {formatLabel(key)}
                    </span>

                    <strong>
                      {formatValue(value)}
                    </strong>

                  </div>

                )
              )}

            </div>

          )}

          {/* CHANGE METRICS */}

          {Object.keys(
            getChangeMetrics()
          ).length > 0 && (

            <div className="change-block">

              <div className="change-title">
                CHANGE ANALYSIS
              </div>

              <div className="change-grid">

                {Object.entries(
                  getChangeMetrics()
                ).map(
                  ([key, value]) => (

                    <div
                      className="change-item"
                      key={key}
                    >

                      <span>
                        {formatLabel(key)}
                      </span>

                      <strong>
                        {formatValue(value)}
                      </strong>

                    </div>

                  )
                )}

              </div>

            </div>

          )}

          {/* SATELLITE OBSERVATION */}

          {result.analysis_result?.scene && (

            <div className="scene-card">

              <div className="scene-card-title">
                SATELLITE OBSERVATION
              </div>

              <div className="scene-grid">

                <div>
                  <span>SCENE ID</span>

                  <strong>
                    {
                      result.analysis_result
                        .scene.scene_id
                    }
                  </strong>
                </div>

                <div>
                  <span>ACQUIRED</span>

                  <strong>
                    {
                      result.analysis_result
                        .scene.datetime
                      ? new Date(
                          result.analysis_result
                            .scene.datetime
                        ).toLocaleString()
                      : "—"
                    }
                  </strong>
                </div>

                <div>
                  <span>CLOUD COVER</span>

                  <strong>
                    {
                      result.analysis_result
                        .scene.cloud_cover !==
                        null &&
                      result.analysis_result
                        .scene.cloud_cover !==
                        undefined
                        ? `${result.analysis_result.scene.cloud_cover}%`
                        : "—"
                    }
                  </strong>
                </div>

                <div>
                  <span>VALID PIXELS</span>

                  <strong>
                    {
                      result.analysis_result
                        .scene
                        .valid_pixel_fraction !==
                        null &&
                      result.analysis_result
                        .scene
                        .valid_pixel_fraction !==
                        undefined
                        ? `${(
                            result.analysis_result
                              .scene
                              .valid_pixel_fraction *
                            100
                          ).toFixed(1)}%`
                        : "—"
                    }
                  </strong>
                </div>

              </div>

            </div>

          )}

          {/* EXPLANATION */}

          {result.explanation && (

            <div className="explanation">

              <div className="explanation-label">
                AI-GROUNDED EXPLANATION
              </div>

              <p>
                {result.explanation}
              </p>

            </div>

          )}

          {/* =================================================
              VISUAL ANALYSIS
          ================================================= */}

          {result.geographic_result?.latitude !==
            null &&
            result.geographic_result?.longitude !==
              null && (

              <div className="map-tools">

                <div className="map-tools-header">

                  <div>
                    <div className="section-label">
                      VISUAL ANALYSIS
                    </div>

                    <h3>
                      Explore the Satellite Data
                    </h3>

                    <p>
                      Generate spectral maps or
                      compare satellite observations
                      across time.
                    </p>
                  </div>

                </div>

                <div className="map-buttons">

                  <button
                    onClick={() =>
                      handleMapRequest("ndvi")
                    }
                    disabled={mapLoading}
                  >
                    <Leaf size={16} />

                    {mapLoading &&
                    mapType === "ndvi"
                      ? "Generating..."
                      : "Generate NDVI Map"}
                  </button>

                  <button
                    onClick={() =>
                      handleMapRequest("ndwi")
                    }
                    disabled={mapLoading}
                  >
                    <Droplets size={16} />

                    {mapLoading &&
                    mapType === "ndwi"
                      ? "Generating..."
                      : "Generate NDWI Map"}
                  </button>

                  <button
                    onClick={handleChangeDetection}
                    disabled={changeLoading}
                  >
                    <Activity size={16} />

                    {changeLoading
                      ? "Analyzing Change..."
                      : "Detect Change"}
                  </button>

                </div>

                {mapLoading && (

                  <div className="map-loading">
                    <span className="loading-spinner"></span>

                    Generating satellite
                    analysis...
                  </div>

                )}

                {mapUrl && (

                  <div
                    className="map-display"
                    id="satellite-map"
                  >

                    <div className="map-display-header">

                      <div className="map-title-group">

                        <div className="map-title-row">

                          {mapType === "ndvi" ? (
                            <Leaf size={18} />
                          ) : (
                            <Droplets size={18} />
                          )}

                          <div>

                            <h3>
                              {mapType === "ndvi"
                                ? "Vegetation Health"
                                : "Water & Moisture"}
                            </h3>

                            <span>
                              {mapType.toUpperCase()}{" "}
                              SATELLITE ANALYSIS
                            </span>

                          </div>

                        </div>

                        <p>
                          {mapType === "ndvi"
                            ? "Normalized Difference Vegetation Index"
                            : "Normalized Difference Water Index"}
                        </p>

                      </div>

                      <a
                        className="map-open-button"
                        href={mapUrl}
                        target="_blank"
                        rel="noreferrer"
                      >
                        Open full map
                        <ArrowRight size={15} />
                      </a>

                    </div>

                    <div className="map-image-wrapper">

                      <img
                        src={mapUrl}
                        alt={`${mapType} satellite analysis`}
                        onError={() =>
                          setError(
                            "The map was generated, but the image could not be displayed."
                          )
                        }
                      />

                    </div>

                  </div>

                )}

              </div>

            )}

          {/* =================================================
              CHANGE DETECTION RESULT
          ================================================= */}

          {changeResult?.result && (

            <div
              className="change-detection-card"
              id="change-detection-result"
            >

              <div className="change-detection-header">

                <div>

                  <div className="section-label">
                    CHANGE DETECTION
                  </div>

                  <h3>
                    Vegetation Change Analysis
                  </h3>

                  <p>
                    Comparison between two
                    satellite observation periods.
                  </p>

                </div>

                <div className="change-status">

                  <CheckCircle2 size={14} />

                  ANALYSIS COMPLETE

                </div>

              </div>

              <div className="change-scenes">

                <div>
                  <span>FIRST SCENE</span>

                  <strong>
                    {
                      changeResult.result
                        .first_scene_id
                    }
                  </strong>
                </div>

                <div>
                  <span>SECOND SCENE</span>

                  <strong>
                    {
                      changeResult.result
                        .second_scene_id
                    }
                  </strong>
                </div>

              </div>

              <div className="change-result-grid">

                <div className="change-result-item">
                  <span>
                    FIRST NDVI MEAN
                  </span>

                  <strong>
                    {changeResult.result
                      .first_ndvi_mean
                      ?.toFixed(4)}
                  </strong>
                </div>

                <div className="change-result-item">
                  <span>
                    SECOND NDVI MEAN
                  </span>

                  <strong>
                    {changeResult.result
                      .second_ndvi_mean
                      ?.toFixed(4)}
                  </strong>
                </div>

                <div className="change-result-item">
                  <span>
                    MEAN NDVI CHANGE
                  </span>

                  <strong>
                    {changeResult.result
                      .mean_ndvi_change
                      ?.toFixed(4)}
                  </strong>
                </div>

                <div className="change-result-item">
                  <span>
                    MINIMUM CHANGE
                  </span>

                  <strong>
                    {changeResult.result
                      .minimum_change
                      ?.toFixed(4)}
                  </strong>
                </div>

                <div className="change-result-item">
                  <span>
                    MAXIMUM CHANGE
                  </span>

                  <strong>
                    {changeResult.result
                      .maximum_change
                      ?.toFixed(4)}
                  </strong>
                </div>

              </div>

            </div>

          )}

          {/* =================================================
              PIPELINE DETAILS
          ================================================= */}

          <details className="technical-details">

            <summary>
              <span>
                VIEW PIPELINE DETAILS
              </span>

              <ArrowRight size={14} />
            </summary>

            <div className="technical-grid">

              <div>
                <span>01 · QUERY</span>

                <strong>
                  {result.query || "—"}
                </strong>
              </div>

              <div>
                <span>QUERY ID</span>

                <strong>
                  {result.query_id || "—"}
                </strong>
              </div>

              <div>
                <span>02 · VISUAL QUERY</span>

                <strong>
                  {result.structured_query
                    ?.visual_query || "—"}
                </strong>
              </div>

              <div>
                <span>ANALYSIS TYPE</span>

                <strong>
                  {formatLabel(analysisType)}
                </strong>
              </div>

              <div>
                <span>03 · LOCATION</span>

                <strong>
                  {result.geographic_result
                    ?.name || "—"}
                </strong>
              </div>

              <div>
                <span>COORDINATES</span>

                <strong>
                  {result.geographic_result
                    ?.latitude !== null &&
                  result.geographic_result
                    ?.latitude !== undefined &&
                  result.geographic_result
                    ?.longitude !== null &&
                  result.geographic_result
                    ?.longitude !== undefined
                    ? `${formatValue(
                        result.geographic_result
                          .latitude
                      )}, ${formatValue(
                        result.geographic_result
                          .longitude
                      )}`
                    : "—"}
                </strong>
              </div>

              <div>
                <span>04 · VALIDATION</span>

                <strong
                  className={
                    validationPassed
                      ? "pipeline-success"
                      : ""
                  }
                >
                  {result.validation_status ||
                    "—"}
                </strong>
              </div>

              <div>
                <span>VALIDATION REASON</span>

                <strong>
                  {result.validation_reason ||
                    "—"}
                </strong>
              </div>

              <div>
                <span>05 · SATELLITE SCENE</span>

                <strong className="scene-id">
                  {result.analysis_result
                    ?.scene?.scene_id || "—"}
                </strong>
              </div>

              <div>
                <span>ACQUIRED</span>

                <strong>
                  {result.analysis_result
                    ?.scene?.datetime
                    ? new Date(
                        result.analysis_result
                          .scene.datetime
                      ).toLocaleString()
                    : "—"}
                </strong>
              </div>

              <div>
                <span>CLOUD COVER</span>

                <strong>
                  {result.analysis_result
                    ?.scene?.cloud_cover !==
                    null &&
                  result.analysis_result
                    ?.scene?.cloud_cover !==
                    undefined
                    ? `${result.analysis_result.scene.cloud_cover}%`
                    : "—"}
                </strong>
              </div>

              <div>
                <span>VALID PIXELS</span>

                <strong>
                  {result.analysis_result
                    ?.scene
                    ?.valid_pixel_fraction !==
                    null &&
                  result.analysis_result
                    ?.scene
                    ?.valid_pixel_fraction !==
                    undefined
                    ? `${(
                        result.analysis_result
                          .scene
                          .valid_pixel_fraction *
                        100
                      ).toFixed(1)}%`
                    : "—"}
                </strong>
              </div>

              <div>
                <span>06 · SPECTRAL ANALYSIS</span>

                <strong>
                  {formatLabel(analysisType)}
                </strong>
              </div>

              {getMetrics().map(
                ([key, value]) => (

                  <div
                    key={`pipeline-${key}`}
                  >
                    <span>
                      {formatLabel(key)}
                    </span>

                    <strong>
                      {formatValue(value)}
                    </strong>
                  </div>

                )
              )}

              <div>
                <span>07 · NDVI MAP</span>

                <strong
                  className={
                    mapType === "ndvi" &&
                    mapUrl
                      ? "pipeline-success"
                      : ""
                  }
                >
                  {mapType === "ndvi" &&
                  mapUrl
                    ? "GENERATED"
                    : "READY"}
                </strong>
              </div>

              <div>
                <span>08 · NDWI MAP</span>

                <strong
                  className={
                    mapType === "ndwi" &&
                    mapUrl
                      ? "pipeline-success"
                      : ""
                  }
                >
                  {mapType === "ndwi" &&
                  mapUrl
                    ? "GENERATED"
                    : "READY"}
                </strong>
              </div>

              <div>
                <span>09 · CHANGE DETECTION</span>

                <strong
                  className={
                    changeResult?.result
                      ? "pipeline-success"
                      : ""
                  }
                >
                  {changeResult?.result
                    ? "COMPLETE"
                    : "READY"}
                </strong>
              </div>

              <div className="pipeline-source">
                <span>DATA SOURCE</span>

                <strong>
                  {result.evidence?.source ||
                    "Satellite imagery"}
                </strong>
              </div>

            </div>

          </details>

        </section>

      )}

      {/* =====================================================
          EXPLORE
      ===================================================== */}

      <section
        id="explore"
        className="product-section explore-section"
      >

        <div className="section-inner">

          <div className="section-label">
            01 · EXPLORE
          </div>

          <div className="section-heading-row">

            <div>

              <h2>
                Explore the Earth.
              </h2>

              <p className="section-intro">
                Start with a question. GeoQuery
                translates natural language into
                a geographic and satellite analysis.
              </p>

            </div>

            <div className="section-number">
              01
            </div>

          </div>

          <div className="explore-grid">

            <div className="explore-card large">

              <div className="explore-icon">
                <Sparkles size={22} />
              </div>

              <span className="card-index">
                01
              </span>

              <h3>
                Ask naturally.
              </h3>

              <p>
                Describe what you want to find
                using ordinary language. No
                complex GIS commands are required.
              </p>

              <button
                onClick={() =>
                  selectExample(
                    "Vegetation loss near Bengaluru"
                  )
                }
              >
                Try a query
                <ArrowRight size={15} />
              </button>

            </div>

            <div className="explore-card">

              <div className="explore-icon">
                <MapPin size={22} />
              </div>

              <span className="card-index">
                02
              </span>

              <h3>
                Locate precisely.
              </h3>

              <p>
                GeoQuery grounds the request to a
                geographic location before analysis.
              </p>

            </div>

            <div className="explore-card">

              <div className="explore-icon">
                <Satellite size={22} />
              </div>

              <span className="card-index">
                03
              </span>

              <h3>
                Discover satellite scenes.
              </h3>

              <p>
                Relevant satellite observations
                become the evidence behind the
                analysis.
              </p>

            </div>

          </div>

          <div className="explore-strip">

            <div>
              <Globe2 size={18} />
              <span>
                GEOGRAPHIC GROUNDING
              </span>
            </div>

            <div>
              <Brain size={18} />
              <span>
                AI QUERY UNDERSTANDING
              </span>
            </div>

            <div>
              <Database size={18} />
              <span>
                SATELLITE EVIDENCE
              </span>
            </div>

            <div>
              <Target size={18} />
              <span>
                ANALYSIS READY
              </span>
            </div>

          </div>

        </div>

      </section>

      {/* =====================================================
          ANALYSIS
      ===================================================== */}

      <section
        id="analysis"
        className="product-section analysis-section"
      >

        <div className="section-inner">

          <div className="section-label">
            02 · ANALYSIS
          </div>

          <div className="section-heading-row">

            <div>

              <h2>
                Understand the Change.
              </h2>

              <p className="section-intro">
                Turn satellite observations into
                measurable environmental signals.
              </p>

            </div>

            <div className="section-number">
              02
            </div>

          </div>

          <div className="analysis-grid">

            {/* NDVI */}

            <div className="analysis-card">

              <div className="analysis-card-top">

                <div className="analysis-icon">
                  <Leaf size={23} />
                </div>

                <span>
                  SPECTRAL INDEX
                </span>

              </div>

              <h3>
                NDVI
              </h3>

              <p>
                Measure vegetation health and
                identify areas where plant cover
                is strong, weak, or changing.
              </p>

              <div className="analysis-meta">
                <span>
                  VEGETATION
                </span>

                <span>
                  −1 → +1
                </span>
              </div>

              <button
                onClick={() =>
                  selectAnalysis("ndvi")
                }
              >
                Explore NDVI
                <ArrowRight size={15} />
              </button>

            </div>

            {/* NDWI */}

            <div className="analysis-card">

              <div className="analysis-card-top">

                <div className="analysis-icon">
                  <Droplets size={23} />
                </div>

                <span>
                  SPECTRAL INDEX
                </span>

              </div>

              <h3>
                NDWI
              </h3>

              <p>
                Highlight water and moisture
                patterns using spectral information
                from satellite imagery.
              </p>

              <div className="analysis-meta">
                <span>
                  WATER
                </span>

                <span>
                  −1 → +1
                </span>
              </div>

              <button
                onClick={() =>
                  selectAnalysis("ndwi")
                }
              >
                Explore NDWI
                <ArrowRight size={15} />
              </button>

            </div>

            {/* CHANGE */}

            <div className="analysis-card featured">

              <div className="analysis-card-top">

                <div className="analysis-icon">
                  <Activity size={23} />
                </div>

                <span>
                  TEMPORAL ANALYSIS
                </span>

              </div>

              <h3>
                Change Detection
              </h3>

              <p>
                Compare two satellite observation
                periods and quantify how vegetation
                signals changed over time.
              </p>

              <div className="analysis-meta">
                <span>
                  TIME SERIES
                </span>

                <span>
                  Δ NDVI
                </span>
              </div>

              <button
                onClick={() =>
                  selectAnalysis("change")
                }
              >
                Detect Change
                <ArrowRight size={15} />
              </button>

            </div>

          </div>

          <div className="analysis-flow">

            <div className="flow-title">
              ANALYSIS PIPELINE
            </div>

            <div className="flow-steps">

              <div className="flow-step">
                <span>01</span>
                <Search size={17} />
                <strong>
                  Query
                </strong>
              </div>

              <div className="flow-line"></div>

              <div className="flow-step">
                <span>02</span>
                <MapPin size={17} />
                <strong>
                  Location
                </strong>
              </div>

              <div className="flow-line"></div>

              <div className="flow-step">
                <span>03</span>
                <Satellite size={17} />
                <strong>
                  Satellite
                </strong>
              </div>

              <div className="flow-line"></div>

              <div className="flow-step">
                <span>04</span>
                <Layers3 size={17} />
                <strong>
                  Analysis
                </strong>
              </div>

              <div className="flow-line"></div>

              <div className="flow-step">
                <span>05</span>
                <Eye size={17} />
                <strong>
                  Insight
                </strong>
              </div>

            </div>

          </div>

        </div>

      </section>

      {/* =====================================================
          ABOUT
      ===================================================== */}

      <section
        id="about"
        className="product-section about-section"
      >

        <div className="section-inner">

          <div className="section-label">
            03 · ABOUT GEOQUERY
          </div>

          <div className="about-main">

            <div className="about-copy">

              <h2>
                Intelligence
                <br />
                from above.
              </h2>

              <p>
                GeoQuery is designed to make
                geospatial intelligence easier to
                access. Instead of navigating
                complicated satellite-data
                workflows, users can ask questions
                in natural language and receive
                measurable satellite-derived
                insights.
              </p>

              <button
                className="about-action"
                onClick={() =>
                  scrollToSection("home")
                }
              >
                Ask GeoQuery
                <ArrowRight size={16} />
              </button>

            </div>

            <div className="about-visual">

              <div className="about-orbit"></div>

              <div className="about-core">
                <Globe2 size={34} />
                <span>
                  GEO
                </span>
                <strong>
                  QUERY
                </strong>
              </div>

              <div className="about-node node-a">
                AI
              </div>

              <div className="about-node node-b">
                GIS
              </div>

              <div className="about-node node-c">
                EO
              </div>

            </div>

          </div>

          {/* TECHNOLOGY */}

          <div className="technology-section">

            <div className="technology-heading">

              <span>
                UNDER THE HOOD
              </span>

              <h3>
                From question to satellite insight.
              </h3>

            </div>

            <div className="technology-grid">

              <div className="technology-item">

                <Brain size={19} />

                <div>
                  <span>
                    AI
                  </span>

                  <strong>
                    Natural Language Understanding
                  </strong>

                  <p>
                    Converts user questions into
                    structured analysis intent.
                  </p>
                </div>

              </div>

              <div className="technology-item">

                <MapPin size={19} />

                <div>
                  <span>
                    GEO
                  </span>

                  <strong>
                    Geographic Grounding
                  </strong>

                  <p>
                    Connects a requested place with
                    usable coordinates.
                  </p>
                </div>

              </div>

              <div className="technology-item">

                <Satellite size={19} />

                <div>
                  <span>
                    EO
                  </span>

                  <strong>
                    Earth Observation
                  </strong>

                  <p>
                    Uses satellite observations as
                    evidence for analysis.
                  </p>
                </div>

              </div>

              <div className="technology-item">

                <BarChart3 size={19} />

                <div>
                  <span>
                    DATA
                  </span>

                  <strong>
                    Quantitative Analysis
                  </strong>

                  <p>
                    Produces measurable metrics,
                    maps, and change indicators.
                  </p>
                </div>

              </div>

            </div>

          </div>

          <div className="about-footer">

            <span>
              GEOQUERY
            </span>

            <span>
              EXPLORE · ANALYZE · UNDERSTAND
            </span>

            <span>
              SATELLITE INTELLIGENCE
            </span>

          </div>

        </div>

      </section>

    </div>
  );
}

export default App;