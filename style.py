import streamlit as st

THEMES = {
    "light": {
        "bg-deep": "#F3F5F8",
        "panel": "#FFFFFF",
        "panel-border": "#DCE3EC",
        "ink": "#1B2430",
        "muted": "#5B6675",
        "accent-teal": "#0E7A70",
        "accent-amber": "#A8670A",
        "danger": "#B8402B",
        "header-bg": "rgba(255,255,255,0.86)",
    },
    "dark": {
        "bg-deep": "#0B1220",
        "panel": "#121A2B",
        "panel-border": "#22304A",
        "ink": "#E8EDF5",
        "muted": "#8A97AC",
        "accent-teal": "#2FB8AC",
        "accent-amber": "#E8A33D",
        "danger": "#D6553F",
        "header-bg": "rgba(11,18,32,0.86)",
    },
}

CSS_TEMPLATE = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root{
__ROOT_VARS__
}

html, body, .stApp{
    background-color:var(--bg-deep) !important;
    color:var(--ink);
    font-family:'Inter', sans-serif;
    scroll-behavior:smooth;
}
[data-testid="stHeader"]{ background-color:transparent; pointer-events:none; }
.block-container{ padding-top:0 !important; max-width:1600px; }
#MainMenu, footer{ visibility:hidden; }
[data-testid="stToolbar"],
[data-testid="stAppDeployButton"],
[data-testid="stMainMenu"]{ display:none !important; }

h1,h2,h3, .brand{ font-family:'Space Grotesk', sans-serif; letter-spacing:0.01em; }

/* fixed top bar */
#app-header{
    position:fixed; top:0; left:0; right:0; z-index:9999;
    display:flex; align-items:center; justify-content:space-between;
    padding:16px 100px 16px 40px;
    background:var(--header-bg);
    backdrop-filter:blur(10px);
    border-bottom:1px solid var(--panel-border);
    transition:transform 0.35s ease, background 0.2s ease;
}
#app-header.header-hidden{ transform:translateY(-100%); }
.brand{ font-weight:600; font-size:1.05rem; color:var(--ink); }
.brand span{ color:var(--accent-teal); }
.nav a{
    color:var(--muted); text-decoration:none; margin-left:32px;
    font-size:0.88rem; font-weight:500; transition:color 0.2s;
}
.nav a:hover{ color:var(--accent-teal); }
.header-spacer{ height:78px; }

/* theme toggle button, pinned over the header's right edge */
.st-key-theme_toggle{
    position:fixed; top:13px; right:40px; z-index:10000;
    width:44px !important; height:44px !important;
}
.st-key-theme_toggle .stButton button{
    width:40px; height:40px; border-radius:50%;
    background:var(--panel) !important; border:1px solid var(--panel-border) !important;
    color:var(--ink) !important; font-size:1.05rem;
    padding:0 !important; box-shadow:none !important;
    display:flex; align-items:center; justify-content:center;
    transition:border-color 0.2s;
}
.st-key-theme_toggle .stButton button:hover{ border-color:var(--accent-teal) !important; }

/* hero */
.hero-eyebrow{
    color:var(--accent-teal); font-family:'IBM Plex Mono', monospace;
    font-size:0.8rem; letter-spacing:0.12em; text-transform:uppercase;
    margin-bottom:6px;
}
.hero-title{ font-size:2.15rem; font-weight:600; margin:0 0 8px 0; line-height:1.25; }
.hero-sub{ color:var(--muted); font-size:1rem; max-width:640px; margin-bottom:8px; }

/* scanner-style panels */
.scan-panel{
    position:relative;
    background:var(--panel);
    border:1px solid var(--panel-border);
    border-radius:6px;
    padding:18px;
    min-height:200px;
}

.panel-label{
    font-family:'IBM Plex Mono', monospace; font-size:0.9rem;
    letter-spacing:0.08em; text-transform:uppercase; color:var(--muted);
    margin-bottom:10px;
}

.placeholder-box{ position:relative; overflow:hidden; border-radius:4px; height:240px;
    background:var(--bg-deep); display:flex; align-items:center; justify-content:center; }
.placeholder-text{
    color:var(--muted); font-size:0.85rem; text-align:center; padding:0 24px;
}

/* prediction verdict card (Yes/No result) */
.verdict-card{
    background:var(--panel);
    border:1px solid var(--panel-border);
    border-left:4px solid var(--accent-teal);
    border-radius:6px;
    padding:20px 18px;
}
.verdict-line{
    font-family:'Space Grotesk', sans-serif;
    font-size:1.2rem;
    font-weight:700;
    letter-spacing:0.01em;
    margin-bottom:10px;
}
.verdict-sub{
    color:var(--muted);
    font-size:0.9rem;
    margin-top:4px;
}
.verdict-sub b{ color:var(--ink); }

/* mini panels (segmentation grid) */
.mini-panel{
    background:var(--panel); border:1px solid var(--panel-border);
    border-radius:6px; padding:10px; text-align:center;
}
.mini-tag{
    margin-top:8px; font-family:'IBM Plex Mono', monospace; font-size:0.72rem;
    color:var(--muted); letter-spacing:0.04em; text-transform:uppercase;
}

/* screen tabs (Screen 1 / Screen 2) */
[data-testid="stTabs"] [role="tablist"]{ gap:4px; border-bottom:1px solid var(--panel-border); }
[data-testid="stTabs"] [role="tab"],
[data-testid="stTabs"] [role="tab"] *{
    font-family:'IBM Plex Mono', monospace; font-size:0.8rem; color:var(--muted) !important;
    letter-spacing:0.05em; text-transform:uppercase; padding:8px 4px;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"],
[data-testid="stTabs"] [role="tab"][aria-selected="true"] *{ color:var(--accent-teal) !important; }
[data-testid="stTabs"] .react-aria-SelectionIndicator{
    background-color:var(--accent-teal) !important;
    top:auto !important;
    bottom:0 !important;
    height:3px !important;
    padding:0 !important;
    border-radius:2px;
}
.st-key-workstation{
    background:var(--panel); border:1px solid var(--panel-border);
    border-radius:6px; padding:20px;
}

/* feature-map lightbox viewer: prev/next arrows + filmstrip */
/* text-align:center handles Streamlit's image wrapper being inline-block
   (where margin:auto on the <img> itself doesn't center it); margin:auto is
   kept too as a fallback for the case where it's a block element instead */
.st-key-main_feature_plot{ text-align:center; }
.st-key-main_feature_plot img{ display:inline-block; margin-left:auto; margin-right:auto; }
.st-key-prev_layer button, .st-key-next_layer button{
    width:40px; height:40px; border-radius:50%;
    background:var(--panel) !important; border:1px solid var(--panel-border) !important;
    color:var(--ink) !important; font-size:1.2rem; padding:0 !important;
    box-shadow:none !important;
}
.st-key-prev_layer button:hover, .st-key-next_layer button:hover{
    border-color:var(--accent-teal) !important; color:var(--accent-teal) !important;
}
.filmstrip-caption{
    background:var(--bg-deep); border:1px solid var(--panel-border); border-radius:4px;
    padding:8px 14px; margin-bottom:6px; text-align:center;
    font-family:'IBM Plex Mono', monospace; font-size:0.8rem; color:var(--muted);
}
.st-key-thumb_selected{
    width:100px; margin:0 auto;
    border:2px solid var(--accent-teal); border-radius:6px;
    padding:2px; line-height:0;
}
.st-key-thumb_selected img{ display:block; width:100%; border-radius:4px; }
/* clickable filmstrip thumbnails: the button's own background-image IS the
   plot (set per-layer in app.py) — dimmed until hovered, no visible label.
   `!important` is required: the native-button theming rule further down
   (`[data-testid="stBaseButton-secondary"] { background:...!important }`)
   otherwise wipes out background-image/-size/-position entirely.
   Fixed width (instead of 100%) keeps these centered instead of
   stretching to fill their column. */
[class*="st-key-thumb_dim_"]{
    width:100px; margin:0 auto;
    border:1px solid var(--panel-border); border-radius:6px;
    padding:0; line-height:0; opacity:0.55; transition:opacity 0.2s; overflow:hidden;
}
[class*="st-key-thumb_dim_"]:hover{ opacity:1; }
[class*="st-key-thumb_dim_"] .stButton{ margin:0 !important; }
[class*="st-key-thumb_dim_"] button{
    display:block; width:100%; aspect-ratio:1/1; height:auto; min-height:0 !important; cursor:pointer;
    border:none !important; box-shadow:none !important; border-radius:4px;
    padding:0 !important; margin:0 !important;
    background-repeat:no-repeat !important;
    background-position:center !important;
    background-size:cover !important;
}
/* filmstrip row: make the columns shrink-wrap around each fixed-size
   thumbnail and sit close together, instead of Streamlit's default of
   stretching them to evenly fill the whole row width */
.st-key-filmstrip_row [data-testid="stHorizontalBlock"]{
    justify-content:center; gap:10px;
}
.st-key-filmstrip_row [data-testid*="olumn"]{
    flex:0 0 auto !important; width:auto !important; min-width:0 !important;
}

/* about */
.about-section{ border-top:1px solid var(--panel-border); padding-top:48px; margin-top:24px; }

/* workflow steps */
.workflow-step{
    background:var(--bg-deep); border:1px solid var(--panel-border); border-radius:6px;
    padding:20px 16px; text-align:center; height:100%;
}
.workflow-num{
    display:inline-flex; align-items:center; justify-content:center;
    width:32px; height:32px; border-radius:50%;
    background:var(--accent-teal); color:var(--bg-deep);
    font-family:'Space Grotesk', sans-serif; font-weight:700; font-size:0.95rem;
    margin-bottom:12px;
}
.workflow-title{ font-weight:600; margin-bottom:6px; font-size:0.95rem; }
.workflow-desc{ color:var(--muted); font-size:0.82rem; line-height:1.45; }
.workflow-arrow{
    display:flex; align-items:center; justify-content:center; height:100%;
    color:var(--panel-border); font-size:1.4rem;
}

/* native widget theming, so undecorated Streamlit elements still match */
[data-testid="stFileUploaderDropzone"]{
    background:var(--bg-deep) !important;
    border:1px dashed var(--panel-border) !important;
    border-radius:6px;
}
[data-testid="stExpander"]{
    background:var(--panel);
    border:1px solid var(--panel-border);
    border-radius:6px;
}
[data-testid="stVerticalBlockBorderWrapper"]{
    background:var(--panel);
    border:1px solid var(--panel-border) !important;
    border-radius:6px;
}
[data-testid="stMetric"]{
    background:var(--bg-deep);
    border:1px solid var(--panel-border);
    border-radius:6px;
    padding:12px 16px;
}
[data-testid="stMetricValue"]{ font-family:'Space Grotesk', sans-serif; color:var(--accent-teal); }
[data-testid="stMetricLabel"]{
    font-family:'IBM Plex Mono', monospace; font-size:0.75rem;
    letter-spacing:0.06em; text-transform:uppercase; color:var(--muted);
}
[data-testid="stDataFrame"]{ border:1px solid var(--panel-border); border-radius:6px; overflow:hidden; }

/* model summary table (plain HTML, so it follows our theme instead of the
   native st.dataframe grid, which renders on a canvas tied to Streamlit's
   own light/dark theme and ignores this app's custom toggle) */
.model-summary-table-wrap{
    border:1px solid var(--panel-border); border-radius:6px;
    overflow:auto; max-height:520px;
}
.model-summary-table{ width:100%; border-collapse:collapse; font-size:0.85rem; }
.model-summary-table thead th{
    position:sticky; top:0; z-index:1;
    background:var(--bg-deep); color:var(--muted);
    font-family:'IBM Plex Mono', monospace; font-size:0.72rem;
    letter-spacing:0.06em; text-transform:uppercase;
    text-align:left; padding:10px 14px; border-bottom:1px solid var(--panel-border);
}
.model-summary-table tbody td{
    padding:8px 14px; color:var(--ink);
    border-bottom:1px solid var(--panel-border);
}
.model-summary-table tbody tr:last-child td{ border-bottom:none; }
.model-summary-table tbody tr:hover td{ background:var(--bg-deep); }

/* force native buttons (e.g. the file uploader's Browse button) to follow our
   theme instead of the browser/OS's own dark-mode preference */
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-primary"]{
    background:var(--panel) !important;
    color:var(--ink) !important;
    border:1px solid var(--panel-border) !important;
}
[data-testid="stBaseButton-secondary"]:hover,
[data-testid="stBaseButton-primary"]:hover{
    color:var(--accent-teal) !important;
    border-color:var(--accent-teal) !important;
}

/* react-aria form controls (selectbox, slider, etc.) also follow the OS/browser
   dark-mode preference natively — force them onto our theme too */
[data-testid="stSelectbox"] [role="group"],
[data-testid="stNumberInput"] [role="group"]{
    background:var(--panel) !important;
    border:1px solid var(--panel-border) !important;
    color:var(--ink) !important;
}
[data-testid="stSelectbox"] input,
[data-testid="stSelectbox"] [role="group"] *{
    color:var(--ink) !important;
    background:transparent !important;
}

@media (max-width: 768px){
    #app-header{ padding:14px 76px 14px 20px; }
    .nav a{ margin-left:16px; font-size:0.8rem; }
    .hero-title{ font-size:1.6rem; }
    .st-key-theme_toggle{ top:11px; right:16px; width:36px !important; height:36px !important; }
    .st-key-theme_toggle .stButton button{ width:34px; height:34px; }
}
</style>
"""


def inject_style(theme: str = "light"):
    """Injects the app's visual theme (light/dark clinical style, header, panels)."""
    colors = THEMES.get(theme, THEMES["light"])
    root_vars = "\n".join(f"    --{k}:{v};" for k, v in colors.items())
    css = CSS_TEMPLATE.replace("__ROOT_VARS__", root_vars)
    st.markdown(css, unsafe_allow_html=True)
