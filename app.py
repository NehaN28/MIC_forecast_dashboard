
import os
import base64
import io
import pandas as pd
import numpy as np
from functools import lru_cache

from dash import Dash, dcc, html, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objects as go

# ---------- Config ----------
DATA_PATH = os.getenv("DATA_PATH", "data/MIC_params_forecasts_v4_twofold_with_global.xlsx")
PORT = int(os.getenv("PORT", "8080"))

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = "MIC Forecast Dashboard"

# ---------- Loaders ----------
@lru_cache(maxsize=2)
def load_default():
    if os.path.exists(DATA_PATH):
        df = pd.read_excel(DATA_PATH, sheet_name=0, engine="openpyxl")
        return df
    return pd.DataFrame()

def harmonize(df):
    df = df.copy()
    # Expected columns from your file
    # 'Isolate','WHO region','Target_Drug','Year','RowType',
    #   %R, %R_pred, %R_pred_CI_low, %R_pred_CI_high,
    #   Mean_log2_MIC, Mean_log2_MIC_pred, Mean_log2_MIC_pred_CI_low, Mean_log2_MIC_pred_CI_high,
    #   MIC50, MIC50_pred, MIC50_pred_CI_low, MIC50_pred_CI_high,
    #   MIC90, MIC90_pred, MIC90_pred_CI_low, MIC90_pred_CI_high
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    df["Combo"] = df["Isolate"].astype(str).str.strip() + " | " + df["Target_Drug"].astype(str).str.strip()
    return df

PARAMS = {
    "Mean_log2_MIC": {
        "obs": "Mean_log2_MIC",
        "pred": "Mean_log2_MIC_pred",
        "low": "Mean_log2_MIC_pred_CI_low",
        "high": "Mean_log2_MIC_pred_CI_high",
        "ylabel": "Mean log2 MIC",
        "break_y": 4.0,  # log2 16
    },
    "%R (Proportion Resistant)": {
        "obs": "%R",
        "pred": "%R_pred",
        "low": "%R_pred_CI_low",
        "high": "%R_pred_CI_high",
        "ylabel": "% Resistant",
        "break_y": None
    },
    "MIC50": {
        "obs": "MIC50",
        "pred": "MIC50_pred",
        "low": "MIC50_pred_CI_low",
        "high": "MIC50_pred_CI_high",
        "ylabel": "MIC50",
        "break_y": None
    },
    "MIC90": {
        "obs": "MIC90",
        "pred": "MIC90_pred",
        "low": "MIC90_pred_CI_low",
        "high": "MIC90_pred_CI_high",
        "ylabel": "MIC90",
        "break_y": None
    }
}

# ---------- Layout ----------
app.layout = html.Div([
    dcc.Location(id="url"),
    html.H1("MIC Parameters — Observed vs Predicted", className="title"),
    html.Div("Upload a file or use the bundled dataset.", className="subtitle"),

    dcc.Upload(
        id="upload",
        children=html.Div(["Drag & Drop or ", html.A("Select Excel")]),
        style={"width":"100%","height":"56px","lineHeight":"56px","border":"1px dashed #bbb","borderRadius":"10px","textAlign":"center","marginBottom":"10px"},
        multiple=False
    ),
    dcc.Store(id="store-data"),
    dcc.Store(id="store-state"),

    html.Div([
        html.Div([
            html.Label("Combo"),
            dcc.Dropdown(id="combo")
        ], className="col"),
        html.Div([
            html.Label("Parameter"),
            dcc.Dropdown(
                id="param",
                options=[{"label":k, "value":k} for k in PARAMS.keys()],
                value="Mean_log2_MIC"
            )
        ], className="col"),
        html.Div([
            html.Label("WHO Regions"),
            dcc.Dropdown(id="regions", multi=True)
        ], className="col"),
        html.Div([
            html.Label("Options"),
            dcc.Checklist(
                id="options",
                options=[
                    {"label":" Show 95% CI", "value":"ci"},
                    {"label":" Show observed", "value":"obs"},
                    {"label":" Show predicted", "value":"pred"},
                ],
                value=["ci","obs","pred"]
            )
        ], className="col")
    ], className="row"),

    html.Div(id="kpi-cards", className="kpis"),
    dcc.Graph(id="fig", style={"height":"560px"}),
    html.Div([
        html.Button("Download PNG", id="btn-png", className="btn"),
        dcc.Download(id="download-png")
    ], style={"textAlign":"right","marginTop":"8px"}),
    html.H3("Data Table"),
    dash_table.DataTable(
        id="table",
        page_size=12,
        style_table={"overflowX":"auto"},
        style_cell={"padding":"6px","textAlign":"left"},
        sort_action="native",
        export_format="csv"
    ),
    html.Div(className="footer", children=[
        html.Span("Tip: Use the URL to share a specific view — filters are encoded in the address."),
    ])
], className="container")

# ---------- Callbacks ----------
@app.callback(
    Output("store-data","data"),
    Output("combo","options"),
    Output("combo","value"),
    Output("regions","options"),
    Output("regions","value"),
    Input("upload","contents"),
    State("upload","filename"),
    Input("url","href")
)
def ingest(uploaded, filename, href):
    # Load data
    if uploaded is None:
        df = load_default()
    else:
        content_type, content_string = uploaded.split(",")
        decoded = base64.b64decode(content_string)
        df = pd.read_excel(io.BytesIO(decoded), sheet_name=0, engine="openpyxl")

    if df is None or df.empty:
        return {}, [], None, [], []

    df = harmonize(df)

    combo_opts = sorted(df["Combo"].dropna().unique().tolist())
    regions_all = sorted(df["WHO region"].dropna().astype(str).unique().tolist())

    # default selection or from URL
    default_combo = combo_opts[0] if combo_opts else None
    default_regions = regions_all[:1]

    # parse query params
    import urllib.parse as up
    if href:
        try:
            q = up.urlparse(href).query
            qs = dict(up.parse_qsl(q))
            if "combo" in qs and qs["combo"] in combo_opts:
                default_combo = qs["combo"]
            if "regions" in qs:
                want = [r for r in qs["regions"].split(",") if r in regions_all]
                if want:
                    default_regions = want
        except Exception:
            pass

    return df.to_json(orient="records"), [{"label":c,"value":c} for c in combo_opts], default_combo,            [{"label":r,"value":r} for r in regions_all], default_regions

@app.callback(
    Output("param","value"),
    Input("url","href"),
    State("param","value")
)
def set_param_from_url(href, current):
    if not href:
        return current
    import urllib.parse as up
    try:
        q = up.urlparse(href).query
        qs = dict(up.parse_qsl(q))
        if "param" in qs and qs["param"] in PARAMS.keys():
            return qs["param"]
    except Exception:
        pass
    return current

@app.callback(
    Output("fig","figure"),
    Output("table","data"),
    Output("table","columns"),
    Output("kpi-cards","children"),
    Input("store-data","data"),
    Input("combo","value"),
    Input("regions","value"),
    Input("param","value"),
    Input("options","value"),
)
def update(data_json, combo, regions, param_label, opts):
    if not data_json or not combo or not regions:
        return go.Figure(), [], [], []

    df = pd.DataFrame.from_records(pd.read_json(data_json, orient="records"))
    p = PARAMS[param_label]
    obs_col, pred_col = p["obs"], p["pred"]
    low_col, high_col = p["low"], p["high"]
    ylabel = p["ylabel"]
    show_ci = "ci" in (opts or [])
    show_obs = "obs" in (opts or [])
    show_pred = "pred" in (opts or [])

    fig = go.Figure()
    palette = px.colors.qualitative.Plotly
    region_to_color = {r: palette[i % len(palette)] for i, r in enumerate(regions)}

    kpi_children = []
    for region in regions:
        d = df[(df["Combo"]==combo) & (df["WHO region"].astype(str)==str(region))].copy()
        if d.empty:
            continue
        d = d.sort_values("Year")

        if show_obs and obs_col in d.columns:
            obs = d[d["RowType"].str.lower()=="observed"][["Year", obs_col]].dropna()
            if not obs.empty:
                fig.add_trace(go.Scatter(
                    x=obs["Year"], y=obs[obs_col],
                    mode="lines+markers",
                    name=f"{region} — Observed",
                    line=dict(dash="solid", color=region_to_color[region]),
                    connectgaps=True
                ))
                # KPI latest observed
                last_obs = obs.dropna().tail(1)[obs_col].values[0]
            else:
                last_obs = np.nan
        else:
            last_obs = np.nan

        if show_pred and pred_col in d.columns:
            pred = d[d["RowType"].str.lower()=="predicted"][["Year", pred_col, low_col, high_col]].dropna(subset=[pred_col])
            if not pred.empty:
                fig.add_trace(go.Scatter(
                    x=pred["Year"], y=pred[pred_col],
                    mode="lines+markers",
                    name=f"{region} — Predicted",
                    line=dict(dash="dot", color=region_to_color[region]),
                    connectgaps=True
                ))
                if show_ci and (low_col in pred.columns) and (high_col in pred.columns):
                    fig.add_trace(go.Scatter(
                        x=list(pred["Year"]) + list(pred["Year"])[::-1],
                        y=list(pred[high_col]) + list(pred[low_col])[::-1],
                        fill="toself",
                        fillcolor="rgba(0,0,0,0.08)",
                        line=dict(width=0),
                        hoverinfo="skip",
                        name=f"{region} — 95% CI",
                        showlegend=False
                    ))
                last_pred = pred.dropna().tail(1)[pred_col].values[0]
            else:
                last_pred = np.nan
        else:
            last_pred = np.nan

        # KPI card per region
        kpi_children.append(html.Div(className="kpi", children=[
            html.Div(region, className="kpi-title"),
            html.Div([
                html.Div([html.Span("Latest Observed"), html.Strong(f"{last_obs:.3g}" if pd.notna(last_obs) else "—")], className="kpi-line"),
                html.Div([html.Span("Latest Predicted"), html.Strong(f"{last_pred:.3g}" if pd.notna(last_pred) else "—")], className="kpi-line"),
            ], className="kpi-body")
        ]))

    # Breakpoint for log2MIC
    if p.get("break_y") is not None:
        fig.add_hline(y=p["break_y"], line_dash="dash",
                      annotation_text="log2 16 breakpoint", annotation_position="top left")

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title=ylabel,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=10, r=10, t=30, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    # Alternate years, horizontal labels
    fig.update_xaxes(tickmode="linear", dtick=2, tickangle=0, automargin=True)

    # Table
    cols = ["Combo","WHO region","Year","RowType",obs_col,pred_col,low_col,high_col]
    cols = [c for c in cols if c in df.columns]
    table_df = df[(df["Combo"]==combo) & (df["WHO region"].astype(str).isin(regions))][cols].sort_values(["WHO region","Year"])
    table_data = table_df.to_dict("records")
    table_columns = [{"name":c, "id":c} for c in table_df.columns]

    return fig, table_data, table_columns, kpi_children

# Download PNG of figure
@app.callback(
    Output("download-png","data"),
    Input("btn-png","n_clicks"),
    State("fig","figure"),
    prevent_initial_call=True
)
def download_png(n, fig_dict):
    import plotly.io as pio
    fig = go.Figure(fig_dict)
    png_bytes = pio.to_image(fig, format="png", scale=2)
    return dict(content=png_bytes, filename="mic_dashboard_plot.png")

# Encode selection into URL (permalinks)
@app.callback(
    Output("url","search"),
    Input("combo","value"),
    Input("regions","value"),
    Input("param","value")
)
def write_url(combo, regions, param):
    import urllib.parse as up
    if not combo or not regions or not param:
        return ""
    qs = up.urlencode({"combo":combo, "regions":",".join(regions), "param":param})
    return f"?{qs}"

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=PORT, debug=False)
