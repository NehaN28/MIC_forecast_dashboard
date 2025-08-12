
import os, base64, io, json
import pandas as pd
import numpy as np
from functools import lru_cache
from dash import Dash, dcc, html, Input, Output, State, dash_table
import plotly.express as px
import plotly.graph_objects as go

# Read from Excel (env var overridable)
DATA_PATH = os.getenv("DATA_PATH", "data/MIC_params_forecasts_v4_twofold_with_global.xlsx")
PORT = int(os.getenv("PORT", "8080"))

app = Dash(__name__, suppress_callback_exceptions=True)
server = app.server
app.title = "MIC Forecast Dashboard"

@lru_cache(maxsize=2)
def load_default():
    if os.path.exists(DATA_PATH):
        return pd.read_excel(DATA_PATH, sheet_name=0, engine="openpyxl")
    return pd.DataFrame()

def harmonize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["Isolate","Target_Drug","WHO region","RowType"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
    if "RowType" in df.columns:
        df["RowType"] = df["RowType"].str.lower()
    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
    if {"Isolate","Target_Drug"} <= set(df.columns):
        df["Combo"] = df["Isolate"] + " | " + df["Target_Drug"]
    return df

PARAMS = {
    "Mean_log2_MIC": {"obs":"Mean_log2_MIC","pred":"Mean_log2_MIC_pred","low":"Mean_log2_MIC_pred_CI_low","high":"Mean_log2_MIC_pred_CI_high","ylabel":"Mean log2 MIC","break_y":4.0},
    "%R (Proportion Resistant)": {"obs":"%R","pred":"%R_pred","low":"%R_pred_CI_low","high":"%R_pred_CI_high","ylabel":"% Resistant","break_y":None},
    "MIC50": {"obs":"MIC50","pred":"MIC50_pred","low":"MIC50_pred_CI_low","high":"MIC50_pred_CI_high","ylabel":"MIC50","break_y":None},
    "MIC90": {"obs":"MIC90","pred":"MIC90_pred","low":"MIC90_pred_CI_low","high":"MIC90_pred_CI_high","ylabel":"MIC90","break_y":None},
}

app.layout = html.Div([
    dcc.Location(id="url"),
    html.H1("MIC Parameters — Observed vs Predicted", className="title"),
    dcc.Upload(
        id="upload",
        children=html.Div(["Drag & Drop or ", html.A("Select Excel")]),
        style={"width":"100%","height":"56px","lineHeight":"56px","border":"1px dashed #bbb","borderRadius":"10px","textAlign":"center","marginBottom":"10px"},
        multiple=False
    ),
    dcc.Store(id="store-data"),
    html.Div([
        html.Div([html.Label("Combo"), dcc.Dropdown(id="combo", persistence=True, persistence_type="memory")], className="col"),
        html.Div([html.Label("Parameter"), dcc.Dropdown(id="param",
            options=[{"label":k, "value":k} for k in PARAMS.keys()],
            value="Mean_log2_MIC", persistence=True, persistence_type="memory")], className="col"),
        html.Div([html.Label("WHO Regions"), dcc.Dropdown(id="regions", multi=True, persistence=True, persistence_type="memory")], className="col"),
        html.Div([html.Label("Options"), dcc.Checklist(id="options",
            options=[{"label":" Show 95% CI", "value":"ci"},
                     {"label":" Show observed", "value":"obs"},
                     {"label":" Show predicted", "value":"pred"}],
            value=["ci","obs","pred"], persistence=True, persistence_type="memory")], className="col"),
    ], className="row"),
    html.Div(id="info-msg", className="info"),
    dcc.Graph(id="fig", style={"height":"560px"}),
    html.H3("Data Table"),
    dash_table.DataTable(id="table", page_size=12, style_table={"overflowX":"auto"},
                         style_cell={"padding":"6px","textAlign":"left"}, sort_action="native", export_format="csv"),
], className="container")

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
    if uploaded is None:
        df = load_default()
    else:
        try:
            content_type, content_string = uploaded.split(",")
            decoded = base64.b64decode(content_string)
            df = pd.read_excel(io.BytesIO(decoded), sheet_name=0, engine="openpyxl")
        except Exception:
            df = pd.DataFrame()

    if df is None or df.empty:
        return {}, [], None, [], []

    df = harmonize(df)
    combos = sorted(df["Combo"].dropna().unique().tolist())
    regions_all = sorted(df["WHO region"].dropna().astype(str).unique().tolist())
    default_combo = combos[0] if combos else None
    default_regions = regions_all[:1]

    return df.to_json(orient="records"), [{"label":c,"value":c} for c in combos], default_combo,            [{"label":r,"value":r} for r in regions_all], default_regions

@app.callback(
    Output("fig","figure"),
    Output("table","data"),
    Output("table","columns"),
    Output("info-msg","children"),
    Input("store-data","data"),
    Input("combo","value"),
    Input("regions","value"),
    Input("param","value"),
    Input("options","value"),
)
def update(data_json, combo, regions, param_label, opts):
    if not data_json or not combo or not regions:
        return go.Figure(), [], [], ""

    if isinstance(regions, str):
        regions = [regions]

    df = pd.DataFrame(json.loads(data_json))
    df = harmonize(df)
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

    missing_regions = []

    for region in regions:
        d = df[(df["Combo"]==combo) & (df["WHO region"]==region)]
        if d.empty:
            missing_regions.append(region)
            continue
        d = d.sort_values("Year")

        if show_obs and obs_col in d.columns:
            obs = d[d["RowType"]=="observed"][["Year", obs_col]].dropna()
            if not obs.empty:
                fig.add_trace(go.Scatter(
                    x=obs["Year"], y=obs[obs_col],
                    mode="lines+markers",
                    name=f"{region} — Observed",
                    line=dict(dash="solid", color=region_to_color[region]),
                    connectgaps=True
                ))

        if show_pred and pred_col in d.columns:
            pred = d[d["RowType"]=="predicted"][["Year", pred_col, low_col, high_col]].dropna(subset=[pred_col])
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

    if p.get("break_y") is not None:
        fig.add_hline(y=p["break_y"], line_dash="dash", annotation_text="log2 16 breakpoint", annotation_position="top left")

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title=ylabel,
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=10, r=10, t=30, b=80),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    fig.update_xaxes(tickmode="linear", dtick=2, tickangle=0, automargin=True)

    cols = ["Combo","WHO region","Year","RowType",obs_col,pred_col,low_col,high_col]
    cols = [c for c in cols if c in df.columns]
    table_df = df[(df["Combo"]==combo) & (df["WHO region"].isin(regions))][cols].sort_values(["WHO region","Year"])
    info = ""
    if missing_regions:
        info = "No rows for: " + ", ".join(missing_regions)

    return fig, table_df.to_dict("records"), [{"name":c,"id":c} for c in cols], info

if __name__ == "__main__":
    app.run_server(host="0.0.0.0", port=PORT, debug=False)
