
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html

# Load data
df = pd.read_csv('data/mic_data.csv')

# Normalize strings to avoid mismatches
for col in ['WHO_region', 'RowType', 'Isolate', 'Target_Drug']:
    df[col] = df[col].astype(str).str.strip().str.upper()

# Initialize app
app = Dash(__name__)

# Layout
app.layout = html.Div([
    html.H1("MIC Forecast Dashboard", style={'textAlign': 'center'}),

    html.Div([
        html.Label('Select Isolate:'),
        dcc.Dropdown(
            id='isolate-dropdown',
            options=[{'label': i, 'value': i} for i in sorted(df['Isolate'].unique())],
            value=sorted(df['Isolate'].unique())[0],
            multi=False
        ),
        html.Label('Select Target Drug:'),
        dcc.Dropdown(
            id='drug-dropdown',
            options=[{'label': i, 'value': i} for i in sorted(df['Target_Drug'].unique())],
            value=sorted(df['Target_Drug'].unique())[0],
            multi=False
        ),
        html.Label('Select WHO Region(s):'),
        dcc.Dropdown(
            id='region-dropdown',
            options=[{'label': i, 'value': i} for i in sorted(df['WHO_region'].unique())],
            value=[sorted(df['WHO_region'].unique())[0]],
            multi=True
        ),
        html.Label('Show Confidence Interval:'),
        dcc.RadioItems(
            id='ci-toggle',
            options=[
                {'label': 'Yes', 'value': 'yes'},
                {'label': 'No', 'value': 'no'}
            ],
            value='yes',
            inline=True
        )
    ], style={'width': '30%', 'display': 'inline-block', 'verticalAlign': 'top'}),

    html.Div([
        dcc.Graph(id='mic-plot')
    ], style={'width': '68%', 'display': 'inline-block', 'paddingLeft': '2%'})
])

# Callbacks
from dash.dependencies import Input, Output

@app.callback(
    Output('mic-plot', 'figure'),
    Input('isolate-dropdown', 'value'),
    Input('drug-dropdown', 'value'),
    Input('region-dropdown', 'value'),
    Input('ci-toggle', 'value')
)
def update_plot(selected_isolate, selected_drug, selected_regions, show_ci):
    if not isinstance(selected_regions, list):
        selected_regions = [selected_regions]

    selected_regions = [str(r).strip().upper() for r in selected_regions]

    fig = go.Figure()

    for region in selected_regions:
        filtered = df[
            (df['Isolate'] == str(selected_isolate).strip().upper()) &
            (df['Target_Drug'] == str(selected_drug).strip().upper()) &
            (df['WHO_region'] == region)
        ]

        if filtered.empty:
            fig.add_annotation(text=f"No data for: {region}",
                               xref="paper", yref="paper",
                               x=0.5, y=0.5, showarrow=False, font=dict(color="red"))
            continue

        filtered = filtered.sort_values('Year')

        fig.add_trace(go.Scatter(
            x=filtered['Year'],
            y=filtered['Mean_log2_MIC'],
            mode='lines+markers',
            name=f"{region} (Observed)"
        ))

        pred_df = filtered[filtered['RowType'] == 'PREDICTED']
        if not pred_df.empty:
            fig.add_trace(go.Scatter(
                x=pred_df['Year'],
                y=pred_df['Mean_log2_MIC'],
                mode='lines+markers',
                name=f"{region} (Predicted)"
            ))

            if show_ci == 'yes' and 'Lower_CI' in pred_df.columns and 'Upper_CI' in pred_df.columns:
                fig.add_trace(go.Scatter(
                    x=list(pred_df['Year']) + list(pred_df['Year'])[::-1],
                    y=list(pred_df['Upper_CI']) + list(pred_df['Lower_CI'])[::-1],
                    fill='toself',
                    fillcolor='rgba(0,100,80,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=False
                ))

    fig.update_layout(
        title=f"{selected_isolate} - {selected_drug} MIC Trends",
        xaxis_title="Year",
        yaxis_title="Mean log2 MIC",
        xaxis=dict(tickmode='array', tickvals=sorted(df['Year'].unique()), tickangle=0),
        legend_title="Region"
    )
    return fig

server = app.server

if __name__ == '__main__':
    app.run_server(debug=True)
