
# MIC Forecast Dashboard

Interactive dashboard for observed vs predicted MIC parameters by drug–bug combo and WHO region.

## Run Locally
```bash
pip install -r requirements.txt
python app.py
# open http://localhost:8080
```

## Deploy (Render)
1. Push this folder to a new GitHub repo.
2. On Render.com → New → Web Service → connect repo.
3. First deploy reads `render.yaml`. Keep Free plan.
4. Upload your Excel as `data/MIC_params_forecasts_v4_twofold_with_global.xlsx` or set `DATA_PATH` env var.
5. Share the public URL. URL query string encodes filters (permalinks).

## Expected Columns
`Isolate, WHO region, Target_Drug, Year, RowType` and for each parameter: observed, predicted, CI low/high, as in your file.
