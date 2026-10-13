# Amazon sales and demand forecasting

## Adding chart knowledge for the assistant

Add one entry to `CHART_REGISTRY` in `chart_knowledge.py` for each new chart or
KPI. Give it a unique `id`, the assistant `tab` (`sales` or `forecast`), a
plain-language `title`, one-sentence `what_it_shows` and `how_to_read`
descriptions, the `data_key` for the dataframe used by the chart, and a
`get_insight(filtered_df)` function. The insight function should calculate
two or three short facts from the dataframe it receives, and return a helpful
message when that dataframe is empty.

Use the same prepared or filtered dataframe that the chart renderer uses.
Registering the entry makes it available through `build_tab_context`, which
adds its description and current facts to the assistant context. For page
summaries, also include the new visual in the matching local summary builder
in `agent_core.py`; keep the summary in chart order and use only values
calculated from the supplied frames.

Run the focused checks after adding chart knowledge:

```powershell
python -m pytest -q
```
