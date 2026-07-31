## Agentic Ecommerce LangGraph

### Local Development
Run the FastAPI app locally:

```bash
uv run fastapi dev
```

### Calling the API
Once the server is running, you can call the graph endpoint with a JSON body:

```bash
curl -X POST "http://127.0.0.1:8000/run-graph" \
  -H "Content-Type: application/json" \
  -d '{"text": "Calculate the cargo weight for 10 boxes at 3.5 kg each"}'
```

The response includes:
- `summary`: a short summary of the run
- `original_query`: the original input text
- `react.steps`: the ReAct-style execution trace
- `react.final_answer`: the final answer produced by the model

You can also test the route directly in the browser by opening:

```text
http://127.0.0.1:8000/docs
```
