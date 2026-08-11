# decision-tree-forecast

Decision tree regression for prediction and forecasting tasks

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your credentials
python main.py
```


## Tools

- **predict** — Make predictions from features
- **train** — Train model on dataset

## Docker

```bash
docker build -t decision-tree-forecast .
docker run -p 8080:8080 --env-file .env decision-tree-forecast
```

## Health Check

```bash
curl http://localhost:8080/health
```

## API

Connect via MCP protocol:
- **SSE endpoint:** `http://localhost:8080/sse`
- **Message endpoint:** `http://localhost:8080/message`

## License

MIT
