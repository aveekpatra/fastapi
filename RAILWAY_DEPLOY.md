# Railway Deployment (FastAPI)

## Files Structure

```
📁 new jusctice.cz/
├── main.py                  # FastAPI app with pipeline control
├── vectorize_new_fast.py    # Vectorization pipeline
├── requirements.txt         # Python dependencies
├── railway.toml             # Railway config
└── README.md
```

## Quick Deploy

### 1. Push to GitHub

```bash
git init
git add main.py vectorize_new_fast.py requirements.txt railway.toml README.md
git commit -m "RAG pipeline FastAPI"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### 2. Deploy on Railway

1. Go to [railway.app](https://railway.app)
2. **New Project** → **Deploy from GitHub repo**
3. Select your repository
4. Railway auto-detects and deploys

### 3. Environment Variables (in Railway dashboard)

```
QDRANT_HOST=hopper.proxy.rlwy.net
QDRANT_PORT=48447
QDRANT_API_KEY=your_api_key
AUTO_START_PIPELINE=false
```

Set `AUTO_START_PIPELINE=true` to auto-start on deploy.

## API Endpoints

Once deployed, your app will be at `https://your-app.railway.app`:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service info |
| `/health` | GET | Health check |
| `/status` | GET | Pipeline status & stats |
| `/start` | POST | Start vectorization |
| `/stop` | POST | Stop vectorization |

## Usage

### Check Status
```bash
curl https://your-app.railway.app/status
```

### Start Pipeline
```bash
curl -X POST https://your-app.railway.app/start
```

### Monitor Progress
```bash
# Poll status every 30 seconds
while true; do curl -s https://your-app.railway.app/status | jq; sleep 30; done
```

### Stop Pipeline
```bash
curl -X POST https://your-app.railway.app/stop
```

## Response Examples

### Status Response
```json
{
  "status": "running",
  "started_at": "2024-12-14T03:45:00Z",
  "last_update": "2024-12-14T03:50:00Z",
  "stats": {
    "decisions_fetched": 5000,
    "decisions_processed": 4800,
    "chunks_created": 12000,
    "chunks_uploaded": 11500,
    "errors": 2
  },
  "error": null
}
```

### Status Values
- `idle` - Not started
- `starting` - Initializing
- `running` - Processing data
- `stopping` - Stop requested
- `stopped` - Manually stopped
- `completed` - Finished successfully
- `error` - Failed with error

## Notes

- **Memory**: Needs ~2-4GB RAM for embedding model
- **Timeout**: Railway Pro has no timeout; Hobby has limits
- **Persistence**: Checkpoint file doesn't persist between deploys
- **Estimated Time**: ~8-12 hours for full 600k decisions

