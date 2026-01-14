# CausalPy Inference API

A FastAPI service that runs synthetic control inference for geo experiments. This API is designed to work with the GeoLift Platform web application.

## Features

- Bayesian synthetic control methodology
- Constrained optimization for donor weights (non-negative, sum to one)
- Bootstrap confidence intervals
- Pre-treatment fit metrics (R²)
- Treatment effect estimation (ATT, percent lift)

## Quick Deploy to Railway (Recommended)

1. Fork this repository to your GitHub account
2. Go to [railway.app](https://railway.app) and sign up/login
3. Click "New Project" → "Deploy from GitHub repo"
4. Select this repository
5. Railway will auto-detect Python and deploy
6. Copy the generated URL (e.g., `https://causalpy-api-production.up.railway.app`)
7. Add this URL to your GeoLift Platform as `CAUSALPY_API_URL`

## Quick Deploy to Render

1. Fork this repository to your GitHub account
2. Go to [render.com](https://render.com) and sign up/login
3. Click "New" → "Web Service"
4. Connect your GitHub and select this repository
5. Render will auto-detect settings from `render.yaml`
6. Copy the generated URL
7. Add this URL to your GeoLift Platform as `CAUSALPY_API_URL`

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python main.py
# or
uvicorn main:app --reload --port 8000
```

## API Endpoints

### Health Check
```
GET /health
```
Returns `{"status": "healthy"}` if the service is running.

### Run Inference
```
POST /inference
Content-Type: application/json

{
  "csv_data": "date,market,sales\n2024-01-01,new_york,1000\n...",
  "treated_markets": ["new_york", "los_angeles"],
  "treatment_start_date": "2024-06-01",
  "treatment_end_date": "2024-06-30"
}
```

**Response:**
```json
{
  "att": 1500.50,
  "percent_lift": 15.5,
  "r_squared": 0.85,
  "p_value": 0.02,
  "ci_lower": 1200.00,
  "ci_upper": 1800.00,
  "synthetic_control": [...],
  "observed_treated": [...],
  "dates": [...],
  "weights": {"chicago": 0.4, "houston": 0.3, ...}
}
```

## Environment Variables

No environment variables required. The service runs on the port specified by the `PORT` environment variable (defaults to 8000).

## Technical Details

The synthetic control method:
1. Finds optimal weights for control markets to match pre-treatment trends
2. Uses constrained optimization (SLSQP) with non-negative weights summing to 1
3. Calculates treatment effect as difference between observed and synthetic post-treatment
4. Provides bootstrap confidence intervals and p-values

## License

MIT
