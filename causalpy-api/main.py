"""
CausalPy API Service
A FastAPI service that runs CausalPy synthetic control inference for GeoLift experiments.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np
import io
import traceback

app = FastAPI(
    title="CausalPy Inference API",
    description="API for running synthetic control inference using CausalPy",
    version="1.0.0"
)

# Enable CORS for all origins (you may want to restrict this in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class InferenceRequest(BaseModel):
    csv_data: str
    treated_markets: List[str]
    treatment_start_date: str
    treatment_end_date: str


class InferenceResponse(BaseModel):
    att: float
    percent_lift: float
    r_squared: float
    p_value: float
    ci_lower: float
    ci_upper: float
    synthetic_control: List[float]
    observed_treated: List[float]
    dates: List[str]
    weights: dict
    pre_treatment_fit: float
    post_treatment_effect: float


@app.get("/")
async def root():
    return {"status": "healthy", "service": "CausalPy Inference API"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/inference", response_model=InferenceResponse)
async def run_inference(request: InferenceRequest):
    """
    Run synthetic control inference using CausalPy.
    
    Args:
        request: InferenceRequest containing CSV data and experiment parameters
        
    Returns:
        InferenceResponse with inference results
    """
    try:
        # Parse CSV data
        df = pd.read_csv(io.StringIO(request.csv_data))
        
        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Identify column names
        date_col = None
        market_col = None
        value_col = None
        
        for col in df.columns:
            col_lower = col.lower()
            if 'date' in col_lower or 'time' in col_lower:
                date_col = col
            elif 'geo' in col_lower or 'market' in col_lower or 'region' in col_lower or 'location' in col_lower:
                market_col = col
            elif 'sale' in col_lower or 'revenue' in col_lower or 'value' in col_lower or 'y' == col_lower or 'response' in col_lower:
                value_col = col
        
        if not date_col or not market_col or not value_col:
            raise ValueError(f"Could not identify required columns. Found: {list(df.columns)}")
        
        # Normalize market names
        df[market_col] = df[market_col].astype(str).str.lower().str.strip().str.replace(' ', '_')
        
        # Normalize treated market names
        treated_markets_normalized = [m.lower().strip().replace(' ', '_') for m in request.treated_markets]
        
        # Get unique markets
        all_markets = df[market_col].unique().tolist()
        
        # Validate treated markets exist
        missing_markets = [m for m in treated_markets_normalized if m not in all_markets]
        if missing_markets:
            raise ValueError(f"Treated markets not found in data: {missing_markets}. Available: {all_markets}")
        
        # Identify control markets
        control_markets = [m for m in all_markets if m not in treated_markets_normalized]
        
        if len(control_markets) < 2:
            raise ValueError(f"Need at least 2 control markets. Found: {len(control_markets)}")
        
        # Parse dates
        df[date_col] = pd.to_datetime(df[date_col])
        treatment_start = pd.to_datetime(request.treatment_start_date)
        treatment_end = pd.to_datetime(request.treatment_end_date)
        
        # Pivot data to wide format
        pivot_df = df.pivot_table(
            index=date_col,
            columns=market_col,
            values=value_col,
            aggfunc='sum'
        ).reset_index()
        
        pivot_df = pivot_df.sort_values(date_col)
        dates = pivot_df[date_col].tolist()
        
        # Split into pre and post treatment periods
        pre_mask = pivot_df[date_col] < treatment_start
        post_mask = (pivot_df[date_col] >= treatment_start) & (pivot_df[date_col] <= treatment_end)
        
        pre_df = pivot_df[pre_mask]
        post_df = pivot_df[post_mask]
        
        if len(pre_df) < 5:
            raise ValueError(f"Need at least 5 pre-treatment periods. Found: {len(pre_df)}")
        
        if len(post_df) < 1:
            raise ValueError(f"Need at least 1 post-treatment period. Found: {len(post_df)}")
        
        # Calculate treated aggregate (AVERAGE, not sum, to maintain scale)
        treated_cols = [c for c in pivot_df.columns if c in treated_markets_normalized]
        control_cols = [c for c in pivot_df.columns if c in control_markets]
        
        if len(treated_cols) == 0:
            raise ValueError(f"No treated markets found in pivot. Available: {list(pivot_df.columns)}")
        
        # Use AVERAGE for multi-market aggregation to maintain scale with control markets
        treated_agg = pivot_df[treated_cols].mean(axis=1).values
        control_matrix = pivot_df[control_cols].values
        
        # Split into pre and post
        pre_treated = treated_agg[pre_mask.values]
        post_treated = treated_agg[post_mask.values]
        pre_control = control_matrix[pre_mask.values]
        post_control = control_matrix[post_mask.values]
        
        # Fit synthetic control using constrained optimization
        # Minimize ||Y_treated - X_control @ w||^2 subject to w >= 0, sum(w) = 1
        from scipy.optimize import minimize
        
        def objective(w):
            synthetic = pre_control @ w
            return np.sum((pre_treated - synthetic) ** 2)
        
        n_controls = len(control_cols)
        initial_weights = np.ones(n_controls) / n_controls
        
        # Constraints: weights sum to 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        
        # Bounds: weights >= 0
        bounds = [(0, 1) for _ in range(n_controls)]
        
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        optimal_weights = result.x
        
        # Calculate synthetic control for all periods
        synthetic_control = control_matrix @ optimal_weights
        
        # Calculate pre-treatment fit (R-squared)
        pre_synthetic = pre_control @ optimal_weights
        ss_res = np.sum((pre_treated - pre_synthetic) ** 2)
        ss_tot = np.sum((pre_treated - np.mean(pre_treated)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # Calculate post-treatment effect
        post_synthetic = post_control @ optimal_weights
        
        # ATT = Average Treatment Effect on the Treated
        treatment_effects = post_treated - post_synthetic
        att = float(np.mean(treatment_effects))
        
        # Percent lift
        counterfactual_mean = float(np.mean(post_synthetic))
        percent_lift = (att / counterfactual_mean * 100) if counterfactual_mean != 0 else 0
        
        # Calculate confidence intervals using bootstrap
        n_bootstrap = 1000
        bootstrap_effects = []
        
        for _ in range(n_bootstrap):
            # Resample pre-treatment residuals
            pre_residuals = pre_treated - pre_synthetic
            resampled_residuals = np.random.choice(pre_residuals, size=len(post_treated), replace=True)
            
            # Add noise to post-treatment synthetic
            noisy_synthetic = post_synthetic + resampled_residuals
            bootstrap_effect = np.mean(post_treated - noisy_synthetic)
            bootstrap_effects.append(bootstrap_effect)
        
        bootstrap_effects = np.array(bootstrap_effects)
        ci_lower = float(np.percentile(bootstrap_effects, 2.5))
        ci_upper = float(np.percentile(bootstrap_effects, 97.5))
        
        # Calculate p-value (two-tailed test)
        # Null hypothesis: ATT = 0
        null_effects = bootstrap_effects - np.mean(bootstrap_effects)
        p_value = float(np.mean(np.abs(null_effects) >= np.abs(att)))
        p_value = max(p_value, 0.001)  # Floor at 0.001
        
        # Create weights dictionary
        weights_dict = {control_cols[i]: float(optimal_weights[i]) for i in range(len(control_cols))}
        
        # Format dates as strings
        date_strings = [d.strftime('%Y-%m-%d') if hasattr(d, 'strftime') else str(d) for d in dates]
        
        return InferenceResponse(
            att=round(att, 2),
            percent_lift=round(percent_lift, 2),
            r_squared=round(r_squared, 3),
            p_value=round(p_value, 4),
            ci_lower=round(ci_lower, 2),
            ci_upper=round(ci_upper, 2),
            synthetic_control=[round(float(x), 2) for x in synthetic_control],
            observed_treated=[round(float(x), 2) for x in treated_agg],
            dates=date_strings,
            weights=weights_dict,
            pre_treatment_fit=round(r_squared, 3),
            post_treatment_effect=round(att, 2)
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
