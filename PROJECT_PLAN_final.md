# MSc Thesis Project Plan — Bayesian Online Learning on Ushant AIS

_Last updated: 2025-08-10_

## 1) Core Objective
Implement and evaluate a **generalized Bayesian online learning** method with **adaptive** likelihood tempering on **irregular** AIS trajectories. Show that the method adapts **faster** and remains **well‑calibrated** under **non‑stationarity** (turns, slowdowns) than simple baselines.

## 2) Dataset (Ushant AIS)
- **Scope:** 6 months of AIS in the **Ushant TSS** (western entrance of English Channel).
- **Unit:** each file = **one trajectory** (a single vessel episode).
- **Columns:** `x` (lon °), `y` (lat °), `vx`, `vy` (knots from AIS), `t` (seconds **since start of that trajectory**).
- **Irregular sampling:** Δt ranges **5 s – 15 h**; ~95% gaps < **3 min**.
- **No absolute timestamp / vessel type**. Time is **relative per file**.
- **Why suitable:** strong **non‑stationarity** (cruise ↔ turn/stop), spikes/outliers, and **irregular Δt** → ideal to showcase **adaptive online Bayes**.

## 3) Method (BONE‑style)
### 3.1. Forecasting Task
We pose a prequential forecasting problem on **speed**:  
\( s_t = \sqrt{v_{x,t}^2 + v_{y,t}^2} \).  
At each step, we produce a **one‑step‑ahead probabilistic prediction**, score it, then **update online**. We handle **irregular sampling** by scaling state diffusion with **\(\Delta t\)**.  
**Later extension (optional):** In addition to speed, we will forecast the full velocity vector \([v_x, v_y]\), enabling short-term trajectory prediction. Predicted positions will be visualized on a map to demonstrate potential real-world applications in navigation and traffic monitoring.

### 3.2. Observation model (likelihood)
Start with **Gaussian**; consider **Student‑t** (\(\nu\) ~ 5–10) for robustness to AIS spikes/outliers. Predictive is used for **prequential log‑score**.

### 3.3. State evolution (Δt‑aware)
Random‑walk / OU‑like drift on latent speed with **variance proportional to Δt**:  
\( \theta_t = \theta_{t-1} + \varepsilon_t,\quad \varepsilon_t \sim \mathcal N(0, Q\,\Delta t) \).

### 3.4. Generalized Bayes (tempered update)
\[ p_t(\theta) \propto p_{t-1}(\theta)\, p(y_t\mid\theta)^{\lambda_t},\quad \lambda_{\min} \le \lambda_t \le \lambda_{\max}}. \]  
**Adaptive \(\lambda_t\):** a function of standardized residual (surprise). Calm periods → smaller \(\lambda_t\); shocks → larger \(\lambda_t\).

### 3.5. Auxiliary variable / Prior‑reset (PR) (BONE‑inspired)
Compute a **kinematic surprise** (e.g., from heading‑rate and speed change). Map to  
\( \nu_t = \sigma(a\,(z_t - b)) \in [0,1] \) and **mix priors**:  
carry‑over prior vs **reset prior** \((M_0, V_0)\). Combine with adaptive \(\lambda_t\).

## 4) Baselines
1. **Naive persistence** (predict last value).  
2. **Fixed‑\(\lambda\)** generalized Bayes.  
3. **Simple Kalman‑like** (no adaptation).  
4. *(Optional)* **C‑ACI** (constant discount).

## 5) Data processing & QC
- Parse **semicolon‑separated** files; attach `traj_id`.
- Features: **speed**, **heading = atan2(vy, vx)**, **Δt**.
- QC: cap implausible speeds (e.g., > 35–40 kts), drop duplicated timestamps, optional bbox clip.

## 6) Evaluation protocol (prequential, BONE‑style)
- **Dev/Test split** **by trajectory** to avoid leakage; tune only on dev, **report only** on held‑out test.
- **Per‑trajectory loop:** warm‑up window → then **test‑then‑train** at every step.
- **Reset** model state at **trajectory boundaries**.
- **Primary metric:** **prequential NLL / log‑score** (lower better).  
- **Also report:** MSE (secondary), **calibration** (PIT/reliability), and **adaptation‑lag** (median steps to recover after heading‑change events).
- *(Optional)* Paired Wilcoxon/t‑test across trajectories.

## 7) Experimental Protocol — Concrete Numbers
- **Split:** 70/30 **trajectory‑level** dev/test (no leakage).  
- **Warm‑up per trajectory:** Warm-up per trajectory: first **10%** of available points (min 5, max 100), used only for initialization; warm-up steps are excluded from metrics. Then strict **test-then-train.**.
- **Reset at boundaries:** re‑initialize state for each new trajectory.
- **Primary metric:** **prequential NLL**; also report MSE.
- **Calibration:** reliability/PIT plots.
- **Adaptation‑lag:** detect heading‑change events; report **median steps**.

## 8) Ablations
- **Adaptive‑\(\lambda\)** vs **Fixed‑\(\lambda\)** (same base model).  
- **Gaussian** vs **Student‑t** likelihoods.  
- **Δt‑aware** vs **Δt‑agnostic** diffusion.  
- **PR on** vs **PR off** (disable \(\nu_t\)).

## 9) Tuning & Reproducibility
- **Tune (dev only):** \(Q, R, \lambda_{\min}, \lambda_{\max}\), adaptive mapping (k, p), PR (a, b), warm‑up length.  
- **Search:** small grid or random search with fixed **seed**.
- **Config:** YAML under `configs/` (e.g., `configs/mvp.yaml`).  
- **Logging:** results to `results/` (CSV + figures).  
- **Determinism:** fix seeds; store config snapshot + git hash per run.

## 10) Implementation Strategy & Code Reuse

To maximize efficiency and build upon a proven foundation, this project will **not** reimplement the BONE algorithms from scratch. Instead, we will **adapt the existing codebase** provided in the `/notebooks` directory, which contains implementations of the reference paper's models.

- **Primary Task: Adaptation over Recreation:** The main focus will be on adapting the existing code to our specific use case. This involves:
    - Creating a new data loader for the `ushant_ais` dataset.
    - Configuring the chosen BONE model to work with the `speed` variable of our new dataset.
    - Modifying the evaluation scripts to match our defined protocol.

- **Library Usage for Baselines:** For baseline models not present in the reference codebase (like the Kalman Filter), we will use standard, well-tested Python libraries (e.g., `pykalman`, `statsmodels`) to ensure correctness and save development time.

- **Intermediate State Visualization:** We will add visualization code to the existing framework to plot internal model states during execution (e.g., `lambda_t`, surprise scores). This is critical for debugging and understanding model behavior.

- **Library & Style Consistency:** The choice of core numerical libraries (e.g., JAX, PyTorch, NumPy) and the overall coding style will be consistent with the patterns used in the reference notebooks in the `/notebooks` directory. This ensures seamless adaptation of the existing code.

## 11) Timeline (fast)
- **W1:** QC + EDA + MVP subset (200–500 trajectories); finalize target & likelihood.  
- **W2:** Implement Δt‑aware GBayes + baselines; first prequential results.  
- **W3:** Calibration + adaptation‑lag + ablations; draft figures.  
- **W4:** Polish, write‑up (6–8 pages) + optional 2D extension.

## 12) Deliverables
- Clean repo; reproducible code + configs; figures (speed vs pred, \(\lambda_t\), \(\nu_t\), avg‑NLL).  
- Tables of mean/median metrics; short write‑up with method & results.

## 13) One‑paragraph prompt (for another AI)
> You are given Ushant AIS trajectories and meeting notes. Implement an MVP: Δt‑aware generalized Bayes with **adaptive** \(\lambda_t\) and BONE‑style **prior‑reset** (aux weight \(\nu_t\)) to forecast **speed**. Compare to **fixed‑\(\lambda\)**, **naive**, and a **Kalman‑like** baseline. Evaluate with **prequential NLL** (primary), **calibration**, and **adaptation‑lag**. Provide clean Python code, configs, and export figures/tables for a short 6–8 page report.