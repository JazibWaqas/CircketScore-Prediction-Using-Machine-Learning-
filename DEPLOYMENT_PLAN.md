# ODI Progressive Predictor - Deployment Plan & Handoff Summary

## 📌 Context for Cursor Agent
This document serves as the deployment handoff for the "Cricket Score Prediction Using Machine Learning" dashboard. The application has been polished, bugs fixed, and it is now ready for production deployment. Using configured MCP servers for Vercel and Render, please execute the deployment strategy outlined below.

---

## 🏗️ Project Architecture & Summary
*   **Application Type:** Full-stack machine learning dashboard for predicting International ODI cricket scores and fantasy team building.
*   **Frontend Setup:** React 18, Tailwind CSS, Framer Motion. Located in `dashboard/frontend/`.
*   **Backend Setup:** Python Flask, Pandas, XGBoost model. Located in `dashboard/backend/`.
*   **Data/Models:** Pre-trained ML models (`.pkl` files) loaded into memory on backend startup.

## ✨ Recent Pre-Launch Enhancements (Completed)
Before deployment, the following consumer-grade UX features were successfully implemented:
1.  **Model Streamlining:** Removed manual model selection (defaulted to the superior XGBoost model) to simplify UX.
2.  **Smart Button Validation:** "Predict Final Score" button is disabled intelligently until exactly 11 players are chosen per team, with dynamic helper text calculating remaining players needed.
3.  **Composition Intelligence:** Fixed strength calculation logic (preventing negative strength/economy bars) and improved team composition warnings.
4.  **Auto-scrolling:** Added a `useRef` to auto-scroll users down to the prediction results seamlessly upon API success.
5.  **Marketing & Setup:** Added an informational banner explaining the app's features (What-If scenarios, Live match context) directly onto the homepage with realistic ML accuracy ranges.

---

## 🚀 Deployment Strategy
We will employ a **Split Deployment Strategy** for maximum performance and reliability on free tiers.

1.  **Frontend (UI):** Deploy to **Vercel** (Edge CDN, blazing fast React serving).
2.  **Backend (API):** Deploy to **Render Web Services** (handles Python, Pandas, and ML model execution).

---

## 📋 Execution Plan for Cursor

Please execute the following steps via your Render and Vercel MCP integrations:

### Phase 1: Preparation & Configuration
1.  **Update Backend Requirements:**
    *   Ensure `gunicorn` is present in `dashboard/backend/requirements.txt`. Render requires a production WSGI server, not the Flask dev server.
2.  **Update Frontend API URLs:**
    *   Check how the frontend calls the backend (likely in `dashboard/frontend/src/utils/api.js` or `App.js`).
    *   Ensure the frontend is configured to point to the production backend URL (which we will get from Render) using environment variables (e.g., `REACT_APP_API_URL`) instead of `localhost:5002`.

### Phase 2: Render Backend Deployment
1.  **Target Directory:** `dashboard/backend`
2.  **Environment:** Python 3
3.  **Build Command:** `pip install -r requirements.txt`
4.  **Start Command:** `gunicorn app:app`
5.  **Environment Variables:** Add necessary environment variables (if any) and ensure the application binds to the `$PORT` automatically assigned by Render.
6.  *Crucial Check - Memory Limits:* Render Free Tier provides 512MB RAM. Ensure loading the `.pkl` files and `app.py` does not trigger OOM (Out of Memory) kills.

### Phase 3: Vercel Frontend Deployment
1.  **Target Directory:** `dashboard/frontend`
2.  **Framework Preset:** Create React App
3.  **Build Command:** `npm run build`
4.  **Output Directory:** `build`
5.  **Environment Variables:** Set `REACT_APP_API_URL` (or equivalent) to the Render URL generated in Phase 2.

### Phase 4: CORS Configuration (Post-Deployment)
1.  Once the Vercel app is live and has a URL, update the backend `dashboard/backend/config.py` (or environment variables on Render).
2.  Update `CORS_ORIGINS` to explicitly allow the production Vercel URL to prevent Cross-Origin Resource Sharing blocks.

**End Goal:** A fully live, production-grade URL that can be immediately linked on LinkedIn and a Resume, demonstrating full-stack ML engineering capability.
