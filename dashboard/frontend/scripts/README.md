# Bundled data

`src/data/snapshot.json` and `src/data/presetPredictions.json` ship with the
build so the page renders a complete, already-predicted scenario on first paint
instead of waiting on a cold API container.

Regenerate both after changing the model, the player database, or the presets in
`src/utils/squad.js`:

    python scripts/generate_data.py

The API must be awake first (the free tier sleeps, and the first request can take
around 45 seconds):

    curl https://cricket-score-predictor-api.onrender.com/api/health
