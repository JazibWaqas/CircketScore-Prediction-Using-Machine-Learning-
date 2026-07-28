import React, { useState, useEffect, useRef, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertCircle, RotateCcw, Sparkles, X } from 'lucide-react';

import Header from './components/Header';
import Hero from './components/Hero';
import TeamSelector from './components/TeamSelector';
import MatchScenario from './components/MatchScenario';
import PredictionDisplay from './components/PredictionDisplay';
import ImpactLab from './components/ImpactLab';
import HowItWorks from './components/HowItWorks';
import api from './utils/api';
import snapshot from './data/snapshot.json';
import presetPredictions from './data/presetPredictions.json';
import { PRESETS, buildBalancedXI, resolveVenue } from './utils/squad';

const EMPTY_TEAM = { team_id: null, team_name: '', players: [] };

/**
 * The reference data is bundled with the build, so the page renders complete on
 * first paint even while the API container is still waking from sleep. The live
 * API is only required for the prediction call itself.
 */
const seedScenario = () => {
  const preset = PRESETS[1];
  const a = snapshot.teams.find((t) => t.team_name === preset.teamA);
  const b = snapshot.teams.find((t) => t.team_name === preset.teamB);
  const venue = resolveVenue(snapshot.venues, preset.venuePref);
  return {
    teamA: a
      ? { team_id: a.team_id, team_name: a.team_name, players: buildBalancedXI(snapshot.players, a.team_name) }
      : EMPTY_TEAM,
    teamB: b
      ? { team_id: b.team_id, team_name: b.team_name, players: buildBalancedXI(snapshot.players, b.team_name) }
      : EMPTY_TEAM,
    scenario: {
      venue: venue?.venue_name || '',
      venue_avg_score: venue?.avg_score || 250,
      ...preset.scenario,
      batsman_1: '',
      batsman_2: '',
    },
    presetId: preset.id,
  };
};

const SEED = seedScenario();

function App() {
  const [teams, setTeams] = useState(snapshot.teams);
  const [players, setPlayers] = useState(snapshot.players);
  const [venues, setVenues] = useState(snapshot.venues);
  const [predicting, setPredicting] = useState(false);
  // Preset results are precomputed at build time, so the landing scenario shows
  // a real projection immediately instead of waiting on a cold container.
  const [prediction, setPrediction] = useState(presetPredictions[SEED.presetId] || null);
  const [error, setError] = useState(null);

  const [teamA, setTeamA] = useState(SEED.teamA);
  const [teamB, setTeamB] = useState(SEED.teamB);
  const [whatIfAllPlayers, setWhatIfAllPlayers] = useState(false);
  const [activePreset, setActivePreset] = useState(SEED.presetId);

  const [matchScenario, setMatchScenario] = useState(SEED.scenario);

  const predictorRef = useRef(null);
  const resultRef = useRef(null);

  // Wake the API immediately and refresh the reference data in the background.
  // Nothing here blocks the first paint; the page is already interactive.
  useEffect(() => {
    let cancelled = false;

    const warm = async () => {
      // Free-tier containers sleep, so the first request pays the spin-up cost.
      // Firing it on mount means the wait overlaps with the visitor reading.
      try {
        await api.health();
      } catch {
        /* the retry below covers a failed wake-up */
      }
      if (cancelled) return;

      try {
        const [t, p, v] = await Promise.all([api.getTeams(), api.getPlayers(), api.getVenues()]);
        if (cancelled) return;
        // Only adopt live data if it actually looks complete; otherwise the
        // bundled snapshot already on screen stays put.
        if (t.data?.teams?.length) setTeams(t.data.teams);
        if (p.data?.players?.length) setPlayers(p.data.players);
        if (v.data?.venues?.length) setVenues(v.data.venues);
      } catch {
        /* snapshot data remains in place */
      }
    };

    warm();
    return () => { cancelled = true; };
  }, []);

  const handleTeamSelect = (type, id, name) => {
    const next = { team_id: id, team_name: name, players: [] };
    if (type === 'A') setTeamA(next);
    else setTeamB(next);
    setPrediction(null);
    setActivePreset(null);
  };

  // Any squad change invalidates the shown projection, including the cached
  // preset results, so a stale number never sits above edited inputs.
  const invalidate = () => {
    setPrediction(null);
    setActivePreset(null);
  };

  const handlePlayerSelect = (type, id, name, country) => {
    const add = (prev) =>
      prev.players.length < 11 ? { ...prev, players: [...prev.players, { id, name, country }] } : prev;
    if (type === 'A') setTeamA(add);
    else setTeamB(add);
    invalidate();
  };

  const handleRemovePlayer = (type, id) => {
    const drop = (prev) => ({ ...prev, players: prev.players.filter((p) => p.id !== id) });
    if (type === 'A') setTeamA(drop);
    else setTeamB(drop);
    invalidate();
  };

  const handleAutoFill = (type, xi) => {
    if (type === 'A') setTeamA((prev) => ({ ...prev, players: xi }));
    else setTeamB((prev) => ({ ...prev, players: xi }));
    invalidate();
  };

  /** Build the API payload for an arbitrary batting XI (used by the Impact Lab too). */
  const buildRequest = useCallback(
    (battingPlayers) => ({
      batting_team_players: battingPlayers.map((p) => p.name),
      bowling_team_players: teamB.players.map((p) => p.name),
      venue: matchScenario.venue,
      venue_avg_score: matchScenario.venue_avg_score,
      current_score: Number(matchScenario.current_score) || 0,
      wickets_fallen: Number(matchScenario.wickets_fallen) || 0,
      balls_bowled: (Number(matchScenario.overs) || 0) * 6,
      runs_last_10_overs: Number(matchScenario.runs_last_10) || 0,
      batsman_1: matchScenario.batsman_1,
      batsman_2: matchScenario.batsman_2,
      model: 'xgboost',
    }),
    [teamB.players, matchScenario]
  );

  const ready = teamA.players.length === 11 && teamB.players.length === 11 && !!matchScenario.venue;

  const handlePredict = async () => {
    if (!ready) return;
    setPredicting(true);
    setError(null);
    const payload = buildRequest(teamA.players);

    // A first request against a sleeping container can time out. Retry once
    // before showing an error, since by then the instance is usually awake.
    const attempt = () => api.predict(payload);
    try {
      let res;
      try {
        res = await attempt();
      } catch (first) {
        if (first.response) throw first; // a real API error, not a cold start
        res = await attempt();
      }
      setPrediction(res.data);
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' }), 120);
    } catch (err) {
      setError(
        err.response?.data?.error ||
          'The prediction service is still starting up. Give it a few seconds and try again.'
      );
    } finally {
      setPredicting(false);
    }
  };

  /** One click: fills both XIs, venue and match state from a curated scenario. */
  const applyPreset = (preset) => {
    const findTeam = (name) =>
      teams.find((t) => (t.team_name || '').toLowerCase() === name.toLowerCase());

    const a = findTeam(preset.teamA);
    const b = findTeam(preset.teamB);
    if (!a || !b) {
      setError('Those teams are not available in the current dataset.');
      return;
    }

    setTeamA({ team_id: a.team_id, team_name: a.team_name, players: buildBalancedXI(players, a.team_name) });
    setTeamB({ team_id: b.team_id, team_name: b.team_name, players: buildBalancedXI(players, b.team_name) });

    const venue = resolveVenue(venues, preset.venuePref);
    setMatchScenario({
      venue: venue?.venue_name || '',
      venue_avg_score: venue?.avg_score || 250,
      ...preset.scenario,
      batsman_1: '',
      batsman_2: '',
    });

    setActivePreset(preset.id);
    // Cached result for this scenario, so switching presets stays instant.
    setPrediction(presetPredictions[preset.id] || null);
    setError(null);
    setTimeout(() => predictorRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 80);
  };

  const scrollToPredictor = () => {
    if (players.length && teams.length) applyPreset(PRESETS[1]);
    else predictorRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  /** Clear both XIs and the match state so a scenario can be built by hand. */
  const startFromScratch = () => {
    setTeamA(EMPTY_TEAM);
    setTeamB(EMPTY_TEAM);
    setMatchScenario({
      venue: '',
      venue_avg_score: 250,
      current_score: '',
      wickets_fallen: '',
      overs: '',
      runs_last_10: '',
      batsman_1: '',
      batsman_2: '',
    });
    setActivePreset(null);
    setPrediction(null);
    setError(null);
    setTimeout(() => predictorRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }), 80);
  };

  return (
    <div id="top" className="min-h-screen bg-dark-bg">
      <Header />
      <Hero onTryScenario={scrollToPredictor} />

      <main ref={predictorRef} id="predictor" className="mx-auto max-w-6xl px-6 py-8">
        {/* ---- Presets ---------------------------------------------- */}
        <div className="mb-4">
          <div className="mb-2 flex items-center gap-2">
            <Sparkles className="h-3.5 w-3.5 text-accent" />
            <span className="eyebrow">Load a live match situation</span>
            <button
              onClick={startFromScratch}
              className="ml-auto inline-flex items-center gap-1.5 text-[11px] text-dark-muted transition-colors hover:text-dark-text"
            >
              <RotateCcw className="h-3 w-3" />
              Start from scratch
            </button>
          </div>
          <div className="grid gap-2.5 sm:grid-cols-3">
            {PRESETS.map((p) => {
              const active = activePreset === p.id;
              return (
                <button
                  key={p.id}
                  onClick={() => applyPreset(p)}
                  aria-pressed={active}
                  className={`surface group px-3.5 py-2.5 text-left transition-colors ${
                    active ? '!border-accent/50 bg-accent/[0.06]' : 'hover:border-ink-500'
                  }`}
                >
                  <span
                    className={`block text-[13px] font-semibold ${
                      active ? 'text-accent' : 'text-white group-hover:text-accent'
                    }`}
                  >
                    {p.label}
                  </span>
                  <span className="mt-0.5 block text-[11px] text-dark-muted">{p.sub}</span>
                </button>
              );
            })}
          </div>
        </div>

        {/* ---- Error ------------------------------------------------- */}
        <AnimatePresence>
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              role="alert"
              className="mb-4 flex items-start gap-3 rounded-xl border border-cricket-red/30 bg-cricket-red/8 px-4 py-3"
            >
              <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-cricket-red" />
              <p className="flex-1 text-sm text-dark-text">{error}</p>
              <button onClick={() => setError(null)} aria-label="Dismiss" className="text-dark-muted hover:text-white">
                <X className="h-4 w-4" />
              </button>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ---- Scenario in, projection out — kept adjacent so changing an
                input never scrolls the answer off screen. ---------------- */}
        <MatchScenario
          scenario={matchScenario}
          onChange={(s) => { setMatchScenario(s); setPrediction(null); setActivePreset(null); }}
          venues={venues}
          battingPlayers={teamA.players}
          onPredict={handlePredict}
          predicting={predicting}
          ready={ready}
          teamA={teamA}
          teamB={teamB}
          whatIfAllPlayers={whatIfAllPlayers}
          onToggleWhatIf={setWhatIfAllPlayers}
        />

        <div ref={resultRef} className="mt-4">
          <PredictionDisplay
            prediction={prediction}
            scenario={matchScenario}
            predicting={predicting}
          />
        </div>

        <div className="mt-4 grid items-start gap-4 lg:grid-cols-2">
          <TeamSelector
            teamType="A"
            team={teamA}
            teams={teams}
            players={players}
            whatIfAllPlayers={whatIfAllPlayers}
            onTeamSelect={handleTeamSelect}
            onPlayerSelect={handlePlayerSelect}
            onRemovePlayer={handleRemovePlayer}
            onAutoFill={handleAutoFill}
          />

          <TeamSelector
            teamType="B"
            team={teamB}
            teams={teams}
            players={players}
            whatIfAllPlayers={whatIfAllPlayers}
            onTeamSelect={handleTeamSelect}
            onPlayerSelect={handlePlayerSelect}
            onRemovePlayer={handleRemovePlayer}
            onAutoFill={handleAutoFill}
          />
        </div>

        <div className="mt-4">
          <ImpactLab teamA={teamA} players={players} />
        </div>
      </main>

      <HowItWorks />
    </div>
  );
}

export default App;
