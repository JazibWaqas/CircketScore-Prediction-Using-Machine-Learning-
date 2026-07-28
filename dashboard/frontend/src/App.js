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
import LoadingSpinner from './components/LoadingSpinner';
import api from './utils/api';
import { PRESETS, buildBalancedXI, resolveVenue } from './utils/squad';

const EMPTY_TEAM = { team_id: null, team_name: '', players: [] };

function App() {
  const [teams, setTeams] = useState([]);
  const [players, setPlayers] = useState([]);
  const [venues, setVenues] = useState([]);
  const [loading, setLoading] = useState(true);
  const [predicting, setPredicting] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);

  const [teamA, setTeamA] = useState(EMPTY_TEAM);
  const [teamB, setTeamB] = useState(EMPTY_TEAM);
  const [whatIfAllPlayers, setWhatIfAllPlayers] = useState(false);
  const [activePreset, setActivePreset] = useState(null);

  const [matchScenario, setMatchScenario] = useState({
    venue: '',
    venue_avg_score: 250,
    current_score: '',
    wickets_fallen: '',
    overs: '',
    runs_last_10: '',
    batsman_1: '',
    batsman_2: '',
  });

  const predictorRef = useRef(null);
  const resultRef = useRef(null);

  useEffect(() => {
    const load = async () => {
      try {
        const [t, p, v] = await Promise.all([api.getTeams(), api.getPlayers(), api.getVenues()]);
        const allTeams = t.data.teams;
        const allPlayers = p.data.players;
        const allVenues = v.data.venues;

        setTeams(allTeams);
        setPlayers(allPlayers);
        setVenues(allVenues);

        // Seed a full scenario so the app never renders an empty shell. An empty
        // two-column form is the worst-looking state and it was the landing state.
        const seed = PRESETS[1];
        const find = (n) => allTeams.find((x) => (x.team_name || '').toLowerCase() === n.toLowerCase());
        const a = find(seed.teamA);
        const b = find(seed.teamB);
        if (a && b) {
          setTeamA({ team_id: a.team_id, team_name: a.team_name, players: buildBalancedXI(allPlayers, a.team_name) });
          setTeamB({ team_id: b.team_id, team_name: b.team_name, players: buildBalancedXI(allPlayers, b.team_name) });
          const venue = resolveVenue(allVenues, seed.venuePref);
          setMatchScenario({
            venue: venue?.venue_name || '',
            venue_avg_score: venue?.avg_score || 250,
            ...seed.scenario,
            batsman_1: '',
            batsman_2: '',
          });
          setActivePreset(seed.id);
        }
      } catch (err) {
        setError('Could not reach the prediction API. It may still be waking up, so try again in a moment.');
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  const handleTeamSelect = (type, id, name) => {
    const next = { team_id: id, team_name: name, players: [] };
    if (type === 'A') setTeamA(next);
    else setTeamB(next);
    setPrediction(null);
  };

  const handlePlayerSelect = (type, id, name, country) => {
    const add = (prev) =>
      prev.players.length < 11 ? { ...prev, players: [...prev.players, { id, name, country }] } : prev;
    if (type === 'A') setTeamA(add);
    else setTeamB(add);
  };

  const handleRemovePlayer = (type, id) => {
    const drop = (prev) => ({ ...prev, players: prev.players.filter((p) => p.id !== id) });
    if (type === 'A') setTeamA(drop);
    else setTeamB(drop);
  };

  const handleAutoFill = (type, xi) => {
    if (type === 'A') setTeamA((prev) => ({ ...prev, players: xi }));
    else setTeamB((prev) => ({ ...prev, players: xi }));
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
    try {
      const res = await api.predict(buildRequest(teamA.players));
      setPrediction(res.data);
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: 'smooth', block: 'center' }), 120);
    } catch (err) {
      setError(err.response?.data?.error || 'Prediction failed. Please try again.');
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
    setPrediction(null);
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

  if (loading) {
    return (
      <div className="min-h-screen bg-dark-bg">
        <Header />
        <LoadingSpinner />
      </div>
    );
  }

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
