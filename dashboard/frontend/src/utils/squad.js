/**
 * Squad-building helpers.
 *
 * The player database is broad (every name that ever appeared in the ODI data),
 * so an unranked list surfaces tail-enders above stars. These helpers rank by a
 * relevance score and pick a *balanced* XI rather than the top 11 batting
 * averages — which is what made the old default XI report "Bowlers: 0".
 */

const MIN_MATCHES = 20;

/** Higher is more "recognisable / reliable". Used for list ordering only. */
export const relevanceScore = (p) => {
  const matches = p.total_matches || 0;
  const bat = p.batting_avg || 0;
  const econ = p.bowling_economy || 0;
  // Sample-size confidence: a 10-match career shouldn't outrank a 200-match one.
  const experience = Math.min(1, matches / 100);
  const battingComponent = Math.min(bat, 60) / 60;
  // Lower economy is better; 3.5–7.5 maps onto 1–0.
  const bowlingComponent = econ > 0 ? Math.max(0, Math.min(1, (7.5 - econ) / 4)) : 0;
  const skill = Math.max(battingComponent, bowlingComponent * 0.9);
  return skill * 0.6 + experience * 0.4;
};

export const sortByRelevance = (list) =>
  [...list].sort((a, b) => relevanceScore(b) - relevanceScore(a));

/** True when a player has enough data behind them to be worth showing by default. */
export const isEstablished = (p) =>
  (p.total_matches || 0) >= MIN_MATCHES && ((p.batting_avg || 0) > 0 || (p.bowling_economy || 0) > 0);

/**
 * Pick a realistic XI for a country: 5 batsmen, 2 all-rounders, 4 bowlers,
 * falling back to best-available when a country lacks depth in a role.
 */
export const buildBalancedXI = (players, countryName) => {
  const pool = players.filter(
    (p) => (p.country || '').toLowerCase() === (countryName || '').toLowerCase()
  );
  if (pool.length === 0) return [];

  const ranked = sortByRelevance(pool);
  const established = ranked.filter(isEstablished);
  const source = established.length >= 11 ? established : ranked;

  const byRole = (role) => source.filter((p) => (p.player_role || 'All-rounder') === role);
  const quota = [
    ['Batsman', 5],
    ['All-rounder', 2],
    ['Bowler', 4],
  ];

  const picked = [];
  const taken = new Set();
  quota.forEach(([role, n]) => {
    byRole(role)
      .filter((p) => !taken.has(p.player_id))
      .slice(0, n)
      .forEach((p) => {
        picked.push(p);
        taken.add(p.player_id);
      });
  });

  // Top up from best available if any role ran short.
  for (const p of source) {
    if (picked.length >= 11) break;
    if (!taken.has(p.player_id)) {
      picked.push(p);
      taken.add(p.player_id);
    }
  }

  // Order the XI so the batsmen open and the bowlers bat last.
  const order = { Batsman: 0, 'All-rounder': 1, Bowler: 2 };
  return picked
    .slice(0, 11)
    .sort((a, b) => {
      const ra = order[a.player_role] ?? 1;
      const rb = order[b.player_role] ?? 1;
      if (ra !== rb) return ra - rb;
      return (b.batting_avg || 0) - (a.batting_avg || 0);
    })
    .map((p) => ({ id: p.player_id, name: p.player_name, country: p.country }));
};

/** Aggregate composition counts used by the team-analysis panels. */
export const squadComposition = (squadPlayers, allPlayers) => {
  const lookup = new Map(allPlayers.map((p) => [p.player_id, p]));
  const detailed = squadPlayers.map((s) => lookup.get(s.id)).filter(Boolean);
  const count = (role) => detailed.filter((p) => (p.player_role || 'All-rounder') === role).length;
  const battingAvgs = detailed.map((p) => p.batting_avg || 0);
  const economies = detailed.map((p) => p.bowling_economy || 0).filter((e) => e > 0);
  return {
    detailed,
    batsmen: count('Batsman'),
    allRounders: count('All-rounder'),
    bowlers: count('Bowler'),
    avgBatting: battingAvgs.length ? battingAvgs.reduce((a, b) => a + b, 0) / battingAvgs.length : 0,
    avgEconomy: economies.length ? economies.reduce((a, b) => a + b, 0) / economies.length : 0,
    elite: detailed.filter((p) => (p.batting_avg || 0) >= 40).length,
    depth: detailed.filter((p) => (p.batting_avg || 0) >= 30).length,
  };
};

/** Curated one-click scenarios; venue/team names are resolved against live data. */
export const PRESETS = [
  {
    id: 'wc-final',
    label: 'India vs Australia',
    sub: '198/4 after 35 overs',
    teamA: 'India',
    teamB: 'Australia',
    venuePref: ['Narendra Modi Stadium', 'Wankhede', 'Eden Gardens'],
    scenario: { current_score: 198, wickets_fallen: 4, overs: 35, runs_last_10: 62 },
  },
  {
    id: 'pak-ind',
    label: 'Pakistan vs India',
    sub: '234/5 after 40 overs',
    teamA: 'Pakistan',
    teamB: 'India',
    venuePref: ['Dubai International', 'Arun Jaitley', 'Eden Gardens'],
    scenario: { current_score: 234, wickets_fallen: 5, overs: 40, runs_last_10: 46 },
  },
  {
    id: 'eng-nz',
    label: 'England vs New Zealand',
    sub: '78/1 after 12 overs',
    teamA: 'England',
    teamB: 'New Zealand',
    venuePref: ["Lord's", 'Trent Bridge', 'The Oval'],
    scenario: { current_score: 78, wickets_fallen: 1, overs: 12, runs_last_10: 68 },
  },
];

/**
 * The venue table holds 300+ grounds, most of them minor and carrying the
 * global fallback par score. Showing the whole list makes the control feel
 * like raw data, so major international venues are surfaced first and
 * duplicate "Ground" / "Ground, City" pairs are collapsed.
 */
const MAJOR_VENUES = [
  'Melbourne Cricket Ground', 'Sydney Cricket Ground', 'Adelaide Oval', 'Bellerive Oval',
  'The Gabba', 'W.A.C.A.', 'Perth Stadium', 'Eden Gardens', 'Wankhede Stadium',
  'M Chinnaswamy Stadium', 'Arun Jaitley Stadium', 'Narendra Modi Stadium',
  'MA Chidambaram Stadium', 'Rajiv Gandhi International Stadium', "Lord's", 'The Oval',
  'Trent Bridge', 'Edgbaston', 'Headingley', 'Old Trafford', 'Sophia Gardens',
  'Eden Park', 'Basin Reserve', 'Hagley Oval', 'Bay Oval', 'Seddon Park',
  'Newlands', 'The Wanderers Stadium', 'SuperSport Park', 'Kingsmead',
  'Gaddafi Stadium', 'National Stadium', 'Dubai International Cricket Stadium',
  'Sharjah Cricket Stadium', 'Sheikh Zayed Stadium', 'R Premadasa Stadium',
  'Pallekele International Cricket Stadium', 'Galle International Stadium',
  'Kensington Oval', 'Queen’s Park Oval', 'Sabina Park', 'Providence Stadium',
  'Shere Bangla National Stadium', 'Zahur Ahmed Chowdhury Stadium', 'Harare Sports Club',
];

const isMajor = (name) =>
  MAJOR_VENUES.some((m) => (name || '').toLowerCase().includes(m.toLowerCase()));

/** Ordered, de-duplicated venue list for the picker. */
export const curateVenues = (venues) => {
  const seen = new Map();
  for (const v of venues) {
    // "Eden Gardens" and "Eden Gardens, Kolkata" are the same ground.
    const key = (v.venue_name || '').split(',')[0].trim().toLowerCase();
    const prev = seen.get(key);
    // Prefer the shorter, cleaner label; prefer a real par score over the fallback.
    if (
      !prev ||
      (v.avg_score !== 244 && prev.avg_score === 244) ||
      (v.venue_name.length < prev.venue_name.length && (v.avg_score !== 244) === (prev.avg_score !== 244))
    ) {
      seen.set(key, v);
    }
  }
  const list = [...seen.values()];
  const major = list.filter((v) => isMajor(v.venue_name)).sort((a, b) => a.venue_name.localeCompare(b.venue_name));
  const rest = list.filter((v) => !isMajor(v.venue_name)).sort((a, b) => a.venue_name.localeCompare(b.venue_name));
  return { major, rest };
};

export const resolveVenue = (venues, prefs) => {
  for (const pref of prefs) {
    const hit = venues.find((v) => (v.venue_name || '').toLowerCase().includes(pref.toLowerCase()));
    if (hit) return hit;
  }
  return venues[0];
};
