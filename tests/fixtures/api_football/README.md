# API-Football fixture bundle

Real provider responses captured on 2026-11-08 from the API-Football
v3 free tier (100 requests/day, 10/minute). These are raw response
bodies, saved unchanged, for offline tests of the live refresh path.
No credential material of any kind is included or needed to use them.

## Files

- `fixtures_date_2026-11-07.json` — endpoint `fixtures`, parameters
  `date=2026-11-07`. The free plan rejects league queries for the
  2026 season and blocks the `last` parameter, so the day was fetched
  without a league filter; tests select the finished English Premier
  League (league 39) fixtures from the response.
- `events_1557463.json`, `events_1557457.json`, `events_1557465.json`
  — endpoint `fixtures/events`, parameter `fixture=<id>`, for Leeds
  vs Tottenham, Arsenal vs Hull City, and Nottingham Forest vs
  Manchester City (all 2026-11-07, all finished).
- `lineups_1557463.json`, `lineups_1557457.json`,
  `lineups_1557465.json` — endpoint `fixtures/lineups`, parameter
  `fixture=<id>`, for the same three fixtures.

Capture used 8 quota-counted requests on 2026-11-08 (one date query,
one league-plus-date query that returned a season-required error,
and six events/lineups queries); the provider reported 92 requests
remaining for the day after capture.
