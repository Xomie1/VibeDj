# How Spotify's DJ works: what's public

Researched 12 Oct 2026. Spotify has NOT published the DJ's recommender internals.
What's below is from Spotify's own product posts, plus Spotify's general
recommendation research, which may or may not be what DJ uses.

## Confirmed by Spotify (product posts)
- **Three pieces:** Spotify's personalisation tech picks the music, OpenAI generative AI helps write commentary, and a voice model from Spotify's Sonantic acquisition speaks it. (Newsroom, 22 Feb 2023)
- **Commentary is grounded in editors:** music editors (genre experts) supply the facts; the AI scales that knowledge. A "writers' room" of music experts and scriptwriters shaped the commentary. (Newsroom 2023; TechCrunch, 13 Mar 2023)
- **Voice:** modelled on Xavier "X" Jernigan, trained from studio recordings he made in different emotions; now fully AI-generated. (TechCrunch 2023)
- **What it plays:** new releases it expects you'll like, mixed with old favourites, sometimes songs you haven't played in years. (Newsroom 2023)
- **Steering:** the lineup refreshes based on feedback; tapping the DJ button switches to a different genre, artist or mood. A short tap skips to the next "section". (Newsroom 2023, May 2025)
- **Requests:** voice requests (May 2025) and text requests + Spanish + 3 personalised prompt suggestions (Oct 2025). Requests combine genre, mood, artist, activity, e.g. "cry-in-the-car songs". The session updates using the request plus listening history and preferences. (Newsroom)
- **Engagement in beta:** on days people used DJ, 25% of their listening went to it; over half of first-timers came back the next day. (Newsroom, 8 Mar 2023)

## Spotify research (general, not confirmed for DJ)
- **Skips are a core implicit signal.** Spotify Research's Music Streaming Sessions Dataset (Brost, Mehrotra, Jehan, 2019; 160M sessions) treats skips and context switches as key implicit feedback, with graded skip labels (very brief, brief, most played, full). Skips in a session correlate with earlier skips.
- **Explore/exploit with explanations.** Spotify's BaRT contextual bandit (McInerney et al., RecSys 2018; patent US11301513, Spotify AB) predicts satisfaction for item + explanation + context, uses conditional epsilon-greedy exploration, and trains counterfactually with propensity scores. Used for Home ranking; DJ use is unconfirmed.

## Sources
- https://newsroom.spotify.com/2023-02-22/spotify-debuts-a-new-ai-dj-right-in-your-pocket/
- https://newsroom.spotify.com/tag/the-dj (Behind the Scenes, 8 Mar 2023)
- https://techcrunch.com/2023/03/13/spotifys-new-dj-feature-is-the-first-step-into-the-streamers-ai-powered-future
- https://newsroom.spotify.com/2025-05-13/dj-voice-requests/
- https://newsroom.spotify.com/tag/interactive-dj
- https://ar5iv.labs.arxiv.org/html/1901.09851
- https://patents.google.com/patent/US11301513
