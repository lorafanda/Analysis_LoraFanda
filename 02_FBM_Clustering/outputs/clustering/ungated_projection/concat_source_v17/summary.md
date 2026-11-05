# The ungated electrodes projected onto the seven archetypes, cohort v17 (run 20261017_194301, nimare maps, loading 0.3)

Seen 2917 electrodes in 29 patients: 1587 gated, 1330 not. The rebuilt features match the run's X_train to 0.
Projection check on the gated electrodes: the argmax agrees with the run's for 100.0 % of them (100.0 % of those at loading >= 0.3); r between the projected and the fitted loading 1.00; variance explained 48.2 %. The projection is the pipeline's own held-out rule (lf_decompose.cv_rank_curve: non-negative least squares onto the fitted components).

| set | n | median loading | share >= 0.3 | share >= 0.5 | median r2 | share r2 >= 0.5 |
|---|---|---|---|---|---|---|
| gated, the run's own G | 1587 | 0.41 | 84 % | 27 % | 0.49 | 48 % |
| gated, projected | 1587 | 0.41 | 84 % | 27 % | 0.49 | 48 % |
| ungated, projected | 1330 | 0.40 | 87 % | 24 % | 0.34 | 20 % |
| gated, time-shifted null | 1587 | 0.51 | 97 % | 52 % | 0.20 | 2 % |
| ungated, time-shifted null | 1330 | 0.46 | 97 % | 40 % | 0.18 | 1 % |

| type | gated members (argmax) | of them at loading >= 0.3 | ungated at loading >= 0.3 | their patients | of them with r2 >= 0.5 | ungated at loading >= 0.5 | ungated argmax |
|---|---|---|---|---|---|---|---|
| c5 production | 200 | 172 | 12 | 7 | 0 | 1 | 19 |
| c0 beta decrease | 318 | 266 | 245 | 20 | 136 | 100 | 275 |
| c2 broadband decrease | 168 | 142 | 122 | 11 | 45 | 27 | 149 |
| c6 response-period low-frequency fall | 224 | 195 | 163 | 25 | 26 | 41 | 180 |
| c4 low-frequency rise | 334 | 291 | 248 | 24 | 23 | 50 | 271 |
| c1 audio-stimulus low-frequency | 143 | 114 | 218 | 23 | 26 | 63 | 253 |
| c3 sentence-stimulus low-frequency | 200 | 161 | 152 | 21 | 5 | 37 | 183 |

| largest share of bins above threshold | n | gated | median loading | share >= 0.3 | median r2 |
|---|---|---|---|---|---|
| 0 to 0.5 % | 80 | 25 % | 0.48 | 90 % | 0.41 |
| 0.5 to 1 % | 396 | 7 % | 0.42 | 89 % | 0.34 |
| 1 to 1.5 % | 502 | 4 % | 0.39 | 88 % | 0.33 |
| 1.5 to 2 % | 446 | 6 % | 0.40 | 87 % | 0.37 |
| 2 to 3 % | 631 | 100 % | 0.41 | 85 % | 0.43 |
| 3 to 5 % | 486 | 100 % | 0.40 | 82 % | 0.47 |
| 5 to 10 % | 247 | 100 % | 0.39 | 79 % | 0.53 |
| 10 to 100 % | 129 | 100 % | 0.41 | 88 % | 0.59 |

| type | gated members | ungated projected | r across maps | outlined in both | well-fit ungated (r2 >= 0.5) | r across maps | outlined in both | maps outlined, gated |
|---|---|---|---|---|---|---|---|---|
| c5 production | 172 | 12 | 0.70 | - | 0 | nan | - | listening, semantic, sentences, sentences_vs_resting_state, sentences_vs_words |
| c0 beta decrease | 266 | 245 | 0.76 | - | 136 | 0.78 | naming_vs_production | naming_vs_production, naming_vs_resting_state, pictures_vs_resting_state |
| c2 broadband decrease | 142 | 122 | 0.78 | - | 45 | 0.79 | - | - |
| c6 response-period low-frequency fall | 195 | 163 | 0.35 | - | 26 | -0.14 | - | - |
| c4 low-frequency rise | 291 | 248 | 0.48 | - | 23 | -0.01 | - | sentences |
| c1 audio-stimulus low-frequency | 114 | 218 | 0.17 | - | 26 | -0.42 | - | auditory, language_vs_resting_state, listening, listening_vs_resting_state, production, production_vs_motor, production_vs_resting_state, sentences_vs_resting_state, sentences_vs_words |
| c3 sentence-stimulus low-frequency | 161 | 152 | 0.54 | - | 5 | 0.08 | - | - |
