## Install

- install `uv`
- `source setup.sh`

## TODO

- [x] prepare training dataset (04/24/2025)
    - Input: ESM encodings for (Ag, Hchain, Lchain)
    - Output: binding affinity measures ($\Delta G$ or $K_d$)

- [ ] do some feature engineering
    - Currently, to deal with different sequence lengths, `output.hidden_states.mean(-2)` approach. This can be improved by applying convolutional several times followed by global pooling, as suggested by Gemini.


## Rosetta relaxation and scoring