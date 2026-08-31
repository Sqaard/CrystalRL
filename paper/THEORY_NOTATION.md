# Notation map — `sec_theory_neurips_with_visuals.tex` vs. `main.tex` (Draft v0.4)

Rule: every symbol in the theory section is either **reused** from Vanya's draft (same meaning,
same macro) or **new** (declared on first use in the section, listed here). No symbol is
redefined.

## Reused from `main.tex`

| Symbol | Meaning in v0.4 | Where defined | Theory-section use |
|---|---|---|---|
| `b_t` | regime posterior `P(bear | F_t)` | eq:belief (l.356) | the named belief, Gen 3 |
| `h_t` | policy's hidden state (64-d latent) | eq:intervene (l.275) | opaque learned state, Gen 1/2 |
| `a_t`, `\pi` | action, policy | eq:intervene, eq:tree | executed policy |
| `\operatorname{do}(\cdot)`, `b^{*}` | belief write | eq:write (l.399), eq:faith (l.461) | interventional probe, controls |
| `F`, `S`, `S\!t` | faithfulness, simulatability, stability | eq:faith / eq:simul / eq:stab | components of `\mathcal{I}(\pi)` and `\mathbf{C}(\pi)` |
| `\text{story}_K`, `K` | K-rule story, parsimony budget | eq:simul (l.464), §metrics l.457 | legibility budget in `D(K)` |
| `\mathcal{R}(\cdot)` | performance used for parity | eq:simul (l.465) | performance in `D(K)` and `\pi^{\star}` |
| `\Delta_{\text{MDL}}` | accuracy lost by a short story | eq:mdl (l.478) | accuracy-side counterpart of `D(K)` |
| `\widehat J`, `\delta_{\text{NI}}` | certified objective, non-inferiority margin | eq:ni (l.564) | instance of the refusal rule |
| `\E`, `\Prob`, `\ind` | macros | preamble l.14–16 | as-is |

## New in `sec_theory_neurips_with_visuals.tex`

| Symbol | Meaning | First use |
|---|---|---|
| `x_t` | observable state | §theory:four P1 |
| `g_t` | goal / constraint state | §theory:four P1 |
| `\hat\pi` | post-hoc surrogate policy | §theory:four P1 |
| `q_\phi` | post-hoc map `h_t \mapsto b_t` | §theory:generations P2 |
| `B(\cdot)` | belief constructor `x_{\le t} \mapsto b_t` (Gen 3) | §theory:generations P3 |
| `\pi_{\text{leg}}` | born-legible policy (the executed small program) | §theory:generations P3 |
| `\mathcal{I}(\pi) = (F, S, S\!t, C)` | four-dimensional interpretability profile | §theory:four P2 |
| `C` | controllability (new fourth dimension) | §theory:four P2 |
| `F_{\text{obs}}` | observational faithfulness `\Prob[q(b_t) = a_t]` | §theory:probe P1 |
| `F_{\text{int}}` | interventional faithfulness; `\equiv` eq:faith | §theory:probe P2 |
| `\tau` | decision threshold on `b_t` | §theory:probe P3 |
| `b^{-}, b^{+}` | matched contrast values, `b^{-} < \tau < b^{+}` | §theory:probe P3 |
| `I`, `T` | intervention; intended target set | §theory:control P1 |
| `\Delta_T(I)` | target effect | §theory:control P1 |
| `B_{\neg T}(I)` | blast radius (non-target effect) | §theory:control P2 |
| `\tau_s, \tau_b, \tau_u` | support / blast-radius / uncertainty thresholds for refusal | §theory:control P3 |
| `U(I)` | uncertainty of an intervention's effect | §theory:control P3 |
| `\pi^{\star}, \pi^{\star}_K` | unrestricted optimum; optimum under budget `K` | §theory:frontier P1 |
| `L(\pi)` | policy description length in leaves, rules, or bits | §theory:frontier P1 |
| `D(K)` | legibility deficit `\mathcal{R}(\pi^{\star}) - \mathcal{R}(\pi^{\star}_K)` | §theory:frontier P1 |
| `\rho(\pi)` | risk functional (named `\rho` to avoid clashing with `\mathcal{R}`) | §theory:finance P2 |
| `\lambda_r, \lambda_c, \lambda_u` | weights on risk, complexity, uncertainty | §theory:finance P2 |
| `\mathbf{C}(\pi)` | score vector `[F, S, S\!t]` | §theory:compare P1 |
| `\operatorname{Comparable}(\pi_i, \pi_j)` | comparability indicator | §theory:compare P2 |

## Environments

- `proposition` — not defined in v0.4 (no `amsthm`). `sec_theory_neurips_with_visuals.tex` guards with
  `\ifcsname proposition\endcsname\else\newtheorem{proposition}{Proposition}\fi`.
  Recommended preamble line for Vanya (optional): `\newtheorem{proposition}{Proposition}` after l.16.

## Labels introduced

- Sections: `sec:theory`, `sec:theory:four`, `sec:theory:generations`, `sec:theory:probe`,
  `sec:theory:control`, `sec:theory:frontier`, `sec:theory:finance`, `sec:theory:compare`.
- Equations: `eq:hierarchy`, `eq:profile`, `eq:nonimply`, `eq:gen1`, `eq:gen2`, `eq:gen3`,
  `eq:noimply`, `eq:probes`, `eq:threshold`, `eq:contrast`, `eq:target_blast`,
  `eq:actionable`, `eq:refuse`, `eq:deficit`, `eq:readings`, `eq:objective`, `eq:vector`,
  `eq:comparable`.
- Propositions: `prop:nonequiv`, `prop:integration`, `prop:probe`, `prop:control`,
  `prop:frontier`, `prop:governance`, `prop:comparable`.
- Figures: `fig:generations`, `fig:probe-control`, `fig:legibility-budget`.
