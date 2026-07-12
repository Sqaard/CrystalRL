# Hello Crystal — the spoken script (v2: the self-evolution focus)

Companion to `pitch.tex` (15 slides, ~20 minutes + discussion). Stage directions in
[brackets]. The through-line of this version: **a transparent agent (CrystalRL) + a coding
agent on top (Heuristic Learning) = self-evolution that is safe because it is readable, and
personalization that is cheap because the loop does the tailoring.** The honesty machinery is
the safety rail, not the hero. The one language rule stands: **no term before its picture.**

---

## Slide 1 — Title

"Hi Joseph. I want to show you what I've been building — an agent that improves *itself* and
adapts itself to every investor — and why the whole construction stands or falls on
interpretability. Which is why the most important open seat on it is shaped like a
neuroscientist."

## Slide 2 — The dream, and the two walls

"Here's the dream, stated plainly: an investment agent that reads its own diagnostics,
rewrites its own machinery, and adapts itself to each person — someone says 'I can lose ten
percent, I have five years, I need a third more,' and the system tailors itself to *that*
sentence, without a human re-engineering it per client.

Everyone who tries this hits the same two walls. Wall one: you can't *read* the agent. A black
box that changes itself is evolution without eyes — you find out what it evolved into after it
costs money. Wall two: you can't trust the *judge*. In markets, the evaluator of any change is
a backtest — noisy, leaky, and gameable by the very optimization it's supposed to judge.

You'll recognize the shape of the problem: it's like being asked to study a brain you can
neither image nor stimulate, using behavioral scores the subject can fake. Neuroscience
answered with imaging, targeted interventions, and experimental controls. So did we — that's
essentially the whole project."

## Slide 3 — The thesis

"One sentence: **an agent may evolve itself exactly as far as it can be read.**

Three pieces. **CrystalRL** is the transparent agent — that's the 'Crystal': its policy is a
table, its memory is named probabilities you can *write to*, its objective is the user's own
contract. Interpretability here is not documentation, it's the load-bearing wall: it is *what
makes self-modification safe to allow*.

**Heuristic Learning** is the LLM coding agent on top: it proposes changes to the machinery
and tailors the system to each investor. Its imagination is unlimited; its authority is zero —
experimental controls decide what survives.

And the third piece — the calibration gates, the refusals, the pre-registration — that's the
safety rail. Important, but it exists so the first two can run at full speed.

So: transparency makes self-evolution *safe*; the coding agent makes it *cheap*."

## Slide 4 — PAUSE

[Stop. Look at him. Give it a real silence.]

"Before I go further — what are you getting from this? Tell it back in your own words."

[Let him talk. Note what he latched onto — that's what to build the collaboration around.
This habit is from Eisner's 'how to work with a professor'; we'll use it on each other weekly.]

## Slide 5 — The system in one picture

"The map. The core, left to right, is CrystalRL: a user's contract, a readable policy, careful
execution. Feeding it from below: the scenario engine with its self-calibration harness, and
the Heuristic-Learning loop — the coding agent under controls. At the far right, the
certification boundary — the safety rail that currently says 'refuse' by design.

For you: a subject you can image *and* stimulate, an experimenter proposing interventions, and
a review board with teeth."

## Slide 6 — CrystalRL: read and write

"Let me make 'transparent' concrete, because I mean something measurable.

Reading: the champion policy is literally a table on two axes — how many years are left, and am
I ahead of or behind my goal. It reads out loud: *behind the goal — take all the risk you're
allowed; at the goal — lock it in.* An eight-leaf decision tree reproduces the whole thing at
ninety-two percent. Not electrodes into a black box and guessing from spikes — a complete
receptive-field atlas.

Writing — this is the part I most want your reaction to. The agent's memory is *named*:
'probability of a bear regime' is an actual probability, and `SET_BELIEF(bear)=0.8` is a
command the policy must obey. Write fidelity is one hundred percent, and a nudge of about two
nats slides the behavior between named codes. It's optogenetics for a policy: not just
observing the state — setting it, and watching behavior follow.

And we don't assert transparency, we audit it — the chart: naming faithfulness one point zero,
seed stability one point zero, rule re-discovery three out of three, simulatability
ninety-two. Forcing the explanation to be short costs zero accuracy.

One honest caveat, and it matters for you: that ceiling is on *today's* substrate. On harder
substrates we certified a real *tension* between returns and legibility — making the agent more
readable cost performance, provably. That frontier is exactly the open research seat."

## Slide 7 — Heuristic Learning: the coding agent on top

"Now the evolution layer. The LLM agent reads the logbook and the diagnostics and proposes
typed changes to the machinery — new action menus, new scenario members, new solver knobs.
Unlimited imagination, zero authority.

Authority sits in the controls, and you know every one of them by another name: a
matched-random control — sham stimulation; a wrong-direction control — the inverted stimulus
must hurt; half-dose — no dose-response, no mechanism; out-of-time — replication in another
regime; and one read on held-out data, opened once.

Why so paranoid? FunSearch-style loops work because their evaluator is exact — a theorem
checker can't be fooled. Our judge is a backtest, which can. When the judge can't be trusted,
controls decide, not scores.

And here's where the two halves lock together: because the *agent* is readable, every accepted
change is inspectable — a diff on a table, a shift in a named belief. Evolution with eyes."

## Slide 8 — The track record

"Does it actually evolve? Three panels, all from our ledger.

Left: on the synthetic polygon — our controlled arena — the loop's own skill went from
seventy-seven percent, hand-built, to ninety-six when the LLM-proposer generation took over
proposing.

Middle: on real market data, under the hard gate. The first *certified* real-data policy came
through with seven accepts and zero placebo accepts — and notice what the output is: a
*readable sentence*. 'If the probability of a bear regime exceeds zero point six six, cut
exposure to zero point seven four.' That's what a certified self-improvement looks like here —
not a weight update, a sentence.

Right: we then handed the certified rule to a fully autonomous run — the loop proposing,
testing, certifying on its own for a whole lifetime. Forty proposals, four certified
refinements, and the thing I find most satisfying: its evidence budget *grew*, from 0.20 to
0.40. Certified wins refill the budget that experiments spend — it's self-sustaining, under
gates.

And it's been attacked: red-teamers broke the loop twice — found a way to derive hold-out
windows once, found an unwired promotion path once. Both fixed, and the fixes verified by the
attacker's own scripts. A loop that has survived being broken is worth more than one that's
never been tried."

## Slide 9 — Adaptation to each user

"Now personalization — the product half of self-evolution.

Every user is a *contract*: what you can lose, what you can stand to lose, and what you're
trying to reach — capacity, tolerance, goal, kept strictly separate. Pain threshold versus pain
tolerance: threshold is physiology, and no amount of bravery moves it. Capacity is a hard
filter that the evolution itself is not allowed to relax — that's a constraint on the *loop*,
not just on the portfolio. If the goal is infeasible, the system refuses and offers the honest
levers: time, contributions, a smaller goal. Never more risk.

Then the loop runs *per profile*. Five investors, one machinery — five differently-evolved
menus. The chart: sixty-five cycles against a no-agent baseline; fifteen changes accepted, all
fifteen confirmed on held-out data. And look where the gains land: the conservative profile —
the user the base system served *worst*, one percent goal probability — ends at fifty-two
percent, at an unchanged risk budget. The machine adapted itself to the user it was failing.
That's the product, in one curve."

## Slide 10 — The safety rail

"Briefly — the rail that lets all this run. We backtest the *promises themselves*: stand at
every past date, quote what the system would have promised, replay the same policy on what
actually happened. Nine hundred thirty-six cells, two markets. US: five of five profiles keep
the promise at least eighty percent of the time; the drawdown promises held in one hundred
percent of all cells, both markets. China: one of five — reported separately, never averaged,
all three failures diagnosed with fixes queued.

The engine grades its own calibration — seven of ten gates green, so every number still wears
an UNCALIBRATED label, and the certification layer refuses to bless the system, by design, with
dated unlocks in 2027 and 2029. Pre-registration, published nulls, no averaging across species
— the replication-crisis lessons, as code."

## Slide 11 — The graveyard

"Quickly, what died — because this is what makes the survivors believable. A famous valuation
indicator: falsified, reverted. The reinforcement-learning challenger: loses to the readable
table — reported. A Nasdaq-tracking sleeve that dominated every Chinese profile: red-flagged
*by the agent itself*, because its edge lives in one untestable decade. The conservative
promise failed its first backtest at 48.5 percent — and the diagnosis became a structural fix
that repaired exactly the failing cells, 48.5 to 84.2. And on hard substrates, legibility
provably isn't free — that certified tension is your frontier, not our embarrassment.

A system that demonstrably kills its own ideas is the one whose self-modifications you can
trust."

## Slide 12 — The investor view

"Why this becomes a company. First: personalization at software cost. Today a tailored
strategy means a human quant per client segment; here the coding agent does the tailoring —
five profiles already, menus evolved per user, every change certified. Second: a transparent
agent is an auditable agent — policies as tables, beliefs named and writable — an audit
standard competitors don't have, in a market where regulators demand explainability. Third:
self-evolution compounds — the autonomous run already funds its own experiments; the machine
improves while we sleep, under gates. Fourth, it's not all future tense: one certified
defensive rule runs live on a paper track — Sharpe 1.46 versus 1.39, max drawdown cut from
minus 15.5 to minus 12.9 percent, on 629 days it had never seen. And all of it with dated,
auditable credibility: eighty forecasts locked before realization, scoring reads in 2027 and
2029."

## Slide 13 — The six-month program

"Now your seat — and I want to pitch it as a research program, not a task list, because this is
six-plus months of work and you should own its shape.

The frame: you'd be the neuroscientist of an artificial subject — read-write access to every
synapse, behavior reproducible by seed, interventions free and ethical. With one twist that
makes it genuinely novel: the subject *changes itself*. So interpretability can't be a one-time
autopsy — it has to keep up with evolution.

Month one — build the measuring instruments. Simulatability psychophysics: can a human predict
the policy's action in cells they haven't seen? — literally your methods, on humans. And harden
our ceiling metrics: do they survive new seeds, new substrates, new profiles, or were we
measuring on easy ground?

Months two and three — the atlas. An *ethogram* of the agent: segment its behavior into
recurring motifs, across all five investor profiles. Belief faithfulness: does 'P(bear)' encode
what its name claims? And map the write-protocol — SET_BELIEF as a proper optogenetics
protocol: dose curves, response mapping.

Months three to five — your own loops. You get your own coding agent and you run
Heuristic-Learning loops on the frontier where legibility is *not* free — pushing the certified
return-versus-legibility frontier outward. Your proposals, our gates.

Months five, six, and beyond — the explanation gate. Every 'why' this product ever tells a user
must pass faithfulness tests that *you* own. A standing red-team seat on all our explanations —
the dead-salmon chair. Co-authorship on Hello Crystal, and you're the natural lead for the
methods paper — working title: *an ethogram of an artificial investor*."

## Slide 14 — How we'll work

"Practicalities — borrowed from Eisner's 'how to work with a professor' and Dredze's PhD
advice, and already battle-tested here. A weekly Sunday sync — live call or, when the week is
crazy, a recorded video. Agenda goes out one-to-two days in advance, always four points: what
you worked on, what you're struggling with, what you need help with, your goals for next time.

Principles: write the paper first — the draft exists and there are margin notes with your name
on them, literally. Shared repo and an honest logbook — every run logged, nulls first, either
of us can hand off at any time. Lead with the null — you saw the graveyard slide; that's
normal here. And ask early — one question on Sunday beats a week of guessing."

## Slide 15 — The ask

"So, the ask. Own the interpretability track — the load-bearing wall of a self-evolving agent —
and co-author Hello Crystal.

By next Sunday, if you're in: run the sandbox and read the top five logbook entries — that's an
afternoon. Then tell me which phase of the six-month program you'd *reshape* — it's a proposal,
not a syllabus; I'd rather you rebuild it than accept it. And bring three objections — breaking
our explanations is the job description.

Hello, Crystal. The subject is waiting to be studied — and it won't sit still."

[End. Hand him the README link and the paper PDF.]

---

## Timing map (20 min total)

| Slides | Minutes |
|---|---|
| 1–3 (dream + thesis) | 4 |
| 4 (pause, his read-back) | 2–3 |
| 5–8 (the machine + its track record) | 6 |
| 9–12 (adaptation, rails, graveyard, investor) | 5 |
| 13–15 (program + process + ask) | 3 |

## Likely questions, one-line answers

- **"If DP is the champion, where is the RL?"** — The RL challenger lives inside CrystalRL and
  currently loses honestly to the readable table; it re-enters when the action space outgrows
  enumeration — and by then the interpretability tooling you build is what keeps it admissible.
- **"Is the self-evolution real or just parameter tuning?"** — Typed machinery changes (menus,
  scenario members, solver structure), certified as readable diffs; the autonomous run
  refined a *rule*, not a weight vector.
- **"Isn't this FunSearch?"** — A relative; FunSearch trusts its evaluator, ours can't be
  trusted, so the contribution is the control machinery — and the readable substrate that
  makes each accepted change inspectable.
- **"What if I break one of your explanations?"** — Then we log it, cite you, and the system
  gets stronger; the loop has survived being broken twice.
- **"Why believe the transparency numbers?"** — They're audited, but on today's substrate —
  hardening them across seeds/substrates/profiles is literally month one of your program.
- **"Why should I believe the US 5/5?"** — Development-tier by our own taxonomy (stated in the
  paper); the dated forward reads (2027/2029) are the real test — that's the point of them.
