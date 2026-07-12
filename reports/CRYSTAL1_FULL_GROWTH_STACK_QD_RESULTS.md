# CRYSTAL-1 full growth stack on a modular POET/QD substrate

Code: [hl_full_stack_qd.py](../interpretability/hl_full_stack_qd.py)  
Raw report: [hl_full_stack_qd_report.json](../interpretability/hl_full_stack_qd_report.json)

## Question

Can the full Heuristic Learning growth stack run as a closed loop when the substrate actually rewards modular growth?

Tested operators:

- `add_rule`: grow structure by covering a new venue.
- `add_knob`: expose the latent unwind lever.
- `retune`: tune the shared provide threshold.
- `recombine`: union-crossover two partial elites.

The important part is the ablation: full stack vs removing each operator.

## Result

| regime | variant | best honest mean | max coverage mean | verdict |
|---|---:|---:|---:|---|
| G6 loose | full stack | 8.317 | 5.4 | strong stack, but recombine not required |
| G6 loose | minus add_knob | 5.254 | 5.8 | add_knob load-bearing |
| G6 loose | minus retune | 5.538 | 6.0 | retune load-bearing |
| G6 loose | minus recombine | 8.699 | 6.0 | recombine redundant here |
| G12 tight | full stack | 5.093 | 5.6 | full stack wins |
| G12 tight | minus add_knob | 3.345 | 6.0 | add_knob load-bearing |
| G12 tight | minus retune | 3.966 | 8.2 | retune load-bearing |
| G12 tight | minus recombine | 4.871 | 5.2 | recombine load-bearing here |

Operator activity in full stack:

| regime | add_rule inserts | add_knob inserts | retune inserts | recombine inserts | recombine yield |
|---|---:|---:|---:|---:|---:|
| G6 loose | 55 | 8 | 57 | 22 | 21.4% |
| G12 tight | 44 | 7 | 45 | 22 | 32.4% |

## Interpretation

`add_knob` and `retune` are consistently load-bearing. Removing `add_knob` removes the unwind mechanism and cuts honest score sharply. Removing `retune` leaves the policy stuck near a poor threshold and also cuts score sharply.

`recombine` is real but conditional. In the small loose G6 setting, `add_rule` can assemble coverage cheaply, so recombine mainly competes for proposal budget and the no-recombine ablation scores higher. In the wider tight G12 setting, coverage is harder to finish within budget; recombine becomes useful because union-crossover can fuse partial elites faster than mutation alone.

## Feynman summary

The growth stack works, but crossover is not magic glue. If the puzzle is small, adding pieces one by one is enough and crossover can waste turns. If the puzzle is wide and modular, crossover matters because it can merge two half-solutions into one larger solution in a single step.

## Honest scope

This is still a polygon / modular POET-QD substrate, not the real market. The result supports the engineering grammar of Heuristic Learning: grow rules, expose knobs, retune knobs, and recombine partial strategies. It does not prove that the same stack will improve R6c/CRYSTAL on real financial data without a firewall-gated transfer test.

