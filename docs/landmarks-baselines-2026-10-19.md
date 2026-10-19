Clean m/600M baselines, mean of 2 seeds, evals every 25M. Half point: first eval with >= 50% of the final above-baseline range; plateau: first eval after which the mean stays within 10% of that range of its final value.

| metric | baseline | final | half point (M tokens) | plateau (M tokens) |
|---|---|---|---|---|
| blimp_acc | 0.500 | 0.707 | 25 | 200 |
| val_loss | 9.088 | 3.458 | 25 | 50 |
| long_range_gain | 0.000 | 0.471 | 50 | 125 |
| induction_gain | 0.002 | 6.589 | 75 | 150 |
| blimp/anaphor_agreement | 0.500 | 0.915 | 25 | 449 |
| blimp/argument_structure | 0.500 | 0.734 | 25 | 474 |
| blimp/binding | 0.500 | 0.724 | 25 | 125 |
| blimp/control_raising | 0.500 | 0.702 | 25 | 600 |
| blimp/determiner_noun_agreement | 0.500 | 0.880 | 25 | 175 |
| blimp/ellipsis | 0.500 | 0.734 | 50 | 574 |
| blimp/filler_gap_dependency | 0.500 | 0.700 | 25 | 100 |
| blimp/irregular_forms | 0.500 | 0.889 | 25 | 350 |
| blimp/island_effects | 0.500 | 0.465 | 25 | 600 |
| blimp/npi_licensing | 0.500 | 0.587 | 25 | 600 |
| blimp/quantifiers | 0.500 | 0.667 | 25 | 600 |
| blimp/s-selection | 0.500 | 0.750 | 50 | 549 |
| blimp/subject_verb_agreement | 0.500 | 0.769 | 50 | 250 |
