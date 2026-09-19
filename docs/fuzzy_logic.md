# Fuzzy Logic Methodology

## 1. What is fuzzy logic?

Classical (crisp) logic forces a value into exactly one category: urgency
79 is either "high" or it isn't. Fuzzy logic instead allows partial,
overlapping membership: urgency 79 might be 70% "high" and 30% "medium"
simultaneously. This models real-world ambiguity far better than hard
thresholds, which is why it is well suited to judging severity from
natural-language complaints.

## 2. Why fuzzy logic for ticket prioritization?

Support tickets rarely fall into clean-cut buckets. A complaint can be
moderately urgent AND highly costly AND mildly annoyed - fuzzy logic lets
all of these partial truths combine smoothly into one score, and the
result can be explained by which rules fired and by how much.

## 3. Input variables

| Variable | Range | Terms |
|---|---|---|
| Urgency | 0-100 | low, medium, high |
| Financial Impact | 0-100 | low, medium, high |
| Sentiment | 0-100 (0=neutral/positive, 100=extremely negative) | neutral, negative, very_negative |
| Delay | 0-100 (0=no delay, 100=extremely long delay) | short, moderate, long |

## 4. Membership functions

All variables use triangular membership functions (`skfuzzy.trimf`) defined
in `src/fuzzy_engine.py`, e.g. for Urgency:

- **low**: triangle at (0, 0, 40)
- **medium**: triangle at (30, 50, 70)
- **high**: triangle at (60, 100, 100)

Financial Impact, Sentiment (neutral/negative/very_negative), and Delay
(short/moderate/long) follow the same shape on the 0-100 scale.

## 5. Output variable

**Priority** (0-100) with terms: low, medium, high, critical - also
triangular, e.g. critical is a triangle at (75, 100, 100).

## 6. Fuzzy rules

17 rules are implemented (`src/fuzzy_engine.py::_build_rules`), for example:

1. IF urgency is HIGH AND financial_impact is HIGH THEN priority is CRITICAL
2. IF urgency is HIGH AND sentiment is VERY_NEGATIVE THEN priority is HIGH
3. IF urgency is HIGH AND delay is LONG THEN priority is CRITICAL
4. IF financial_impact is HIGH AND sentiment is VERY_NEGATIVE THEN priority is CRITICAL
8. IF urgency is LOW AND financial_impact is LOW THEN priority is LOW

(Full list in code, with two extra coverage rules - R16/R17 - added to
avoid zero-firing edge cases where urgency and financial_impact disagree
sharply.)

## 7. Fuzzification

Each crisp input (e.g. urgency=65) is converted into membership degrees
against every term of its variable, using `skfuzzy.interp_membership`.
Example: urgency=65 -> {low: 0.0, medium: 0.25, high: 0.125}.

## 8. Rule evaluation

Each rule's antecedent (an AND/OR combination of terms) is evaluated using
min (AND) / max (OR) operators over the fuzzified degrees, producing a
firing strength per rule.

## 9. Aggregation

The consequent membership functions of all firing rules are combined
(clipped by their firing strength and unioned via max) into a single
aggregated output fuzzy set for Priority.

## 10. Defuzzification

The aggregated fuzzy set is converted to a single crisp number using the
**centroid** (center of gravity) method - the standard scikit-fuzzy
default via `ControlSystemSimulation.compute()`.

## 11. Priority classification

The crisp score is mapped to a label using centrally configured thresholds
(`src/config.py::PriorityThresholds`):

| Score range | Level |
|---|---|
| 0-30 | LOW |
| 30-55 | MEDIUM |
| 55-80 | HIGH |
| 80-100 | CRITICAL |
