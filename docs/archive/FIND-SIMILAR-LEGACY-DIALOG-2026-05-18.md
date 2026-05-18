# Find Similar Legacy Dialog Archive

Status: archived
Archived: 2026-05-18

The previous Find Similar popup exposed comparison method, scan scope, sensitivity,
shape smoothing, control points, fuzziness, outcome actions, saved mini-model
selection, and model training in one settings-heavy dialog.

That UI was useful as an implementation probe, but it made the operator tune the
matcher instead of reviewing candidate events. The product path is now the
interactive review flow: scan candidates, preserve timeline order, play guesses,
mark Match / Not Match / Skip, rerank with local review labels, then apply the
confirmed matched set.

Rollback reference: use git history before this archive date if the old
slider-based `find_similar_dialog.py` implementation is needed for comparison.
