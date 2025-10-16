
# Design Note: Extending This Setup to Real QA Agents

## Why this toy maps well to QA
This project mirrors the framework of a QA navigation problem without UI specifics:

- Graph nodes represent application states/screens; edges represent transitions (select, scroll, tap).
- Random 'popups' emulate unexpected overlays that can nullify actions.
- I have used generic features (local bits + progress scalars + potential φ) instead of node IDs. In QA, you’d replace the 10 local bits with UI-derived features (e.g., detected widgets), while keeping progress-like scalars and a goal potential.

On the control side, PPO-LSTM is a direct fit: short-term memory helps navigate around modals, avoid loops, and remember recent context. Training-only potential shaping remains valid and is turned off in evaluation/production to keep metrics 'honest'.