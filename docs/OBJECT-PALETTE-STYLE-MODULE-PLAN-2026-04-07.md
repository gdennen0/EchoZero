# Object Palette Style Module Plan

## Principles

- Keep shell-level Qt styling out of `widget.py` constructors so structure and presentation are owned separately.
- Use named tokens for spacing, colors, and object names instead of anonymous inline strings.
- Start with timeline shell surfaces first, then migrate painter-block colors in slices to avoid a risky all-at-once rewrite.
- Preserve existing behavior while tightening spacing toward a denser inspector-style palette.

## Module Boundaries

- `echozero/ui/qt/timeline/style.py` owns Stage Zero timeline shell style tokens and stylesheet builders.
- `echozero/ui/qt/timeline/widget.py` owns widget composition and state wiring only; it should consume style tokens, not define them.
- `echozero/ui/FEEL.py` remains the cross-cutting behavior and geometry constant layer for interaction/rendering metrics.
- `echozero/ui/qt/timeline/blocks/*` still own painter logic for their local visuals today; future slices can import shared color tokens from `style.py` once those surfaces are normalized.

## Migration Plan

1. Establish `style.py` as the single owner for object palette and shell stylesheet tokens.
2. Move the `ObjectInfoPanel` stylesheet and layout spacing out of inline constructor code into `style.py`.
3. Move other shell-adjacent stylesheet strings such as scroll-area background into the same module.
4. Introduce grouped color/token families for painter blocks (`ruler`, `layer_header`, `take_row`, `transport`) once each block gets an explicit visual pass.
5. When enough block-level tokens stabilize, decide whether a second module split is warranted:
   `style.py` for shell surfaces and `block_style.py` for painter tokens.

## Why This Fits The Codebase

- It respects the existing EZ2 split where `widget.py` composes reusable parts and block modules render them.
- It avoids mixing FEEL geometry constants with shell skinning concerns; FEEL stays behavioral while `style.py` becomes presentational.
- It creates a narrow import surface that future palette and inspector work can reuse without spreading more inline QSS.
- It enables incremental migration: one module can absorb shell styles now without forcing a large repaint refactor across all timeline blocks.
- It is testable with focused unit coverage because the stylesheet builders and token values are now deterministic module-level outputs.
