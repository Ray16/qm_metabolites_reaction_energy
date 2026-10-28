"""Structure-based routing and physics corrections for the MetaG pipeline.

Each module detects a structural pattern and applies a physically-motivated transform, self-gating so
that "always on" is safe (a reaction the transform doesn't apply to is returned unchanged):

    ph0       -- pH-0 / Alberty pKa transform: compute the neutral microspecies (continuum-solvation
                 valid), bridge to pH 7 analytically with textbook pKa's. Nothing fit to the database.
    cofactor  -- isodesmic cofactor cores: replace the floppy NAD(P)/GSH tail with a canonical ring/thiol.
    truncate  -- spectator truncation: cut atoms that don't change (Δq=0), mass-balance guarded.
    aldehyde  -- population-weighted carbonyl<->gem-diol mixture, gated to α-EWG-activated aldehydes.
    reaction_families -- structural family detectors (classification only; no correction implied).
    anchor    -- legacy per-family offsets retained for frozen-release reproduction. Corrections are
                 disabled in production (ANCHOR_CORRECT off); see docs/anchor-audit.md.
"""
