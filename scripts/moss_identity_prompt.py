"""Shared training/inference prompt for the experimental fixed-roster decoder."""

from build_moss_diarization_dataset import DEFAULT_PROMPT


def identity_prompt(names):
    roster = ", ".join(f"[S{i+1:02d}]={name}" for i, name in enumerate(names))
    return (
        DEFAULT_PROMPT
        + " Use these fixed speaker identities regardless of speaking order: "
        + roster
        + "."
    )
