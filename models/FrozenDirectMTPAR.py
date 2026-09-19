"""Frozen-Q1 version of the causal placeholder Direct-MTP baseline."""

from models.DirectMTPAR import Model as DirectMTPModel


class Model(DirectMTPModel):
    """Train future placeholders while keeping the accepted Q1 immutable."""

    UPDATE_SCOPE = "frozen"
