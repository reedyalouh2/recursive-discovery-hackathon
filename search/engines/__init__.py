"""Engine adapters. Each maps (task, budget, rng, history, initial, workspace) to a construction."""

from integrations.shinka.engine import ShinkaEngine
from search.engines.offline import OfflineILSEngine

ENGINES = {engine.name: engine for engine in (OfflineILSEngine, ShinkaEngine)}
