from __future__ import annotations

from dataclasses import dataclass, field

from cave.commitments.agency import ActionPolicy, default_action_policy
from cave.commitments.attention import AttentionPolicy, AttentionProfile, default_attention_policy
from cave.commitments.affect import ValenceEvaluator, default_valence_evaluator
from cave.commitments.learning import LearningRule, default_learning_rule
from cave.commitments.memory import MemoryParams
from cave.commitments.objective import ObjectiveEvaluator, default_objective_evaluator
from cave.commitments.topology import SubjectiveTopologyParams
from cave.commitments.workspace import WorkspaceCompressor, default_workspace_compressor


@dataclass(frozen=True)
class ModelParams:
    memory: MemoryParams
    attention: AttentionProfile = field(default_factory=AttentionProfile)
    attention_policy: AttentionPolicy = field(default_factory=default_attention_policy)
    topology: SubjectiveTopologyParams = field(default_factory=SubjectiveTopologyParams)
    learning_rule: LearningRule = field(default_factory=default_learning_rule)
    valence_evaluator: ValenceEvaluator = field(default_factory=default_valence_evaluator)
    objective_evaluator: ObjectiveEvaluator = field(default_factory=default_objective_evaluator)
    workspace_compressor: WorkspaceCompressor = field(default_factory=default_workspace_compressor)
    workspace_input_mode: str = "actual"
    action_policy: ActionPolicy = field(default_factory=default_action_policy)

    def __post_init__(self) -> None:
        if self.workspace_input_mode not in {"actual", "workspace"}:
            raise ValueError("workspace_input_mode must be 'actual' or 'workspace'")
