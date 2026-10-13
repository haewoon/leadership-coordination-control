from src.policies.flat_policy import FlatPolicy
from src.policies.inclusive_policy import InclusivePolicy
from src.policies.llm_coordinator_policy import LLMCoordinatorPolicy
from src.policies.mad_policy import MADPolicy
from src.policies.random_action_policy import RandomActionPolicy
from src.policies.simple_hierarchy_policy import SimpleHierarchyPolicy
from src.policies.situational_policy import SituationalPolicy
from src.policies.situational_random_switch_policy import SituationalRandomSwitchPolicy
from src.policies.situational_variants import (
    SituationalSplit3Policy,
    SituationalLenientPolicy,
    SituationalAltLexPolicy,
)
from src.policies.theory_free_control_policy import TheoryFreeControlPolicy
from src.policies.transactional_accept_only_policy import TransactionalAcceptOnlyPolicy
from src.policies.transactional_policy import TransactionalPolicy
from src.policies.transformational_policy import TransformationalPolicy
from src.policies.transformational_directives_only_policy import TransformationalDirectivesOnlyPolicy

POLICY_REGISTRY = {
    "flat": FlatPolicy,
    "inclusive": InclusivePolicy,
    "llm_coordinator": LLMCoordinatorPolicy,
    "mad": MADPolicy,
    "random_action": RandomActionPolicy,
    "simple_hierarchy": SimpleHierarchyPolicy,
    "situational": SituationalPolicy,
    "situational_split3": SituationalSplit3Policy,
    "situational_lenient": SituationalLenientPolicy,
    "situational_altlex": SituationalAltLexPolicy,
    "situational_random_switch": SituationalRandomSwitchPolicy,
    "theory_free_control": TheoryFreeControlPolicy,
    "transactional": TransactionalPolicy,
    "transactional_accept_only": TransactionalAcceptOnlyPolicy,
    "transformational": TransformationalPolicy,
    "transformational_directives_only": TransformationalDirectivesOnlyPolicy,
}
