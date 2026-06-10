import taskqueue
from rq.registry import BaseRegistry
from rq.timeouts import get_default_death_penalty_class


def test_base_registry_uses_platform_death_penalty():
    assert BaseRegistry.death_penalty_class is get_default_death_penalty_class()


def test_queues_use_platform_death_penalty():
    expected = get_default_death_penalty_class()
    assert taskqueue.rq_queue_high.death_penalty_class is expected
    assert taskqueue.rq_queue_default.death_penalty_class is expected


def test_registries_built_from_queues_inherit_platform_death_penalty():
    expected = get_default_death_penalty_class()
    for queue in (taskqueue.rq_queue_high, taskqueue.rq_queue_default):
        assert queue.started_job_registry.death_penalty_class is expected
        assert queue.finished_job_registry.death_penalty_class is expected
        assert queue.failed_job_registry.death_penalty_class is expected
