"""The lockstep scheduler batches tasks' requests and keeps their order and failures."""

from sentinel import fleet


def _serve(seen):
    def handler(payloads):
        seen.append(list(payloads))
        return [p * 10 for p in payloads]

    return handler


def test_requests_of_a_round_share_one_handler_call():
    seen = []
    fleet.register("times10", _serve(seen))

    def task(n):
        return lambda: sum(fleet.call("times10", n + i) for i in range(2))

    assert fleet.run([task(1), task(2), task(3)]) == [30, 50, 70]
    assert seen == [[1, 2, 3], [2, 3, 4]]


def test_call_outside_a_task_is_unserved():
    assert fleet.call("times10", 1) is None
    assert not fleet.in_task()


def test_exception_is_a_failure_and_others_finish():
    fleet.register("times10", _serve([]))

    def bad():
        fleet.call("times10", 1)
        raise ValueError("boom")

    ok, failed = fleet.run([lambda: fleet.call("times10", 2), bad])
    assert ok == 20
    assert isinstance(failed, fleet.Failure) and isinstance(failed.error, ValueError)


def test_decided_ends_the_remaining_tasks():
    fleet.register("times10", _serve([]))
    ran = []

    def task(rounds):
        def go():
            for i in range(rounds):
                ran.append((rounds, i))
                fleet.call("times10", i)
            return rounds

        return go

    out = fleet.run([task(1), task(50)], lambda res: res[0] is not fleet.PENDING)
    assert out == [1, fleet.PENDING]
    assert len(ran) < 10
