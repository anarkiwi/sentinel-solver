"""Lockstep scheduler: searches run as cooperative tasks whose queries are batched.

A task is an ordinary callable; where it would run a query alone it asks :func:`call`,
which outside a task answers None so the caller runs the scalar path.  Modules that own
a query kind :func:`register` a handler taking one round's payloads, one answer each.
"""

from greenlet import getcurrent, greenlet

PENDING = object()
_HANDLERS = {}


class Failure:
    """The ``Exception`` a task ended with, as a result."""

    def __init__(self, error):
        self.error = error


def register(kind, handler):
    """Serve ``kind`` requests with ``handler(payloads) -> answers``."""
    _HANDLERS[kind] = handler


class _Task(greenlet):
    def __init__(self, fn):
        def guarded():
            try:
                return fn()
            except Exception as error:  # pylint: disable=broad-except
                return Failure(error)

        super().__init__(guarded)


def in_task():
    """Whether the caller runs inside a :func:`run` task."""
    return isinstance(getcurrent(), _Task)


def call(kind, payload):
    """The answer to a ``kind`` request, served with the other tasks' requests of the
    same round; None when the caller is not in a task."""
    me = getcurrent()
    if not isinstance(me, _Task):
        return None
    return me.parent.switch((kind, payload))


def run(fns, decided=None):
    """Results of the callables ``fns`` run in lockstep, in order: each return value, or
    a :class:`Failure`.  After a round ``decided(results)`` may end the rest, whose
    results stay :data:`PENDING`."""
    tasks = [_Task(fn) for fn in fns]
    results = [PENDING] * len(tasks)
    reply = dict.fromkeys(range(len(tasks)))
    started = False
    while reply:
        asks = {}
        for i, answer in reply.items():
            out = tasks[i].switch(answer) if started else tasks[i].switch()
            if tasks[i].dead:
                results[i] = out
            else:
                asks[i] = out
        started = True
        by_kind = {}
        for i, (kind, payload) in asks.items():
            by_kind.setdefault(kind, []).append((i, payload))
        reply = {}
        for kind, group in by_kind.items():
            answers = _HANDLERS[kind]([payload for _, payload in group])
            reply.update(zip([i for i, _ in group], answers))
        if decided is not None and decided(results):
            break
    for task in tasks:
        if not task.dead:
            task.throw()
    return results
