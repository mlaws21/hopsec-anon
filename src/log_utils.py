import logging

RESULT_LEVEL_NUM = 25
logging.addLevelName(RESULT_LEVEL_NUM, "RESULT")


def _result(self, message, *args, **kwargs):
    if self.isEnabledFor(RESULT_LEVEL_NUM):
        self._log(RESULT_LEVEL_NUM, message, args, **kwargs)


logging.Logger.result = _result


ROUTE_LEVELS = {
    "forward": 26,
    "execute": 27,
    "split": 28,
    "task": 29,
}
for _name, _num in ROUTE_LEVELS.items():
    logging.addLevelName(_num, _name.upper())


def _route(self, agent_id, decision, reason=""):
    decision_key = (decision or "").lower().strip()
    level = ROUTE_LEVELS.get(decision_key, RESULT_LEVEL_NUM)
    message = f"Agent {agent_id} {decision_key.upper()}: {(reason or '')[:100]}"
    if self.isEnabledFor(level):
        self._log(level, message, ())


logging.Logger.route = _route


def _task(self, message, *args, **kwargs):
    if self.isEnabledFor(ROUTE_LEVELS["task"]):
        self._log(ROUTE_LEVELS["task"], message, args, **kwargs)


logging.Logger.task = _task
