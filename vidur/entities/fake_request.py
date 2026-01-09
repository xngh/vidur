from typing import Tuple

from vidur.entities import Request
from vidur.entities.base_entity import BaseEntity
from vidur.logger import init_logger

logger = init_logger(__name__)


# a decorator which checks if the request has been scheduled
def check_scheduled(func):
    def wrapper(self, *args, **kwargs):
        if not self._scheduled:
            raise ValueError("Request has not been scheduled yet")
        return func(self, *args, **kwargs)

    return wrapper


def check_completed(func):
    def wrapper(self, *args, **kwargs):
        if not self._completed:
            raise ValueError("Request has not been completed yet")
        return func(self, *args, **kwargs)

    return wrapper


class FakeRequest(Request):
    def __init__(
        self,
        arrived_at: float,
        num_prefill_tokens: int,
        num_decode_tokens: int,
        num_processed_tokens: int = 0,
    ):
        super().__init__(arrived_at, num_prefill_tokens, num_decode_tokens,
                     num_processed_tokens)

        self._id = self.generate_id()
        self.counter = 0

    def generate_id(self) -> int:
        return 0

