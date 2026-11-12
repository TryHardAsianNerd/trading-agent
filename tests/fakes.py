"""Shared test fakes."""

from redis import RedisError


class FakeRedis:
    def __init__(self) -> None:
        self.store: dict[str, str] = {}
        self.ttls: dict[str, int] = {}

    def set(self, key, value, ex=None):
        self.store[key] = value
        if ex is not None:
            self.ttls[key] = ex

    def get(self, key):
        return self.store.get(key)

    def delete(self, key):
        self.store.pop(key, None)
        self.ttls.pop(key, None)


class BrokenRedis:
    def get(self, key):
        raise RedisError("connection refused")

    def set(self, key, value, ex=None):
        raise RedisError("connection refused")

    def delete(self, key):
        raise RedisError("connection refused")
