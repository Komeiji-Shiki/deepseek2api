from __future__ import annotations

import asyncio
import random
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Iterable


def get_account_identifier(account: dict[str, Any]) -> str:
    """返回账号的唯一标识，优先 email，否则使用 mobile。"""
    return account.get("email", "").strip() or account.get("mobile", "").strip()


@dataclass(slots=True)
class AccountLease:
    """账号租约，确保同一个租约只会被释放一次。"""

    pool: "AccountPool"
    account_id: str
    account: dict[str, Any]
    _released: bool = field(default=False, init=False, repr=False)

    @property
    def identifier(self) -> str:
        return self.account_id

    @property
    def released(self) -> bool:
        return self._released

    async def release(self) -> None:
        if self._released:
            return
        self._released = True
        await self.pool.release(self.account_id)


class AccountPool:
    """并发安全的账号池。"""

    def __init__(self, accounts: Iterable[dict[str, Any]]):
        self._accounts_by_id: dict[str, dict[str, Any]] = {}
        account_ids: list[str] = []

        for account in accounts:
            account_id = get_account_identifier(account)
            if not account_id or account_id in self._accounts_by_id:
                continue
            self._accounts_by_id[account_id] = account
            account_ids.append(account_id)

        random.shuffle(account_ids)
        self._available: Deque[str] = deque(account_ids)
        self._in_use: set[str] = set()
        self._lock = asyncio.Lock()

    def has_accounts(self) -> bool:
        return bool(self._accounts_by_id)

    def size(self) -> int:
        return len(self._accounts_by_id)

    async def acquire(self, exclude_ids: set[str] | None = None) -> AccountLease | None:
        excluded = set(exclude_ids or ())
        async with self._lock:
            if not self._available:
                return None

            skipped: list[str] = []
            selected_id: str | None = None
            rounds = len(self._available)

            for _ in range(rounds):
                candidate = self._available.popleft()
                if candidate in excluded:
                    skipped.append(candidate)
                    continue
                if candidate in self._in_use:
                    continue
                selected_id = candidate
                self._in_use.add(candidate)
                break

            self._available.extend(skipped)

            if selected_id is None:
                return None

            return AccountLease(
                pool=self,
                account_id=selected_id,
                account=self._accounts_by_id[selected_id],
            )

    async def release(self, account_id: str) -> None:
        async with self._lock:
            if account_id not in self._in_use:
                return
            self._in_use.remove(account_id)
            self._available.append(account_id)