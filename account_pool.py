from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


def get_account_identifier(account: dict[str, Any]) -> str:
    """返回账号的唯一标识，优先 email，否则使用 mobile。"""
    return account.get("email", "").strip() or account.get("mobile", "").strip()


@dataclass(slots=True)
class AccountLease:
    """账号租约（轮询模式下 release 为空操作）。"""

    account_id: str
    account: dict[str, Any]

    @property
    def identifier(self) -> str:
        return self.account_id

    async def release(self) -> None:
        pass


class AccountPool:
    """简单轮询账号池，无并发控制。"""

    def __init__(self, accounts: Iterable[dict[str, Any]]):
        self._accounts: list[tuple[str, dict[str, Any]]] = []
        seen: set[str] = set()
        for account in accounts:
            aid = get_account_identifier(account)
            if aid and aid not in seen:
                seen.add(aid)
                self._accounts.append((aid, account))
        self._index = 0

    def has_accounts(self) -> bool:
        return bool(self._accounts)

    def size(self) -> int:
        return len(self._accounts)

    async def acquire(self, exclude_ids: set[str] | None = None) -> AccountLease | None:
        if not self._accounts:
            return None
        excluded = set(exclude_ids or ())
        n = len(self._accounts)
        start = self._index % n
        for offset in range(n):
            idx = (start + offset) % n
            aid, account = self._accounts[idx]
            if aid not in excluded:
                self._index = idx + 1
                return AccountLease(account_id=aid, account=account)
        return None
