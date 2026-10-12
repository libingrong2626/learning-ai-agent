#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any


class ApiBalanceExhaustedError(RuntimeError):
    """API 账户欠费或计费额度耗尽，继续重试不会成功。"""


_BALANCE_ERROR_MARKERS = (
    "accountoverdue",
    "account_overdue",
    "account overdue",
    "account is overdue",
    "service.overdue",
    "serviceoverdue",
    "insufficientbalance",
    "insufficient_balance",
    "insufficient balance",
    "balance is insufficient",
    "balance insufficient",
    "insufficient_quota",
    "exceeded your current quota",
    "billing hard limit",
    "credit balance",
    "out of credit",
    "credits exhausted",
    "payment required",
    "账户余额不足",
    "账号余额不足",
    "余额不足",
    "账户欠费",
    "账号欠费",
    "账户已欠费",
    "账号已欠费",
    "请充值后重试",
)


def is_api_balance_error(exc: BaseException) -> bool:
    """从异常链、响应状态和响应正文中识别不可重试的余额错误。"""
    details: list[str] = []
    current: BaseException | None = exc
    visited: set[int] = set()

    while current is not None and id(current) not in visited:
        visited.add(id(current))
        details.append(str(current))

        for attr_name in ("body", "code", "message"):
            value = getattr(current, attr_name, None)
            if value:
                details.append(str(value))

        response = getattr(current, "response", None)
        if response is not None:
            status_code = getattr(response, "status_code", None)
            if status_code == 402:
                return True

            response_text = getattr(response, "text", None)
            if response_text:
                details.append(str(response_text))

            try:
                response_json: Any = response.json()
            except Exception:
                response_json = None
            if response_json:
                details.append(str(response_json))

        current = current.__cause__ or current.__context__

    combined_text = "\n".join(details).casefold()
    return any(marker.casefold() in combined_text for marker in _BALANCE_ERROR_MARKERS)


def raise_for_api_balance_error(exc: BaseException) -> None:
    """余额不足时转换为管线能够立即识别并停止的异常。"""
    if isinstance(exc, ApiBalanceExhaustedError):
        raise exc

    if is_api_balance_error(exc):
        raise ApiBalanceExhaustedError(
            "API 账户余额不足、账户欠费或计费额度已耗尽"
        ) from exc
