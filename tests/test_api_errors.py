from __future__ import annotations

import unittest

from paper_classifier.common.api_errors import is_api_balance_error


class FakeResponse:
    def __init__(self, status_code: int, text: str = "") -> None:
        self.status_code = status_code
        self.text = text

    def json(self) -> dict:
        return {}


class FakeApiError(RuntimeError):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.response = FakeResponse(status_code, message)


class ApiBalanceErrorTests(unittest.TestCase):
    def test_http_402_is_balance_error(self) -> None:
        self.assertTrue(is_api_balance_error(FakeApiError("", 402)))

    def test_volcano_account_overdue_code_is_balance_error(self) -> None:
        self.assertTrue(
            is_api_balance_error(
                FakeApiError('{"code": "AccountOverdue"}', 400)
            )
        )

    def test_openai_billing_quota_is_balance_error(self) -> None:
        self.assertTrue(
            is_api_balance_error(
                RuntimeError("insufficient_quota: exceeded your current quota")
            )
        )

    def test_rate_limit_is_not_misidentified_as_balance_error(self) -> None:
        self.assertFalse(
            is_api_balance_error(
                FakeApiError("rate_limit_exceeded: too many requests", 429)
            )
        )


if __name__ == "__main__":
    unittest.main()
