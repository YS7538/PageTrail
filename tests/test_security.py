import pytest
from fastapi.testclient import TestClient

from app.core.config import settings


def test_unset_key_allows_protected_routes(client: TestClient, monkeypatch) -> None:
    monkeypatch.setattr(settings, "app_api_key", None)
    assert client.get("/documents").status_code == 200
    assert client.post("/questions", json={"question": "test"}).status_code == 201
    assert client.get("/health").status_code == 200


@pytest.mark.parametrize("headers", [{}, {"X-API-Key": "wrong-key"}])
def test_configured_key_rejects_missing_or_wrong_key(
    client: TestClient, monkeypatch, headers: dict[str, str]
) -> None:
    monkeypatch.setattr(settings, "app_api_key", "expected-key")
    assert client.get("/documents", headers=headers).status_code == 401
    assert client.post(
        "/questions", json={"question": "test"}, headers=headers
    ).status_code == 401
    assert client.get("/health").status_code == 200


def test_configured_key_allows_requests_with_correct_key(
    client: TestClient, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "app_api_key", "expected-key")
    headers = {"X-API-Key": "expected-key"}
    assert client.get("/documents", headers=headers).status_code == 200
    assert client.post(
        "/questions", json={"question": "test"}, headers=headers
    ).status_code == 201
    assert client.get("/health").status_code == 200
