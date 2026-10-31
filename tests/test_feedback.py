from fastapi.testclient import TestClient

from app.db.session import SessionLocal
from app.models.question import Question


def test_feedback_valid_rating_returns_created(client: TestClient) -> None:
    with SessionLocal() as db:
        question = Question(question_text="Question", answer_text="Answer")
        db.add(question)
        db.commit()
        question_id = question.id
    response = client.post(
        f"/questions/{question_id}/feedback", json={"rating": 5, "comment": "Helpful"}
    )
    assert response.status_code == 201
    assert response.json()["question_id"] == question_id
    assert response.json()["rating"] == 5
    assert response.json()["comment"] == "Helpful"
    assert response.json()["created_at"]


def test_feedback_rejects_out_of_range_ratings(client: TestClient) -> None:
    for rating in (0, 6):
        response = client.post("/questions/1/feedback", json={"rating": rating})
        assert response.status_code == 422


def test_feedback_unknown_question_returns_404(client: TestClient) -> None:
    response = client.post("/questions/999/feedback", json={"rating": 3})
    assert response.status_code == 404
