import pytest

from src.baseline_predict import load_model, prepare_comment, predict_comment


@pytest.mark.parametrize("text", ["", "   ", "a" * 5001, None])
def test_reject_invalid_input(text):
    with pytest.raises(ValueError):
        prepare_comment(text)


def test_reject_unknown_checkpoint():
    with pytest.raises(ValueError):
        load_model("unknown", "A")


def test_inference_uses_prepared_text_and_rejects_oov(monkeypatch):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    model = make_pipeline(TfidfVectorizer(), LogisticRegression())
    model.fit(["bad service", "average service", "good service"], [0, 1, 2])
    # Match the named checkpoint pipeline contract.
    from sklearn.pipeline import Pipeline
    model = Pipeline([("tfidf", model[0]), ("classifier", model[1])])
    monkeypatch.setattr("src.baseline_predict.load_model", lambda *args: model)
    monkeypatch.setattr("src.baseline_predict.prepare_comment", lambda text: "good service")
    result = predict_comment("raw input", "linear_svc", "B")
    assert result["label"] == 2
    assert result["label_name"] == "Tích cực"
    monkeypatch.setattr("src.baseline_predict.prepare_comment", lambda text: "unseenword")
    with pytest.raises(ValueError, match="từ vựng"):
        predict_comment("raw input", "linear_svc", "B")
