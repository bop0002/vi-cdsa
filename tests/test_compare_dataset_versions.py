import pandas as pd
import pytest

from src.compare_dataset_versions import compare_frames


def test_compare_by_id_not_row_order():
    old = pd.DataFrame({"id": ["a", "b"], "label": [0, 2], "text_ml": ["bad", "good"]})
    assert compare_frames(old, old.iloc[::-1])["same_prediction_inputs"]
    new = pd.DataFrame({"id": ["b", "c"], "label": [1, 2], "text_ml": ["changed", "good"]})
    diff = compare_frames(old, new)
    assert diff["added_ids"] == diff["removed_ids"] == 1
    assert diff["changed_label"] == diff["changed_text_ml"] == 1
    assert not diff["same_prediction_inputs"]


def test_duplicate_ids_rejected():
    frame = pd.DataFrame({"id": ["a", "a"], "label": [0, 0], "text_ml": ["bad", "bad"]})
    with pytest.raises(ValueError, match="Duplicate IDs"):
        compare_frames(frame, frame)
