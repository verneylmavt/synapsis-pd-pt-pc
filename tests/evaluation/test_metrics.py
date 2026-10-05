def test_directional_events_match_one_to_one_and_count_errors_include_unmatched():
    from scripts.evaluate import score_events
    truth = [{"type": "enter", "area_id": 1, "time_seconds": 2},
             {"type": "exit", "area_id": 1, "time_seconds": 5}]
    predicted = [{"type": "enter", "area_id": 1, "time_seconds": 2.1},
                 {"type": "enter", "area_id": 1, "time_seconds": 2.2},
                 {"type": "exit", "area_id": 1, "time_seconds": 10}]
    result = score_events(truth, predicted, tolerance_seconds=0.5)
    assert result["enter"]["tp"] == 1
    assert result["enter"]["fp"] == 1
    assert result["enter"]["count_error"] == 1
    assert result["exit"]["fn"] == 1
    assert result["exit"]["fp"] == 1


def test_human_annotations_filter_invalid_rows_and_normalize_boxes(tmp_path):
    from scripts.evaluate import load_mot_truth
    path = tmp_path / "gt.txt"
    path.write_text("1,7,10,20,30,40,1,-1,-1,-1\n1,8,0,0,5,5,0,-1,-1,-1\n")
    truth = load_mot_truth(path, width=100, height=100)
    assert len(truth[1]) == 1
    assert truth[1][0]["tracker_id"] == "7"
    assert truth[1][0]["x2"] == 0.39


def test_mot_one_based_box_origin_becomes_zero_based_pixel_origin(tmp_path):
    from scripts.evaluate import load_mot_truth
    path = tmp_path / "gt.txt"
    path.write_text("1,7,1,1,30,40,1,-1,-1,-1\n")
    box = load_mot_truth(path, width=100, height=100)[1][0]
    assert (box["x1"], box["y1"], box["x2"], box["y2"]) == (0, 0, 0.3, 0.4)
