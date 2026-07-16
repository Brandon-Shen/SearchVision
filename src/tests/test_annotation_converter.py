import json

from src.utils.annotation_converter import convert_to_yolo_format


def test_letterboxed_canvas_coordinates_are_normalized_to_displayed_image():
    annotation = json.dumps({
        "rects": [{"x": 100, "y": 75, "width": 100, "height": 150}],
        "canvasWidth": 300,
        "canvasHeight": 300,
        "imageElement": {
            "displayWidth": 300,
            "displayHeight": 150,
            "offsetX": 0,
            "offsetY": 75,
        },
    })
    assert convert_to_yolo_format(annotation, 1000, 500) == (
        "0 0.500000 0.500000 0.333333 1.000000")


def test_box_outside_displayed_image_is_clipped():
    annotation = json.dumps({
        "rects": [{"x": -10, "y": -10, "width": 30, "height": 30}],
        "imageElement": {
            "displayWidth": 100,
            "displayHeight": 100,
            "offsetX": 0,
            "offsetY": 0,
        },
    })
    assert convert_to_yolo_format(annotation, 100, 100) == (
        "0 0.100000 0.100000 0.200000 0.200000")
