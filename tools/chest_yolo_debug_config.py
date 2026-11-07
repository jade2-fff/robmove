"""Shared configuration and acceptance rules for chest-camera YOLO debug."""

import yaml


def load_config(path):
    with path.open(encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    thresholds = config['confidence']
    for label, value in [('default', thresholds['default']), *thresholds['per_class'].items()]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f'confidence for {label!r} must be a number between 0 and 1')
    return config


def minimum_confidence(thresholds):
    """Keep candidates for any class whose threshold is below the default."""
    return min([thresholds['default'], *thresholds['per_class'].values()])


def selected_indices(names, class_ids, scores, thresholds):
    """Select accepted detections before drawing boxes, masks, or counts."""
    default = thresholds['default']
    overrides = thresholds['per_class']
    return [
        index
        for index, (class_id, score) in enumerate(zip(class_ids, scores))
        if score >= overrides.get(names[int(class_id)], default)
    ]


def filter_result(result, thresholds):
    boxes = result.boxes
    if boxes is None or not len(boxes):
        return result
    keep = selected_indices(result.names, boxes.cls.tolist(), boxes.conf.tolist(), thresholds)
    return result[keep]
