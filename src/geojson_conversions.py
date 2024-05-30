from typing import Dict, List

import geojson

from box import Box


def bbox_to_geojson_polygon(bbox: Box) -> geojson.Polygon:
    """Converts a Box to a GeoJSON Polygon.

    Args:
        bbox (Box): the box to convert.

    Returns:
        geojson.Polygon: the box transformed into a GeoJSON Polygon.
    """
    return geojson.Polygon(
        [
            [
                (bbox.x_min, bbox.y_min),
                (bbox.x_min, bbox.y_max),
                (bbox.x_max, bbox.y_max),
                (bbox.x_max, bbox.y_min),
                (bbox.x_min, bbox.y_min),
            ]
        ]
    )


def bboxes_to_geojson_feature_collection(
    bboxes: List[Box],
    labels: List[str],
    scores: List[float] | None = None,
    ids: List[int] | None = None,
) -> geojson.FeatureCollection:
    n = len(bboxes)
    assert len(labels) == n, "labels must have the same size as bboxes."
    assert (
        ids is None or len(ids) == n
    ), "If ids is not None, then it must have the same size as bboxes."
    assert (
        scores is None or len(scores) == n
    ), "If scores is not None, then it must have the same size as bboxes."

    features_list: List[geojson.Feature] = []
    for i in range(n):
        bbox = bbox_to_geojson_polygon(bboxes[i])
        properties: Dict[str, str | float] = {"label": labels[i]}
        if scores is not None:
            properties["scores"] = scores[i]
        id = ids[i] if ids is not None else None
        features_list.append(geojson.Feature(id=id, geometry=bbox, properties=properties))

    return geojson.FeatureCollection(features_list)


def merge_geojson_feature_collections(
    feature_collections: List[geojson.FeatureCollection],
) -> geojson.FeatureCollection:
    all_features = []
    for feature_collection in feature_collections:
        all_features.extend(feature_collection["features"])
    return geojson.FeatureCollection(all_features)


def save_geojson(geojson_object: geojson.GeoJSON, save_path: str) -> None:
    """Saves a GeoJSON object to the file with the EPSG::28992 coordinates system
    for QGIS.

    Args:
        geojson_object (geojson.GeoJSON): GeoJSON object to save.
        save_path (str): file path to save into.
    """
    geojson_object["crs"] = {
        "type": "name",
        "properties": {"name": "urn:ogc:def:crs:EPSG::28992"},
    }
    with open(save_path, "w") as file:
        geojson.dump(geojson_object, file)
