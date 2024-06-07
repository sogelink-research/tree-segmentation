import os
from copy import deepcopy
from typing import Dict, List, Sequence, Tuple

import geojson
import geojson.utils as geo_utils
import numpy as np

from box_cls import Box


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
    scores: Sequence[float | None] | None = None,
    ids: Sequence[int | None] | None = None,
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
        properties: Dict[str, str | float | None] = {"label": labels[i]}
        if scores is not None:
            properties["score"] = scores[i]
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
    if "crs" not in geojson_object:
        geojson_object["crs"] = {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:EPSG::28992"},
        }
    with open(save_path, "w") as file:
        geojson.dump(geojson_object, file)


def open_geojson(geojson_path: str) -> geojson.GeoJSON:
    with open(geojson_path) as file:
        geo_file = geojson.load(file)
    return geo_file


def open_geojson_feature_collection(geojson_path: str) -> geojson.FeatureCollection:
    geo_file = open_geojson(geojson_path)
    assert isinstance(
        geo_file, geojson.FeatureCollection
    ), f"{geojson_path} is not a GeoJSON FeatureCollection."
    return geo_file


def get_bbox_polygon(polygon: geojson.Polygon) -> Box:
    """Compute the bounding box of a GeoJSON Polygon.

    Args:
        polygon (geojson.Polygon): polygon to compute the bounding box of.

    Returns:
        Box: bounding box.
    """
    # Extract coordinates from the Polygon
    coords = np.array(list(geo_utils.coords(polygon)))
    # Calculate the minimum and maximum latitudes and longitudes
    min_lat = coords[:, 1].min()
    max_lat = coords[:, 1].max()
    min_lon = coords[:, 0].min()
    max_lon = coords[:, 0].max()
    return Box(min_lon, min_lat, max_lon, max_lat)


def add_label(
    feature_coll: geojson.FeatureCollection,
    label_attribute_name: str | None,
    label_all: str | None = None,
) -> geojson.FeatureCollection:
    """Adds a label attribute to each feature of a GeoJSON FeatureCollection.
    If there is already a label attribute, its name should be put as label_attribute_name
    and it will be copied to a new "label" attribute. Otherwise, the same label
    label_all will be given to every feature.

    Args:
        feature_coll (geojson.FeatureCollection): collection of boxes.
        label_attribute_name (str | None): name of the feature attribute containing the
        label name.
        label_all (str | None, optional): label name to put for boxes without any
        label yet. Defaults to None.

    Raises:
        ValueError: _description_
        ValueError: _description_
        ValueError: _description_

    Returns:
        geojson.FeatureCollection: new FeatureCollection with "label" attribute.
    """

    features_coll_modified = deepcopy(feature_coll)

    for feature in features_coll_modified["features"]:
        # Get the class label
        if label_attribute_name is None:
            if label_all is None:
                raise ValueError("If label_attribute_name is None, label_all should be specified.")
            label = label_all
        elif label_attribute_name not in feature["properties"]:
            if label_all is None:
                raise ValueError(
                    f'If label_attribute_name="{label_attribute_name}" is not a property of each feature, label_all should be specified.'
                )
            label = label_all
        else:
            label = feature["properties"][label_attribute_name]
            if isinstance(label, list):
                if len(label) == 1:
                    label = label[0]
                else:
                    raise ValueError(f"There should only be one label per box, {label} is wrong.")

        feature["properties"]["label"] = label

    return features_coll_modified


def multi_pol_to_pol(
    feature_coll: geojson.FeatureCollection,
) -> geojson.FeatureCollection:
    """Transforms MultiPolygons into Polygons

    Args:
        feature_coll (geojson.FeatureCollection): collection of boxes.

    Raises:
        ValueError: raises an Error if it ever encounters a real MultiPolygon containing
        multiple polygons.

    Returns:
        geojson.FeatureCollection: collection of boxes after transformation.
    """

    features_coll_modified = deepcopy(feature_coll)
    for feature in features_coll_modified["features"]:
        feature["geometry"] = geojson.Polygon(geo_utils.coords(feature["geometry"]))
        feature["geometry"]["coordinates"] = [feature["geometry"]["coordinates"]]

    return features_coll_modified


def extract_bboxes_geojson(
    feature_coll: geojson.FeatureCollection,
    label_attribute_name: str | None,
    label_all: str | None = None,
) -> Tuple[List[Box], List[str], List[float | None], List[int | None]]:
    boxes: List[Box] = []
    labels: List[str] = []
    scores: List[float | None] = []
    ids: List[int | None] = []

    for feature in feature_coll["features"]:
        # Get the box
        boxes.append(get_bbox_polygon(feature["geometry"]))

        # Get the class label
        if label_attribute_name is None:
            if label_all is None:
                raise ValueError("If label_attribute_name is None, label_all should be specified.")
            label = label_all
        elif label_attribute_name not in feature["properties"]:
            if label_all is None:
                raise ValueError(
                    f'If label_attribute_name="{label_attribute_name}" is not a property of each feature, label_all should be specified.'
                )
            label = label_all
        else:
            label = feature["properties"][label_attribute_name]
            if isinstance(label, list):
                if len(label) == 1:
                    label = label[0]
                else:
                    raise ValueError(f"There should only be one label per box, {label} is wrong.")

        labels.append(label)

        # Get the scores if they exist
        if "score" in feature["properties"]:
            scores.append(feature["properties"]["score"])
        else:
            scores.append(None)

        # Get the ids if they exist
        if "id" in feature["properties"]:
            ids.append(feature["properties"]["id"])
        else:
            ids.append(None)

    return boxes, labels, scores, ids


def merge_qgis_geojson_annotations(folders_paths: str | List[str], output_path: str) -> None:
    """Merges multiple GeoJSON FeatureCollection files into one file.
    In each file, every feature which doesn't have a "label" attribute
    gets a new one with the name of the original file.
    Moreover, the

    Args:
        folder_path (str): path(s) of the folder(s) containing the files
        to merge.
        output_path (str): path to save the output.
    """

    if isinstance(folders_paths, str):
        folders_paths = [folders_paths]

    feature_colls: List[geojson.FeatureCollection] = []
    for folder_path in folders_paths:
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)

            feature_coll = open_geojson(file_path)
            if not isinstance(feature_coll, geojson.FeatureCollection):
                raise TypeError("Every GeoJSON file should be a FeatureCollection.")

            label_attribute_name = "label"
            label_all = os.path.splitext(file_name)[0]
            feature_coll = add_label(feature_coll, label_attribute_name, label_all=label_all)
            feature_coll = multi_pol_to_pol(feature_coll)

            feature_colls.append(feature_coll)

    # geo_features = bboxes_to_geojson_feature_collection(boxes_all, labels_all, scores_all, ids_all)
    geo_features = merge_geojson_feature_collections(feature_colls)
    save_geojson(geo_features, output_path)


# def intersection_polygons(poly1: geojson.Polygon, poly2: geojson.Polygon):


def geojson_add_area_of_interest(
    feature_coll: geojson.FeatureCollection, polygon: geojson.Polygon
) -> None:
    feature_coll["bbox"] = polygon
