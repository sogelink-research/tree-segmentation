from typing import List

import geojson
from data_processing import Box


def bbox_pixels_to_coordinates(
    bbox: Box, image_pixels_box: Box, image_coordinates_box: Box
) -> Box:
    x_factor = (image_coordinates_box.x_max - image_coordinates_box.x_min) / (
        image_pixels_box.x_max - image_pixels_box.x_min
    )
    y_factor = (image_coordinates_box.y_max - image_coordinates_box.y_min) / (
        image_pixels_box.y_max - image_pixels_box.y_min
    )

    new_x_min = (
        bbox.x_min - image_pixels_box.x_min
    ) * x_factor + image_coordinates_box.x_min
    new_x_max = (
        bbox.x_max - image_pixels_box.x_min
    ) * x_factor + image_coordinates_box.x_min

    new_y_min = (
        image_pixels_box.y_max - bbox.y_max
    ) * y_factor + image_coordinates_box.y_min
    new_y_max = (
        image_pixels_box.y_max - bbox.y_min
    ) * y_factor + image_coordinates_box.y_min

    new_box = Box(x_min=new_x_min, y_min=new_y_min, x_max=new_x_max, y_max=new_y_max)
    return new_box


def bbox_coordinates_to_pixels(
    bbox: Box, image_coordinates_box: Box, image_pixels_box: Box
) -> Box:
    x_factor = (image_pixels_box.x_max - image_pixels_box.x_min) / (
        image_coordinates_box.x_max - image_coordinates_box.x_min
    )
    y_factor = (image_pixels_box.y_max - image_pixels_box.y_min) / (
        image_coordinates_box.y_max - image_coordinates_box.y_min
    )

    new_x_min = (
        bbox.x_min - image_coordinates_box.x_min
    ) * x_factor + image_pixels_box.x_min
    new_x_max = (
        bbox.x_max - image_coordinates_box.x_min
    ) * x_factor + image_pixels_box.x_min

    new_y_min = (
        image_coordinates_box.y_max - bbox.y_max
    ) * y_factor + image_pixels_box.y_min
    new_y_max = (
        image_coordinates_box.y_max - bbox.y_min
    ) * y_factor + image_pixels_box.y_min

    new_box = Box(x_min=new_x_min, y_min=new_y_min, x_max=new_x_max, y_max=new_y_max)
    return new_box


def bbox_to_geojson_polygon(bbox: Box) -> geojson.Polygon:
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
    ids: List[int] | None = None,
):
    n = len(bboxes)
    assert len(labels) == n, "labels must have the same size as bboxes."
    assert (
        ids is None or len(ids) == n
    ), "If ids is not None, then it must have the same size as bboxes."

    features_list: List[geojson.Feature] = []
    for i in range(n):
        bbox = bbox_to_geojson_polygon(bboxes[i])
        label = labels[i]
        id = ids[i] if ids is not None else None
        features_list.append(
            geojson.Feature(id=id, geometry=bbox, properties={"label": label})
        )

    return geojson.FeatureCollection(features_list)


def save_geojson(geojson_object: geojson.GeoJSON, save_path: str) -> None:
    geojson_object["crs"] = {
        "type": "name",
        "properties": {"name": "urn:ogc:def:crs:EPSG::28992"},
    }
    with open(save_path, "w") as file:
        geojson.dump(geojson_object, file)


def main():
    bbox = Box(0, 80, 160, 240)
    image_pixels_box = Box(0, 0, 640, 640)
    image_coordinates_box = Box(122000, 483000, 123000, 484000)
    bbox_coordinates = bbox_pixels_to_coordinates(
        bbox=bbox,
        image_pixels_box=image_pixels_box,
        image_coordinates_box=image_coordinates_box,
    )
    print(f"{bbox_coordinates = }")
    bbox_pixels = bbox_coordinates_to_pixels(
        bbox=bbox_coordinates,
        image_pixels_box=image_pixels_box,
        image_coordinates_box=image_coordinates_box,
    )
    print(f"{bbox_pixels = }")

    save_geojson(bbox_to_geojson_polygon(bbox_coordinates), "output.geojson")

    # Now, let's read the data back from the file to verify
    with open("output.geojson", "r") as file:
        geojson_data = geojson.load(file)
    print(geojson_data)


if __name__ == "__main__":
    main()
