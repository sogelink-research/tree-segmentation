import os

from chm import compute_chm, compute_dtm
from tqdm import tqdm

if __name__ == "__main__":
    # point_cloud_folder = "data/point_clouds_cropped/122000_484000"
    # chm_folder = "data/CHM_cropped_0p08/122000_484000"
    # for file_name in tqdm(os.listdir(point_cloud_folder)):
    #     input_file_path = os.path.join(point_cloud_folder, file_name)
    #     if os.path.splitext(file_name)[1] in [".laz", ".LAZ"]:
    #         output_file_name = f"{os.path.splitext(file_name)[0]}_chm.tif"
    #         output_file_path = os.path.join(chm_folder, output_file_name)
    #         if os.path.isfile(input_file_path):
    #             compute_chm(input_file_path, output_file_path, verbose=True)
    #             print(output_file_path)
    #             break
    #     elif os.path.splitext(file_name)[1] in [".las", ".LAS"]:
    #         os.remove(input_file_path)
    compute_dtm(
        "data/lidar/unfiltered/full/122000_484000_cropped.laz",
        # "data/122000_484000_filtered.tif",
        width=12500,
        height=12500,
        resolution=0.08,
        verbose=True,
    )
