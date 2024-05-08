from os.path import abspath

from requests import get


def download_file(url: str, save_path: str):
    # Send a GET request to the URL
    response = get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Open the file in binary write mode and write the content of the response
        print(f"Downloading {url}...", end=" ", flush=True)
        with open(save_path, "wb") as f:
            f.write(response.content)
        print(f"Done.\nSaved at '{abspath(save_path)}'.")
    else:
        print(
            f"Failed to download file from '{url}'. Status code: {response.status_code}"
        )
