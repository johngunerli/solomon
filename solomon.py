import subprocess
from numba import cuda

# GPU Detection
gpu = False
try:
    # Try to check if nvcc compiler is available which is part of CUDA toolkit
    nvcc_output = subprocess.check_output("nvcc --version", shell=True).decode()
    # print(nvcc_output)
    print("GPU detected, using GPU acceleration wherever possible")
    gpu_available = True
except Exception as e:
    print(f"No GPU detected, defaulting to CPU-optimized processes: {e}")

if gpu_available:
    import cudf as pd
else:
    import pandas as pd


def bq_to_df(
    query, project_id="your-project-id"
):  # FIXME: Provide default project ID here
    """Create a dataframe based on a query, defaults to the specified project environment.
    Args:
        query (str): SQL query to execute.
        project_id (str): Google Cloud Project ID.
    Returns:
        pd.DataFrame: Query results as a DataFrame.
    """
    from google.cloud import bigquery as bq

    client = bq.Client(project=project_id)
    query_job = client.query(query)
    results = query_job.result()
    return results.to_dataframe()


def blob_to_df(
    blob_name, bucket_name="your-default-bucket"
):  # Provide your default bucket name here
    """Download a blob or a list of blobs from a GCS bucket.
    Args:
        blob_name (str or list): Name(s) of the blob(s) to download.
        bucket_name (str): Name of the GCS bucket.
    Returns:
        bytes or list: Blob content as bytes or list of blob contents.
    """
    from google.cloud import storage

    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    if isinstance(blob_name, str):
        blob = bucket.blob(blob_name)
        return blob.download_as_bytes()
    elif isinstance(blob_name, list):
        list_of_blobs = [bucket.blob(name).download_as_bytes() for name in blob_name]
        return list_of_blobs
    else:
        print("Invalid blob_name type; it must be a string or a list.")
        return None


# Download blobs to a local directory
def download_blob_to_directory(blob_name, idx, bucket_name="your-default-bucket"):
    """Download a specific blob to a local directory.
    Args:
        blob_name (str): Name of the blob to download.
        idx (int): Index number for naming the file.
        bucket_name (str): Name of the GCS bucket.
    """
    gcs_client = storage.Client()
    bucket = gcs_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    destination_file_name = os.path.join("clip_image", f"image_{idx + 1}.jpeg")
    os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
    blob.download_to_filename(destination_file_name)


def resize_image(input_img, new_width, new_height):
    """
    Resize an image to a new width and height, using GPU acceleration if available.
    Args:
        input_img (np.array): Image to resize.
        new_width (int): New width for the image.
        new_height (int): New height for the image.

    Raises:
        ValueError: If input_img is not of type np.uint8.

    Returns:
        np.array: Resized image.
    """
    if gpu_available:
        import ctypes
        import numpy as np
        import os

        # execute the cuda compilation
        # os.system("nvcc -o resize.so --shared --compiler-options '-fPIC' resize.cu")
        lib = ctypes.CDLL("./resize.so")
        lib.resize_image.argtypes = [
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.POINTER(ctypes.c_ubyte),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        ]
        in_height, in_width = input_img.shape[:2]
        channels = input_img.shape[2] if input_img.ndim == 3 else 1
        output_img = (
            np.zeros((new_height, new_width, channels), dtype=np.uint8)
            if channels > 1
            else np.zeros((new_height, new_width), dtype=np.uint8)
        )

        if input_img.dtype != np.uint8:
            raise ValueError("input_img must be an array of type np.uint8")

        lib.resize_image(
            input_img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            output_img.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte)),
            in_height,
            in_width,
            new_height,
            new_width,
            channels,
        )
        # delete the shared library
        # os.system("rm resize.so")
        return output_img
    else:
        return input_img.resize((new_width, new_height), Image.NEAREST)
    return None


def convert_image(input_image_path, output_image_path=None):
    from PIL import Image
    import os

    if output_image_path is None:
        base_name = os.path.splitext(input_image_path)[0]
        output_image_path = f"{base_name}.jpeg"
        print(
            f"No output path provided. Converting image to JPEG format at: {output_image_path}"
        )
    file_extension = os.path.splitext(input_image_path)[1].lower()

    if file_extension == ".heic":
        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
        except ImportError:
            import sys

            print(
                "pillow-heif package not found. Please install it using 'pip install pillow-heif' to handle HEIC files."
            )
            sys.exit(1)

    with Image.open(input_image_path) as img:
        img = img.convert("RGB") if img.mode != "RGB" else img
        if output_image_path.endswith(".jpeg") or output_image_path == None:

            img.save(output_image_path, "JPEG")
        else:
            img.save(output_image_path)
