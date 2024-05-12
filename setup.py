from setuptools import setup, find_packages

setup(
    name="solomon",
    version="0.1.0",
    description="A collection of utilities for data science and machine learning.",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "numba",
        "sklearn",
        "matplotlib",
        "google-cloud-bigquery",
        "google-cloud-storage",
        "json",
        "pillow",
    ],
    extras_require={
        "gpu-only": ["cudf", "cuml", "cupy", "opencv-python"],
        "torch": ["torch", "torchvision", "torchaudio"],
        "tf": ["tensorflow", "keras"],
        "jax": ["equinox", "flax"],
        "ml": ["xgboost", "shap", "levenshtein"],
    },
)
