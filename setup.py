from setuptools import setup, find_packages

setup(
    name="neural-avatar-pipeline",
    version="0.1.0",
    author="Soumya Kakani",
    author_email="kakanisoumya1@gmail.com",
    description="Fast neural human avatar reconstruction pipeline",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.9",
)
