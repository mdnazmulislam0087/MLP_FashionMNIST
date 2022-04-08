from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

## edit below variables as per your requirements -
REPO_NAME = "MLP_FashionMNIST"
AUTHOR_USER_NAME = "mdnazmulislam0087"
SRC_REPO = "src"
LIST_OF_REQUIREMENTS = ["numpy",
        "tensorflow",
        "matplotlib",
        "seaborn",
        "pandas",
        "PyYAML"]


setup(
    name=SRC_REPO,
    version="0.0.2",
    author=AUTHOR_USER_NAME,
    description="A small package for MLflow app",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    author_email="md.nazmul.islam0087@gmail.com",
    packages=[SRC_REPO],
    license="MIT",
    python_requires=">=3.7",
    install_requires=LIST_OF_REQUIREMENTS
)