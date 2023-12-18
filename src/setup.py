import setuptools

with open("hyphai/README.md", "r") as fh:
    long_description = fh.read()


if __name__ == "__main__":
    setuptools.setup(
        name="hyphai",
        version="1.0.0",
        author="Rachid El Montassir ",
        author_email="elmontassir@cerfacs.fr",
        description="Hybrid Physics-AI Model for Cloud Cover Nowcasting",
        long_description=long_description,
        long_description_content_type="text/markdown",
        url="",
        packages=setuptools.find_packages(),
        classifiers=[
            "Programming Language :: Python :: 3",
            "License :: OSI Approved :: CeCILL License",
            "Operating System :: OS Independent",
        ],
    )