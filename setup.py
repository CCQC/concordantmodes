import setuptools
import versioneer

short_description = "The Concordant Mode Approach utilizes lower level of theory normal modes to compute higher level force constants at a fraction of the cost!"

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except:
    long_description = short_description

if __name__ == "__main__":
    setuptools.setup(
        name="Concordant Modes",
        description="Uses normal modes at a lower level of theory to compute higher level force constants",
        authors="Dr. Mitchell  Lahm, Nate Kitzmiller, Prof Wesley D. Allen",
        url="https://github.com/CCQC/ConcordantModes",
        license="BSD-3C",
        version=versioneer.get_version(),
        cmdclass=versioneer.get_cmdclass(),
        packages=setuptools.find_packages(),
        include_package_data=True,
        install_requires=[
            "numpy>=1.13.3",
            "qcelemental>=0.24.0",
            "sympy",
            "scipy >= 1.4"
        ],
        extras_require={
            "tests": ["pytest", "pytest-cov", "pytest-pep8",],
        },
        tests_require=["pytest", "pytest-cov", "pytest-pep8",],
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Science/Research",
            "Programming Language :: Python :: 3",
        ],
        zip_safe=False,
        long_description=long_description,
        long_description_content_type="text/markdown",
    )
