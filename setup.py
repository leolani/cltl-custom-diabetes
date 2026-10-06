from setuptools import setup, find_namespace_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

with open("VERSION", "r") as fh:
    version = fh.read().strip()

setup(
    name='cltl.diabetes',
    version=version,
    package_dir={'': 'src'},
    # Only `cltl.diabetes`. The code copied from cltl-kg-driven-chat (the
    # other packages under src/cltl, and src/kg_chat) is not packaged: it
    # imports its modules by bare name and locates its files relative to the
    # working directory, so it runs from src/ in this checkout (or the image),
    # the same way it ran from the notebooks/ directory in cltl-kg-driven-chat.
    packages=find_namespace_packages(include=['cltl.diabetes', 'cltl.diabetes.*'], where='src'),
    data_files=[('VERSION', ['VERSION'])],
    url="https://github.com/leolani/cltl-custom-diabetes",
    license='MIT License',
    author='CLTL',
    description='Knowledge graph driven diabetes lifestyle coach for a Leolani deployment',
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires='>=3.10',
    install_requires=[
        'cltl.combot',
        'emissor',
        'cltl.brain',
        'openai',
        'pydantic',
        'rdflib',
        'sparqlwrapper',
        'python-dateutil',
        'requests',
        'tqdm',
    ],
    extras_require={
        "service": [
            "cltl.combot[external]",
            "flask",
            "werkzeug",
        ],
        # The notebooks in notebooks/, which run the original application
        # without the platform. Its GUI also needs tkinter, which comes with
        # Python rather than from PyPI.
        "notebooks": [
            "jupyter",
            "pandas",
        ]}
)
