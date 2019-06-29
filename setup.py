from setuptools import find_packages
from setuptools import setup


setup(
    name = 'machine-learning',
    packages = find_packages(),
    url = 'http://www.github.com/cortadocodes/machine-learning',
    author = 'Marcus Lugg',
    author_email = 'marcuslugg@googlemail.com',
    install_requires = [
        'matplotlib>=3.1',
        'numpy>=1.16',
        'pandas>=0.24',
        'sklearn'
    ],
    extras_require = {
        'development': [
            'flake8',
            'pydocstyle',
            'pytest'
        ]
    }
)
