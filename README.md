MPRI - Anomaly Detection
========================

Welcome to the practical work for the lecture "Anomaly Detection with SVM". We'll practice one-class SVMs for novelty and outlier detection. Enjoy.


## Dependencies

- [python 3](https://www.python.org/downloads) (>= 3.5 is better)
- [pip](https://pip.pypa.io/en/stable/installing/)
- virtualenv (`pip install virtualenv`)


## Installation

Create a virtual environment with Python 3

	virtualenv -p /usr/bin/python3.6 venv

Note: if you are on a Mac, your path may be located at `/usr/local/bin/` instead of `/usr/bin/`

Enter the virtual environment

	source venv/bin/activate

Install dependencies

	pip install -r requirements.txt


## Usage

To run the code

	python ex1-plot_oneclass.py
	python ex2-breast-cancer.py

To exit the virtual environment

	deactivate


## Alternative

You can use `pipenv` if you prefer. First install pipenv, then

    pipenv --three
    pipenv install
    pipenv run python ex1-plot_oneclass.py
    pipenv run python ex2-breast-cancer.py

But you might have problems with matplotlib. In this case, install it manually on your system.

Links:

- [Virtualenvs in Python](http://docs.python-guide.org/en/latest/dev/virtualenvs)
- [pipenv](https://github.com/kennethreitz/pipenv)