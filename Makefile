.PHONY: build coverage lint graphs tests docs

build:
	python setup.py build_ext --inplace

coverage:
	nosetests --logging-level=INFO --with-coverage --cover-package=CasingSimulations --cover-html
	open cover/index.html

lint:
	pylint --output-format=html CasingSimulations > pylint.html

graphs:
	pyreverse -my -A -o pdf -p CasingSimulations CasingSimulations/**.py CasingSimulations/**/**.py

tests:
	nosetests --logging-level=INFO

docs:
	cd docs;make html

clean:
	cd docs;make clean
	find . -name "*.pyc" | xargs -I {} rm -v "{}"
