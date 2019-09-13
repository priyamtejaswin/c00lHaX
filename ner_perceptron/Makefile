clean:
	rm *.so
	rm *.pyc
	rm *.c

data:
	if [ ! -f brown.zip  ]; then wget https://archive.org/download/BrownCorpus/brown.zip -O brown.zip; fi

	if [ ! -d "$dir" ]; then unzip brown.zip; fi

test:
	python setup.py build_ext --inplace
	python test.py

run:
	python collins_perceptron.py brown/
