all:
	python setup.py build_ext -i

# coverage:
# 	nosetests --with-coverage --cover-html --cover-package=pystruct pystruct

# test-code: all
# 	$(NOSETESTS) -s -v pystruct


clean:
	find | grep .pyc | xargs rm -rf

# test-doc:
# 	$(NOSETESTS) -s -v doc/*.rst doc/modules/

test: 
	pytest

cleanexps:

	rm -rf logs/*