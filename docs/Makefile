# Generic makefile for Sphinx documentation

all: html

html:
	@python3 -m sphinx.cmd.build -b html $(SOURCEDIR) $(BUILDDIR)/html

clean:
	rm -rf $(BUILDDIR)
