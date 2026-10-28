PYTHON ?= $(HOME)/miniforge3/envs/uma/bin/python
.DEFAULT_GOAL := all
-include presentation_sources.mk
.PHONY: all ncs ncs-refresh figures clean
all: MetaG.pdf
ncs: MetaG_NCS.pdf
presentation_numbers.tex: presentation_results.json make_presentation_figures.py
	$(PYTHON) make_presentation_figures.py
ncs-refresh:
	$(PYTHON) make_presentation_figures.py
	$(MAKE) ncs
MetaG_NCS.pdf: MetaG_NCS.tex presentation_numbers.tex $(wildcard sections/*.tex) $(wildcard figures/*.png)
	pdflatex -interaction=nonstopmode -halt-on-error MetaG_NCS.tex > build_ncs.log
	pdflatex -interaction=nonstopmode -halt-on-error MetaG_NCS.tex >> build_ncs.log
MetaG.pdf: MetaG.tex $(wildcard figures/*.png)
	pdflatex -interaction=nonstopmode -halt-on-error MetaG.tex > build.log
	pdflatex -interaction=nonstopmode -halt-on-error MetaG.tex >> build.log
figures:
	$(PYTHON) make_figures.py
clean:
	rm -f MetaG.aux MetaG.log MetaG.nav MetaG.out MetaG.snm MetaG.toc build.log
