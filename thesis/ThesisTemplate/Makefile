all: main.pdf

main.pdf: main.tex
	latexmk -pdf $^

clean:
	rm -f main.aux main.glo main.log main.out main.pdf main.tcp main.toc main.bbl main.blg main.fdb_latexmk main.fls

.PHONY: all clean
