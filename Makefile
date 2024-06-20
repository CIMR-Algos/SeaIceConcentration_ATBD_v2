


all:
	jupyter-book build --all .

pdf:
	jupyter-book build --all . --builder pdflatex

clean:
	rm -rf ./_build
