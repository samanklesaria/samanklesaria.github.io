.SECONDARY:

objects := $(shell find notebooks -maxdepth 1 -type f -not -path '*/.*' | sed -r 's_notebooks/([^\.]+).*_content/\1.md_')

assets := $(shell find notebooks -maxdepth 1 -type d -not -path '*/.*' | sed -r 's_notebooks/(.+)_content/\1_')

main : $(objects) $(assets)
	pelican content

content/%.md: _notebooks/%.ipynb
	jupyter nbconvert --to markdown $< --output-dir=content --template=md_template

content/%.md: notebooks/%.md
	cp $< $@

_notebooks/%.ipynb: notebooks/%.ipynb
	mkdir -p _notebooks
	cp $< $@

_notebooks/%.ipynb: notebooks/%.jl
	julia --project=. -e 'using PlutoNB; PlutoNB.jl2nb("$<", "$@")'

content/%: notebooks/%
	cp -r $< $@

clean:
	rm -rf _notebooks content output
