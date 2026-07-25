# Makefile for converting README.md to HTML, LaTeX, and PDF formats

# Variables
BASENAME := README
MD       := $(BASENAME).md
YAML     := $(BASENAME).yaml
HTML     := $(BASENAME).html
TEX      := $(BASENAME).tex
PDF      := $(BASENAME).pdf

# Check for pandoc
PANDOC := $(shell command -v pandoc 2> /dev/null)

# Default target
.PHONY: all
all: html latex pdf

# HTML conversion
$(HTML): $(MD) $(YAML)
ifndef PANDOC
	$(error "pandoc is not available. Please install pandoc first.")
endif
	pandoc -s -t html5 "$(YAML)" "$<" -o "$@" \
	--variable urlcolor=blue \
	--number-sections \
	--table-of-contents \
	--highlight-style kate \
	-V colorlinks \
	--toc-depth=4

# LaTeX conversion
$(TEX): $(MD) $(YAML)
ifndef PANDOC
	$(error "pandoc is not available. Please install pandoc first.")
endif
	pandoc -s -t latex "$(YAML)" "$<" -o "$@" \
	--variable urlcolor=blue \
	--number-sections \
	--table-of-contents \
	--highlight-style kate \
	-V colorlinks \
	-V geometry:"top=2cm, bottom=1.5cm, left=2cm, right=2cm" \
	--toc-depth=4

# PDF conversion
$(PDF): $(MD) $(YAML)
ifndef PANDOC
	$(error "pandoc is not available. Please install pandoc first.")
endif
	pandoc "$(YAML)" "$<" \
	-t pdf -o "$@" \
	--pdf-engine=xelatex \
	--variable urlcolor=blue \
	--number-sections \
	--table-of-contents \
	--highlight-style kate \
	-V colorlinks \
	-V geometry:"top=2cm, bottom=1.5cm, left=2cm, right=2cm" \
	--toc-depth=4



# Phony targets
.PHONY: html latex pdf clean help

html: $(HTML)
latex: $(TEX)
pdf: $(PDF)

clean:
	rm -f $(HTML) $(TEX) $(PDF)

help:
	@echo "Available targets:"
	@echo "  all    - Build all formats (HTML, LaTeX, PDF)"
	@echo "  html   - Build HTML version"
	@echo "  latex  - Build LaTeX version"
	@echo "  pdf    - Build PDF version"
	@echo "  clean  - Remove generated files"
	@echo "  help   - Show this help"
