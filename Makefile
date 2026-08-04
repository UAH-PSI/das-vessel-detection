# Makefile for converting README.md to HTML, LaTeX, and PDF formats

# Variables
BASENAME := README
MD       := $(BASENAME).md
YAML     := docs/$(BASENAME).yaml
BUILD_DIR := build
HTML     := $(BUILD_DIR)/$(BASENAME).html
TEX      := $(BUILD_DIR)/$(BASENAME).tex
PDF      := $(BUILD_DIR)/$(BASENAME).pdf

# Compilation-safe public README/PDF. Shields.io serves SVG by default, but
# PNG assets are used here because they can be embedded directly by XeLaTeX.
PUBLIC_BASENAME := README-public
PUBLIC_MD       := $(BUILD_DIR)/$(PUBLIC_BASENAME).md
PUBLIC_PDF      := $(BUILD_DIR)/$(PUBLIC_BASENAME).pdf
PUBLIC_TITLE    := $(shell sed -n '1s/^# //p' $(MD))
BADGE_DIR       := $(BUILD_DIR)/readme-badges
BADGE_LINK_DIR  := readme-badges
JSTARS_BADGE    := $(BADGE_DIR)/jstars.png
DATASET_BADGE   := $(BADGE_DIR)/dataset.png
ARXIV_BADGE     := $(BADGE_DIR)/arxiv.png
LICENSE_BADGE   := $(BADGE_DIR)/license.png
BADGES          := $(JSTARS_BADGE) $(DATASET_BADGE) $(ARXIV_BADGE) $(LICENSE_BADGE)

# Check for pandoc
PANDOC := $(shell command -v pandoc 2> /dev/null)

# Default target
.PHONY: all
all: html latex pdf

# HTML conversion
$(BUILD_DIR):
	mkdir -p "$@"

$(HTML): $(MD) $(YAML) | $(BUILD_DIR)
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
$(TEX): $(MD) $(YAML) | $(BUILD_DIR)
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
$(PDF): $(MD) $(YAML) | $(BUILD_DIR)
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



# Download local, PDF-safe versions of the badges used at the top of README.md.
$(BADGE_DIR):
	mkdir -p "$@"

$(JSTARS_BADGE): | $(BADGE_DIR)
	curl -fL --retry 3 \
	  'https://img.shields.io/badge/IEEE%20JSTARS%20%28accepted%29-10.1109%2FJSTARS.2026.3716768-00629B.png' \
	  -o "$@"

$(DATASET_BADGE): | $(BADGE_DIR)
	curl -fL --retry 3 \
	  'https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.15611778-1682D4.png' \
	  -o "$@"

$(ARXIV_BADGE): | $(BADGE_DIR)
	curl -fL --retry 3 \
	  'https://img.shields.io/badge/ArXiV%20Preprint-submitted%20to%20Scientific%20Data-orange.png' \
	  -o "$@"

$(LICENSE_BADGE): | $(BADGE_DIR)
	curl -fL --retry 3 \
	  'https://img.shields.io/badge/License-GPLv3-blue.png' \
	  -o "$@"

# Produce a public Markdown copy with local badges and PDF-friendly headings.
# The source title becomes PDF metadata, so omit it from the Markdown body;
# likewise omit the hand-written TOC because Pandoc generates its own.
$(PUBLIC_MD): $(MD) $(BADGES) | $(BUILD_DIR)
	sed -E \
	  -e '1d' \
	  -e '/^## Table of contents$$/,/^## Project summary$$/ { /^## Project summary$$/!d; }' \
	  -e 's/^#(#+ )/\1/' \
	  -e 's|https://img.shields.io/badge/IEEE%20JSTARS%20%28accepted%29-10.1109%2FJSTARS.2026.3716768-00629B|$(BADGE_LINK_DIR)/jstars.png|g' \
	  -e 's|https://img.shields.io/badge/Zenodo-10.5281%2Fzenodo.15611778-1682D4|$(BADGE_LINK_DIR)/dataset.png|g' \
	  -e 's|https://img.shields.io/badge/ArXiV%20Preprint-submitted%20to%20Scientific%20Data-orange|$(BADGE_LINK_DIR)/arxiv.png|g' \
	  -e 's|https://img.shields.io/badge/License-GPLv3-blue.svg|$(BADGE_LINK_DIR)/license.png|g' \
	  -e 's|\]\(docs/|](../docs/|g' \
	  -e 's|\]\(LICENSE\)|](../LICENSE)|g' \
	  -e 's|\]\(data/|](../data/|g' \
	  -e 's|\]\(logos/|](../logos/|g' \
	  -e 's|(!\[[^]]*\]\([^)]*readme-badges/[^)]*\.png\))|\1{height=15px}|g' \
	  -e '/readme-badges\/license\.png.*\]\(\.\.\/LICENSE\)$$/a\
```{=latex}\n\\newpage\n```' \
	  "$<" > "$@.tmp"
	mv "$@.tmp" "$@"

$(PUBLIC_PDF): $(PUBLIC_MD) $(YAML)
ifndef PANDOC
	$(error "pandoc is not available. Please install pandoc first.")
endif
	pandoc "$<" \
	-t pdf -o "$@.tmp" \
	--pdf-engine=xelatex \
	--resource-path="$(BUILD_DIR)" \
	--metadata-file "$(YAML)" \
	--variable urlcolor=blue \
	--metadata title="$(PUBLIC_TITLE)" \
	--number-sections \
	--table-of-contents \
	--highlight-style kate \
	-V colorlinks \
	-V geometry:"top=2cm, bottom=1.5cm, left=2cm, right=2cm" \
	--toc-depth=4
	mv "$@.tmp" "$@"

# Phony targets
.PHONY: html latex pdf public-pdf clean help

html: $(HTML)
latex: $(TEX)
pdf: $(PDF)

# Build through build/README-public.md, then publish the result under the
# canonical build/README.pdf name. mv ensures a failed compilation leaves it intact.
public-pdf: $(PUBLIC_PDF)
	mv "$(PUBLIC_PDF)" "$(PDF)"

clean:
	rm -rf "$(BUILD_DIR)"

help:
	@echo "Available targets:"
	@echo "  all    - Build all formats under build/"
	@echo "  html   - Build build/README.html"
	@echo "  latex  - Build build/README.tex"
	@echo "  pdf    - Build build/README.pdf"
	@echo "  public-pdf - Build build/README-public.md and publish build/README.pdf"
	@echo "  clean  - Remove generated files"
	@echo "  help   - Show this help"
