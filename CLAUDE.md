# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a text corpus repository containing classic Russian literature for distant reading and computational text analysis. The repository currently contains:

- `Chernyshevsky_What_Is_To_Be_Done_UTF8.txt` - Nikolai Chernyshevsky's "What Is To Be Done?" (1863) in UTF-8 encoding
- `pg600.txt` - Fyodor Dostoyevsky's "Notes from the Underground" from Project Gutenberg (eBook #600)

## Purpose

Distant reading is a computational approach to literary analysis that focuses on patterns across large collections of texts rather than close reading of individual works. This corpus provides source texts for such analysis.

## Working with the Texts

### File Encoding
- `Chernyshevsky_What_Is_To_Be_Done_UTF8.txt` is explicitly UTF-8 encoded
- `pg600.txt` is UTF-8 encoded (Project Gutenberg standard)

### Common Operations

When adding analysis capabilities to this repository, typical tasks include:

**Text preprocessing:**
- Remove Project Gutenberg headers/footers from `pg600.txt`
- Normalize text encoding if needed
- Extract clean text content

**Analysis examples:**
- Word frequency analysis
- Sentiment analysis
- Named entity recognition
- Topic modeling
- Stylometric analysis
- Comparative analysis between the two works

### Adding Analysis Scripts

If adding Python analysis scripts, typical dependencies would include:
```bash
pip install nltk spacy pandas matplotlib numpy
```

If adding R analysis scripts, typical dependencies would include:
```r
install.packages(c("tidyverse", "tidytext", "quanteda"))
```

## Repository Structure

This is currently a minimal corpus repository. Analysis scripts, notebooks, and results would typically be organized into:
- `scripts/` - Analysis scripts
- `notebooks/` - Jupyter/R Markdown notebooks
- `data/processed/` - Cleaned/preprocessed texts
- `results/` - Analysis outputs
