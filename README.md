# DistantReadingClassics

A computational text analysis project for distant reading of Russian classic literature.

## Overview

This repository contains classic Russian literature texts and tools for performing comprehensive distant reading analysis, including sentiment analysis, stylometric comparison, and interactive visualizations.

## Texts Included

- **Dostoyevsky** - Notes from the Underground (Project Gutenberg #600)
- **Chernyshevsky** - What Is To Be Done? (1863)

## Analysis Features

- **Word Frequency Analysis** - Top words, content words, and phrase extraction
- **Vocabulary Richness** - Type-token ratio, lexical diversity, hapax legomena
- **Sentiment Analysis** - Overall sentiment scoring for each text
- **Stylometric Analysis** - Part-of-speech distribution, sentence/word length statistics
- **Readability Metrics** - Flesch Reading Ease and Flesch-Kincaid Grade Level
- **Named Entity Recognition** - Extraction of potential named entities
- **Comparative Analysis** - Side-by-side comparison of both texts

## Quick Start

### 1. Preprocess the Texts

```bash
python3 scripts/preprocess.py
```

This removes Project Gutenberg headers/footers and normalizes the text formatting.

### 2. Run the Analysis

```bash
python3 scripts/analyze.py
```

This performs comprehensive analysis and generates `results/analysis.json`.

### 3. View the Interactive Visualization

Open `results/index.html` in a web browser to explore:
- Word clouds for each text
- Sentiment and readability metrics
- Part-of-speech distribution charts
- Side-by-side comparative analysis
- Statistical comparisons with overlay visualizations

## Project Structure

```
DistantReadingClassics/
├── Chernyshevsky_What_Is_To_Be_Done_UTF8.txt  # Original text
├── pg600.txt                                   # Original text (Dostoyevsky)
├── scripts/
│   ├── preprocess.py                          # Text preprocessing
│   └── analyze.py                             # Comprehensive analysis
├── data/
│   └── processed/                             # Cleaned text files
├── results/
│   ├── analysis.json                          # Analysis results (JSON)
│   └── index.html                             # Interactive visualization
└── CLAUDE.md                                   # Repository guidance for Claude Code
```

## Technologies Used

- **Python 3** - Text processing and analysis
- **HTML/CSS/JavaScript** - Interactive web visualization
- **Chart.js** - Data visualization charts
- **WordCloud2.js** - Word cloud generation

## Analysis Methods

All analysis is performed using custom Python implementations:
- Simple sentiment analysis using positive/negative word lexicons
- Syllable-based readability scoring
- Pattern-based part-of-speech estimation
- Statistical text metrics (TTR, lexical diversity, etc.)

For production use, consider integrating professional NLP libraries like NLTK, spaCy, or TextBlob for more sophisticated analysis.