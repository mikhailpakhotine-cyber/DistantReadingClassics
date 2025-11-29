# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is a text corpus repository containing classic Russian literature for distant reading and computational text analysis.

**Current contents:**
- `Chernyshevsky_What_Is_To_Be_Done_UTF8.txt` - Nikolai Chernyshevsky's "What Is To Be Done?" (1863)
- `pg600.txt` - Fyodor Dostoyevsky's "Notes from the Underground" from Project Gutenberg (eBook #600)

Both files are UTF-8 encoded.

## Purpose

Distant reading is a computational approach to literary analysis that focuses on patterns across large collections of texts rather than close reading of individual works. This corpus provides source texts for such analysis.

## Current State

This is a minimal text corpus repository with no build system, tests, or analysis scripts yet. When adding analysis capabilities, note that `pg600.txt` includes Project Gutenberg headers/footers that may need to be removed during preprocessing.
