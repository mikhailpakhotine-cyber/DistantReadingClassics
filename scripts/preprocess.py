#!/usr/bin/env python3
"""
Text preprocessing script for distant reading analysis.
Cleans and prepares texts for analysis.
"""

import re
import os


def remove_gutenberg_header_footer(text):
    """Remove Project Gutenberg header and footer from text."""
    # Find the start of actual content (after the header)
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "*END*THE SMALL PRINT",
    ]

    start_pos = 0
    for marker in start_markers:
        pos = text.find(marker)
        if pos != -1:
            # Find the end of this line and the next line
            newline_pos = text.find('\n', pos)
            if newline_pos != -1:
                # Skip to after the next few newlines to get past the marker
                start_pos = text.find('\n\n', newline_pos)
                if start_pos != -1:
                    start_pos += 2
                    break

    # Find the end of actual content (before the footer)
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "End of the Project Gutenberg EBook",
        "End of Project Gutenberg's",
    ]

    end_pos = len(text)
    for marker in end_markers:
        pos = text.find(marker)
        if pos != -1:
            end_pos = pos
            break

    if start_pos > 0 or end_pos < len(text):
        text = text[start_pos:end_pos]

    return text.strip()


def clean_text(text):
    """Basic text cleaning while preserving structure."""
    # Remove excessive whitespace while preserving paragraph breaks
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove excessive spaces
    text = re.sub(r' {2,}', ' ', text)
    # Remove leading/trailing whitespace from lines
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    return text.strip()


def preprocess_file(input_path, output_path, remove_pg_header=False):
    """Preprocess a single text file."""
    print(f"Processing {input_path}...")

    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    if remove_pg_header:
        text = remove_gutenberg_header_footer(text)

    text = clean_text(text)

    # Save processed text
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"Saved processed text to {output_path}")
    print(f"Length: {len(text)} characters, {len(text.split())} words\n")

    return text


def main():
    """Preprocess all texts in the corpus."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Process Dostoyevsky (remove PG header/footer)
    preprocess_file(
        os.path.join(base_dir, 'pg600.txt'),
        os.path.join(base_dir, 'data/processed/dostoyevsky_notes_from_underground.txt'),
        remove_pg_header=True
    )

    # Process Chernyshevsky (already clean, just normalize)
    preprocess_file(
        os.path.join(base_dir, 'Chernyshevsky_What_Is_To_Be_Done_UTF8.txt'),
        os.path.join(base_dir, 'data/processed/chernyshevsky_what_is_to_be_done.txt'),
        remove_pg_header=False
    )

    print("Preprocessing complete!")


if __name__ == '__main__':
    main()
