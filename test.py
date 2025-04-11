import bibtexparser
import re

SMALL_WORDS = re.compile(r'\b(a|an|and|as|at|but|by|for|if|in|nor|of|on|or|the|to|vs?\.?|via)\b', re.I)

def title_case(title):
    words = re.split('(\W+)', title)
    new_title = []
    for i, word in enumerate(words):
        if SMALL_WORDS.match(word) and i != 0:
            new_title.append(word.lower())
        else:
            new_title.append(word.capitalize())
    return ''.join(new_title)

def format_title(title):
    # Remove existing braces
    title_clean = re.sub(r'^\{+|\}+$', '', title)
    # Capitalize properly
    title_capitalized = title_case(title_clean)
    # Wrap with double braces
    return '{' + title_capitalized + '}'

def process_bib_file(input_bib, output_bib):
    with open(input_bib, 'r', encoding='utf-8') as bibtex_file:
        bib_database = bibtexparser.load(bibtex_file)

    for entry in bib_database.entries:
        if 'title' in entry:
            entry['title'] = format_title(entry['title'])

    with open(output_bib, 'w', encoding='utf-8') as bibtex_file:
        bibtexparser.dump(bib_database, bibtex_file)

# Usage
input_bib = 'input.bib'
output_bib = 'formatted_output.bib'
process_bib_file(input_bib, output_bib)

print(f"Processed '{input_bib}' and saved formatted entries to '{output_bib}'.")
