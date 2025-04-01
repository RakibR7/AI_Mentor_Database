import re
import os
import sys

 #inludes removing repeated lines and (Hyphenation, Multi-Columns, etc.)

########################################
# 1) Remove repeated lines
########################################

def remove_repeated_lines(input_file, temp_file, noise_phrases, debug=False):
    """
    Removes any lines containing any of the noise_phrases (case-insensitive).
    Saves the intermediate text to temp_file.

    noise_phrases: list of strings to remove if found in line.
    debug: if True, prints lines that are removed.
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(temp_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            original_line = line.rstrip('\n')
            check_line = original_line.strip().lower()

            skip_line = False
            for phrase in noise_phrases:
                phrase_lower = phrase.lower()
                # Simple substring check
                if phrase_lower in check_line:
                    skip_line = True
                    if debug:
                        print(f"SKIPPING: {original_line} (matched: {phrase})")
                    break

            if not skip_line:
                outfile.write(original_line + "\n")

########################################
# 2) Fix hyphenation
########################################

def fix_hyphens(input_file, temp_file):
    """
    Joins words broken across lines, e.g. 'photo-\nsynthesis' => 'photosynthesis'.
    Merges multiple blank lines, too.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        text = f.read()

    # Regex: looks for pattern: word char + hyphen + newline + letter
    # Then merges into one word
    text = re.sub(r'(\w)-\n(\w)', r'\1\2', text)

    # Also remove extra blank lines by merging consecutive newlines
    text = re.sub(r'\n+', '\n', text).strip()

    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write(text)

########################################
# 3) (Naïve) Multi-Column Handling
########################################

def reorder_multicolumn(input_file, output_file, debug=False):
    """
    A naive approach to handle multi-column text by:
    - Splitting lines if they appear too long or contain column-separating whitespace.
    - Attempting to reorder lines by approximate 'left column' vs. 'right column'.

    WARNING: This approach can fail on complex layouts.
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 3A. Attempt to detect potential columns by line length or patterns
    # For example, if a line is REALLY wide, we can guess it's multiple columns jammed together
    # Or if there's a large chunk of spaces in the middle, we might split.

    reorganized_lines = []
    for line in lines:
        line = line.rstrip('\n')

        # If the line is very long, we can attempt a naive split on big spaces
        # e.g., if there's a region with (say) 3+ consecutive spaces, treat it as column boundary
        # Adjust the threshold as needed (this is naive!)
        columns = re.split(r'\s{3,}', line)

        if len(columns) > 1:
            # We have multiple chunks that might be columns
            # For now, store them in a list
            reorganized_lines.append(columns[0].strip())
            reorganized_lines.append(columns[1].strip())
            if debug:
                print("Splitting line into columns:", columns)
        else:
            # Just one column
            reorganized_lines.append(line)

    # 3B. If you know the exact PDF structure (like left col, right col),
    # you might want to reorder lines so all "left col" lines come first,
    # then "right col" lines. But that requires advanced bounding box info
    # or consistent patterns. The below doesn't do that; it just cleans up lines.

    # We'll write out the reorganized lines as-is
    with open(output_file, 'w', encoding='utf-8') as f:
        for out_line in reorganized_lines:
            if out_line.strip():
                f.write(out_line + "\n")

########################################
# MAIN PIPELINE
########################################

def cleanup_pipeline(raw_input, final_output):
    """
    1) Remove repeated lines -> temp1
    2) Fix hyphens -> temp2
    3) Attempt naive multi-column reorder -> final_output
    """

    temp1 = "temp_step1.txt"
    temp2 = "temp_step2.txt"

    # 1) Remove repeated lines
    noise_phrases = [
        "access for free at",
        "openstax",
        "rice university",
        "creative commons",
        "support@openstax.org",
        # etc. Add more if needed
    ]
    remove_repeated_lines(raw_input, temp1, noise_phrases, debug=False)

    # 2) Fix hyphenation
    fix_hyphens(temp1, temp2)

    # 3) (Optional) naive multi-column reorder
    reorder_multicolumn(temp2, final_output, debug=False)

    # Cleanup temp files if you want
    os.remove(temp1)
    os.remove(temp2)

    print(f"Done! Cleaned text saved to: {final_output}")

if __name__ == "__main__":
    # Example usage:
    if len(sys.argv) < 3:
        print("Usage: python cleanup_text.py <raw_input.txt> <final_output.txt>")
        sys.exit(1)

    raw_input_txt = sys.argv[1]
    final_output_txt = sys.argv[2]

    cleanup_pipeline(raw_input_txt, final_output_txt)

