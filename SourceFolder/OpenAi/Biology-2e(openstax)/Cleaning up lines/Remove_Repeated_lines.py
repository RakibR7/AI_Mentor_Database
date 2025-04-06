import re


def remove_repeated_lines(input_file, output_file, noise_phrases, debug=False):
    """
    Removes any lines containing any of the noise_phrases (case-insensitive).
    Saves cleaned text to output_file.

    noise_phrases: list of strings or regex patterns to remove if found in line.
    debug: if True, prints lines that are removed.
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
            open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            # Keep the original line for writing
            original_line = line.rstrip('\n')

            # Prepare a stripped, lowercase version for searching
            check_line = original_line.strip().lower()

            # Flag to indicate if we skip this line
            skip_line = False

            # Check each noise phrase
            for phrase in noise_phrases:
                # Convert phrase to lowercase as well
                phrase_lower = phrase.lower()

                # If phrase is found anywhere in the line, skip
                if phrase_lower in check_line:
                    skip_line = True
                    if debug:
                        print(f"SKIPPING: {original_line} (matched: {phrase})")
                    break

            if not skip_line:
                outfile.write(original_line + "\n")


if __name__ == "__main__":
    input_txt = "Biology2e_extracted.txt"
    cleaned_txt = "biology_textbook.txt"

    noise_phrases = [
        "access for free at",  # partial match
        "openstax",  # partial match
        "rice university",
        "creative commons",
        "for questions regarding this licensing",
        "support@openstax.org"
    ]

    # Turn debug=True temporarily to see what lines get skipped
    remove_repeated_lines(input_txt, cleaned_txt, noise_phrases, debug=False)
    print("Done. Cleaned file saved to:", cleaned_txt)
