import io
import re
from tqdm import tqdm
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBox, LTTextLine, LAParams


def extract_text_from_page(page):
    """
    Extract all text from a single PDF page object.
    """
    text = ""
    for element in page:
        # LTTextBox and LTTextLine are common text containers
        if isinstance(element, LTTextBox) or isinstance(element, LTTextLine):
            text += element.get_text()
    return text


def clean_text(text):
    """
    A simple cleaning step:
    - Merge multiple newlines into one
    - Strip leading/trailing spaces
    """
    # Replace consecutive newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    # Strip leading and trailing whitespace
    text = text.strip()
    return text


def pdf_to_text(pdf_path, output_txt_path):
    """
    Extract text from each page of a PDF with a progress bar.
    Then clean and write to a .txt file.
    """
    # Layout parameters (you can tweak char_margin, line_margin, etc. if needed)
    laparams = LAParams()

    # Convert pages generator to a list so we know total page count
    pages = list(extract_pages(pdf_path, laparams=laparams))
    total_pages = len(pages)

    print(f"Total pages found: {total_pages}")

    all_text = []

    # Iterate through each page with a progress bar
    for page in tqdm(pages, desc="Extracting text", unit="page"):
        page_text = extract_text_from_page(page)
        all_text.append(page_text)

    # Join all pages
    combined_text = "\n\n".join(all_text)

    # Simple cleaning
    cleaned_text = clean_text(combined_text)

    # Write the final text to a file
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(f"\nExtraction complete. Output saved to '{output_txt_path}'")


if __name__ == "__main__":
    input_pdf = "Biology2e.pdf"  # Change to your PDF name/path
    output_txt = "Biology2e_extracted.txt"

    pdf_to_text(input_pdf, output_txt)
