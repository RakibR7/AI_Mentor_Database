import sys
from pdfminer.high_level import extract_text


def pdf_to_text(input_pdf, output_txt):
    text = extract_text(input_pdf)
    cleaned = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(cleaned)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python extract_pdfToText_solutions.py input.pdf output.txt")
        sys.exit(1)

    pdf_path = sys.argv[1]
    txt_path = sys.argv[2]

    pdf_to_text(pdf_path, txt_path)
    print(f"Extraction complete. Output saved to {txt_path}")
