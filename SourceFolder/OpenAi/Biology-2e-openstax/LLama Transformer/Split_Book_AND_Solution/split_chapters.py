import os
import re

INPUT_FILE = "Biology2e_cleaned.txt"
OUTPUT_DIR = "chapters_cleaned"

def split_chapters():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        text = f.read()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    #Match CHAPTER number headers
    pattern = re.compile(r"(CHAPTER\s+\d+.*?)\n(?=CHAPTER\s+\d+|\Z)", re.DOTALL)
    matches = pattern.findall(text)

    for idx, match in enumerate(matches, start=1):
        chapter_num = f"{idx:02d}"
        chapter_file = os.path.join(OUTPUT_DIR, f"chapter_{chapter_num}.txt")

        with open(chapter_file, "w", encoding="utf-8") as out:
            out.write(match.strip())

        print(f"✅ Saved {chapter_file}")

    print(f"\n📚 Split {len(matches)} chapters into {OUTPUT_DIR}/")

if __name__ == "__main__":
    split_chapters()
