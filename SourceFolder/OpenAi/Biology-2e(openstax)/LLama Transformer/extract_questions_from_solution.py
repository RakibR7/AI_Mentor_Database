prompt = f"""
Based on the following biology chapter content, generate a clear standalone question that this answer could belong to.

--- Chapter Content ---
{chapter_text[:1000]}  # Only a chunk for context

--- Answer ---
{answer}

Question:"""
