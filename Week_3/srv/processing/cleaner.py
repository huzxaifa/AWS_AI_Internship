import re

def clean_text(text):
    
    if not text:
        return ""

    if not text:
        return ""

    # 1. Fix broken hyphenated words (e.g., "infor-\nmation" -> "information")
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)

    # 2. Remove non-printable characters (but keep newlines)
    text = re.sub(r'[^\x20-\x7E\n\t]+', '', text)

    # 3. Collapse multiple spaces to single space, but preserve newlines for structure
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Normalize whitespace within the line
        line = re.sub(r'\s+', ' ', line).strip()
        
        # Filter out lines that are just single characters or noise (unless it's a currency/digit)
        # e.g., removal of "o", "-", page numbers like "1" if standalone
        if len(line) < 2 and not line.isdigit() and line not in ['$','€','£']:
            continue
            
        if line:
            cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    return text
