import re
import dateparser

class FieldExtractor:
    def __init__(self):
        print("Initializing Regex Field Extractor...")

    def extract_fields(self, text, category):
        """
        Extracts fields based on the document category.
        """
        data = {}
        
        # 1. Common Fields (Apply to all)
        data['emails'] = self._extract_emails(text)
        data['phones'] = self._extract_phones(text)
        
        # 2. Category-Specific Fields
        if category == "Invoice":
            data['invoice_number'] = self._extract_invoice_number(text)
            data['total_amount'] = self._extract_money(text)
            data['date'] = self._extract_date(text)
            
        elif category == "Resume":
            # For resumes, we might want to find "Skills" section or "Education"
            # simple keyword check for now
            pass
            
        return data

    def _extract_emails(self, text):
        # Standard email regex
        pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        return list(set(re.findall(pattern, text)))

    def _extract_phones(self, text):
        # North American phone format (approximate)
        pattern = r'(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}'
        return list(set(re.findall(pattern, text)))

    def _extract_invoice_number(self, text):
        # Looks for "Invoice # 12345" or "Inv: ABC-99"
        pattern = r'(?i)(?:invoice|inv|bill)[\s#.:]+([a-zA-Z0-9-]+)'
        match = re.search(pattern, text)
        return match.group(1) if match else None

    def _extract_money(self, text):
        # Looks for "Total: $500.00" or "$500.00" near keywords
        # 1. Look for explicit "Total" label
        pattern_total = r'(?i)(?:total|amount|due|balance)[\s:|$]+([\d,]+\.\d{2})'
        match = re.search(pattern_total, text)
        if match:
            return match.group(1)
        
        # 2. Fallback: Find biggest dollar amount (heuristic)
        pattern_money = r'\$([\d,]+\.\d{2})'
        matches = re.findall(pattern_money, text)
        if matches:
            # Convert to float to find max, then return strict string
            try:
                values = [float(m.replace(',', '')) for m in matches]
                max_val = max(values)
                return f"{max_val:.2f}"
            except:
                return None
        return None

    def _extract_date(self, text):
        # Finds dates and parses them to YYYY-MM-DD
        # Look for "Date: Oct 20, 2023"
        pattern = r'(?i)(?:date|dated)[\s:]+([a-zA-Z0-9, \-/]+)'
        match = re.search(pattern, text)
        if match:
            date_str = match.group(1).strip()
            # Use dateparser to normalize
            dt = dateparser.parse(date_str)
            if dt:
                return dt.strftime('%Y-%m-%d')
        
        # Fallback: Look for ISO dates YYYY-MM-DD
        iso_pattern = r'\d{4}-\d{2}-\d{2}'
        match = re.search(iso_pattern, text)
        if match:
            return match.group(0)
            
        return None
