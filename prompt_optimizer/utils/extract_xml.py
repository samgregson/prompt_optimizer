import re


def extract_text_from_xml(xml_string, tag_name):
    pattern = f"<{tag_name}>(.*?)</{tag_name}>"
    matches = re.findall(pattern, xml_string, re.DOTALL)
    cleaned_matches = [match.strip() for match in matches]
    return cleaned_matches
