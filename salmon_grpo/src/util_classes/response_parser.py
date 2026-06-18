import re
import ast

class ResponseParser:
    """
    Utility class for parsing model responses in the XML-like format:
    <think>...</think><reasons>['A', 'B']</reasons><answer>spoof/bonafide</answer>
    """

    @staticmethod
    def parse_xml_tag(text, tag):
        """Extract content from <tag>content</tag>"""
        if not isinstance(text, str):
            return None
        pattern = f"<{tag}>(.*?)</{tag}>"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None

    @staticmethod
    def parse_reasons(text):
        """Extract and parse the list of reasons from <reasons>['A', 'B']</reasons>"""
        content = ResponseParser.parse_xml_tag(text, "reasons")
        if not content:
            return []
        try:
            val = ast.literal_eval(content)
            if isinstance(val, (list, tuple, set)):
                return [str(item) for item in val]
            elif val is Ellipsis:
                return []
            else:
                return [str(val)]
        except:
            # Fallback for malformed lists
            content = content.strip("[]")
            items = [item.strip().strip("'\"") for item in content.split(',') if item.strip()]
            return items

    @staticmethod
    def parse_prediction(pred_text):
        """
        Parse model output to binary prediction (spoof/bonafide).
        Prioritizes <answer> tag, falls back to 'Final Answer:' markers.
        """
        if not isinstance(pred_text, str):
            return None
        pred_text = pred_text.lower().strip()
        
        # 1. Try strictly parsing the <answer> tag
        answer_content = ResponseParser.parse_xml_tag(pred_text, "answer")
        if answer_content:
            target_text = answer_content.lower()
        else:
            # 2. Fallback: Look for "Final Answer:" marker
            marker_match = re.search(r'(?:final answer|conclusion|verdict)\s*[:=]\s*(.*)', pred_text, re.DOTALL)
            target_text = marker_match.group(1) if marker_match else pred_text
            
        # Check for indicators
        has_bonafide = 'bonafide' in target_text or 'genuine' in target_text
        has_spoof = 'spoof' in target_text or 'fake' in target_text
        
        if has_bonafide and not has_spoof:
            return 'bonafide'
        elif has_spoof and not has_bonafide:
            return 'spoof'
        elif has_bonafide and has_spoof:
            # Both present. Heuristic: which one appears LAST?
            idx_bonafide = max(target_text.rfind('bonafide'), target_text.rfind('genuine'))
            idx_spoof = max(target_text.rfind('spoof'), target_text.rfind('fake'))
            return 'bonafide' if idx_bonafide > idx_spoof else 'spoof'
        else:
            return None



