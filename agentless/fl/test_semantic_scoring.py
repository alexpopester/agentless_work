import ast
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity

# --- 1. The Code Parser (The "Chopper") ---
class CodeParser(ast.NodeVisitor):
    """
    Parses a python file and extracts all functions and methods
    with their line numbers and source code.
    """
    def __init__(self, source_code, file_path):
        self.source_code = source_code
        self.file_path = file_path
        self.functions = []
        self.lines = source_code.splitlines()

    def visit_FunctionDef(self, node):
        # Extract the function source code specifically
        # (This is a simple extraction, sophisticated versions handle decorators)
        start_line = node.lineno - 1
        end_line = node.end_lineno
        func_source = "\n".join(self.lines[start_line:end_line])
        
        self.functions.append({
            "name": node.name,
            "file": self.file_path,
            "start_line": start_line + 1,
            "end_line": end_line,
            "code": func_source
        })
        self.generic_visit(node)

def extract_functions(file_path, source_code):
    try:
        tree = ast.parse(source_code)
        parser = CodeParser(source_code, file_path)
        parser.visit(tree)
        return parser.functions
    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return []

# --- 2. The AI Engine (GraphCodeBERT) ---
class SemanticMatcher:
    def __init__(self):
        print("Loading GraphCodeBERT...")
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
        self.model = AutoModel.from_pretrained("microsoft/graphcodebert-base")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def get_embedding(self, text):
        # Truncate to 512 because BERT cannot handle infinite text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # [CLS] token is usually the first token (index 0) representing the whole sequence
        return outputs.last_hidden_state[:, 0, :].cpu().numpy()

# --- 3. Putting it together (The Workflow) ---
def run_analysis(bug_report_text, file_contents_map):
    matcher = SemanticMatcher()

    # A. Embed the Bug Report
    print("Embedding Bug Report...")
    bug_vec = matcher.get_embedding(bug_report_text)

    all_candidates = []

    # B. Process every file Agentless found
    for file_path, source_code in file_contents_map.items():
        # 1. Chop file into functions
        functions = extract_functions(file_path, source_code)
        
        for func in functions:
            # 2. Embed the function source code
            code_vec = matcher.get_embedding(func['code'])
            
            # 3. Compare (Cosine Similarity)
            score = cosine_similarity(bug_vec, code_vec)[0][0]
            
            func['score'] = float(score)
            all_candidates.append(func)

    # C. Sort by relevance
    all_candidates.sort(key=lambda x: x['score'], reverse=True)

    return all_candidates

# ==========================================
# Example Usage (Simulating your Astropy Data)
# ==========================================
if __name__ == "__main__":
    # 1. The Bug Report (You retrieve this from the dataset)
    bug_report = """
    The fitsrec.py module fails when handling non-standard ASCII characters 
    in the column headers. It raises a UnicodeDecodeError when reading 
    tables created with older FITS software.
    """

    # 2. The Files (You retrieved these from your 'Playground' repo)
    # Imagine this content came from: open("playground/astropy/io/fits/fitsrec.py").read()
    fitsrec_code = """
import numpy as np

def _get_col_attributes(col):
    # This function sets up column attributes
    pass

def _convert_ascii(col):
    # Handles ascii conversion
    # Potential bug here handling decoding?
    return col.decode('ascii')

def read_fits_table(file_ptr):
    # Main logic to read table
    data = _convert_ascii(file_ptr.read())
    return data
    """
    
    # This map mimics what you pull from the repo
    files_map = {
        "astropy/io/fits/fitsrec.py": fitsrec_code
    }

    # 3. Run
    results = run_analysis(bug_report, files_map)

    # 4. View Output
    print(f"\nTop Suspect Functions for Bug:")
    for res in results:
        print(f"[{res['score']:.4f}] {res['name']} (Line {res['start_line']})")