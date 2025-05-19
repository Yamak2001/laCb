import re
import os

def fix_method_indentation(file_path):
    """Fix common indentation errors in method definitions"""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Fix methods that have extra indentation before 'def'
    pattern = r'(\n\s+)\n(\s+def\s+\w+\()'
    
    def fix_indent(match):
        indent = match.group(1)
        method_def = match.group(2)
        # Remove one level of indentation (4 spaces)
        if len(indent) >= 4:
            fixed_indent = indent[:-4]
        else:
            fixed_indent = '\n    '
        return fixed_indent + method_def
    
    # Apply the fix
    fixed_content = re.sub(pattern, fix_indent, content)
    
    # Fix the pattern where there's extra blank space before def
    pattern2 = r'(\n\s*)\n(\s+def\s+\w+\()'
    fixed_content = re.sub(pattern2, r'\1\2', fixed_content)
    
    if fixed_content != content:
        with open(file_path, 'w') as f:
            f.write(fixed_content)
        print(f"Fixed {file_path}")
        return True
    return False

# Fix all model files
model_paths = [
    'models/embedding/pyannote_model.py',
    'models/embedding/speechbrain_model.py',
    'models/embedding/titanet_model.py',
    'models/separation/demucs_model.py',
    'models/separation/dprnn_model.py',
    'models/separation/ecapa_model.py',
    'models/separation/hdemucs_model.py'
]

for path in model_paths:
    if os.path.exists(path):
        fix_method_indentation(path)