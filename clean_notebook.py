import nbformat
import sys

def clean_notebook_widgets(notebook_path):
    try:
        # 노트북 파일 읽기
        ntbk = nbformat.read(notebook_path, as_version=4)

        # 메타데이터에서 'widgets' 키가 있는지 확인하고 제거
        if 'widgets' in ntbk.metadata:
            del ntbk.metadata['widgets']
            print(f"'metadata.widgets' key removed from {notebook_path}")

            # 수정된 내용으로 파일 덮어쓰기
            with open(notebook_path, 'w', encoding='utf-8') as f:
                nbformat.write(ntbk, f)
            print(f"Cleaned notebook saved back to {notebook_path}")
        else:
            print(f"'metadata.widgets' key not found in {notebook_path}. No changes made.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clean_notebook.py <huggingface_basics.ipynb>")
    else:
        notebook_file = sys.argv[1]
        clean_notebook_widgets(notebook_file)