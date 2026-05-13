"""
批量上传 5 份东北大学政策 PDF 到 RAG 系统
"""
import os
import sys
import shutil

# 清理旧的 chroma_db 和 pdfs 目录
CHROMA_DIR = 'D:/LLM-RAG-Internship/chroma_db'
PDFS_DIR = 'D:/LLM-RAG-Internship/data/pdfs'

for d in [CHROMA_DIR, PDFS_DIR]:
    if os.path.exists(d):
        shutil.rmtree(d)
        print(f"已清理: {d}")
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(PDFS_DIR, exist_ok=True)

# 添加项目根目录到 path
sys.path.insert(0, 'D:/LLM-RAG-Internship')
from rag_core.document_manager import DocumentManager

DATA_DIR = 'D:/LLM-RAG-Internship/data'

# 用 os.listdir 动态匹配文件名，避免引号等特殊字符编码不一致
all_files = os.listdir(DATA_DIR)
policy_pdfs = [
    f for f in all_files
    if f.endswith('.pdf') and f != 'data.pdf'
]

print(f"找到 {len(policy_pdfs)} 个政策 PDF 文件:")
for f in policy_pdfs:
    print(f"  [{repr(f)}]")

doc_manager = DocumentManager()

for pdf_name in policy_pdfs:
    pdf_path = os.path.join(DATA_DIR, pdf_name)
    with open(pdf_path, 'rb') as f:
        file_bytes = f.read()
    try:
        doc_id = doc_manager.upload_pdf(file_bytes, pdf_name)
        print(f"[OK] 上传成功: {pdf_name} -> doc_id={doc_id}")
    except Exception as e:
        print(f"[FAIL] 上传失败: {pdf_name}: {e}")

# 验证
print("\n=== 上传结果验证 ===")
docs = doc_manager.list_documents()
print(f"共 {len(docs)} 份文档:")
for d in docs:
    print(f"  - {d['original_filename']} ({d['chunk_count']} chunks, status={d['status']})")
