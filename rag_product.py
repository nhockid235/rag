#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RAG Product - Professional Vietnamese RAG System
================================================

Complete RAG system with:
- Section-aware semantic chunking
- Hybrid retrieval (Dense + BM25 + RRF)
- Cross-encoder reranking
- Multi-language support
- Professional PDF processing
- Gradio web interface

Author: RAG Product Team
Version: 1.0.0
"""

import os
import re
import json
import time
import argparse
import pickle
import unicodedata
import textwrap
import hashlib
import shutil
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

import numpy as np
from tqdm import tqdm

# Embedding & search
from sentence_transformers import SentenceTransformer
import faiss

# BM25
from rank_bm25 import BM25Okapi

# PDF processing (truy xuất theo TRANG)
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer

# Ollama
import requests

# Gradio
import gradio as gr


# ======================
# 0) CẤU HÌNH MẶC ĐỊNH
# ======================

class CFG:
    # Đường dẫn mặc định
    DATA_DIR = Path("data/imported")
    INDEX_DIR = Path("indexes/professional")

    # Mô hình embedding & reranker
    EMBEDDING_MODEL = "BAAI/bge-m3"
    RERANK_MODEL = "BAAI/bge-reranker-v2-m3"

    # LLM (Ollama)
    OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gpt-oss:20b")
    OLLAMA_TEMPERATURE = float(os.environ.get("OLLAMA_TEMPERATURE", "0.2"))
    OLLAMA_NUM_CTX = int(os.environ.get("OLLAMA_NUM_CTX", "8192"))

    # Chunking
    CHUNK_MAX_CHARS = 1800
    CHUNK_OVERLAP_CHARS = 200

    # Truy hồi
    TOPK_DENSE = 100
    TOPK_BM25 = 100
    RRF_K = 60
    TOP_N_AFTER_MMR = 12
    MMR_LAMBDA = 0.8

    # Rerank
    RERANK_TOPK = 8
    RERANK_THRESHOLD = 0.05  # bỏ context quá yếu

    # Token hoá
    USE_UNDERTHESEA = True  # nếu có underthesea sẽ dùng, không có thì fallback


# ======================
# 1) Professional PDF Processing
# ======================

def _is_header_line(line: str) -> bool:
    """Heuristic nhận diện heading/section (ngắn, in hoa, hoặc có 'chương/phần/mục/điều/khoản')."""
    if not line:
        return False
    low = line.lower()
    if len(line) < 120 and line.isupper():
        return True
    if re.match(r"^(chương|phần|mục)\s+[ivxlcdm0-9\.\-]+", low):
        return True
    if re.match(r"^điều\s+\d+(\.|:)?", low):
        return True
    if re.match(r"^khoản\s+\d+(\.|:)?", low):
        return True
    # Có thể bổ sung thêm pattern ngành/lĩnh vực
    return False

def extract_pdf_with_metadata(pdf_path: str) -> Dict[str, Any]:
    """
    Parse PDF theo TRANG bằng pdfminer.extract_pages để lấy page number CHUẨN.
    Trả:
        - structured_content: list dict {type: header/paragraph, content, section, page}
        - metadata: file_name, size, time...
    """
    structured_content = []
    try:
        for page_num, layout in enumerate(extract_pages(pdf_path), start=1):
            for element in layout:
                if isinstance(element, LTTextContainer):
                    # Duyệt theo dòng
                    for line_obj in element:
                        line = line_obj.get_text().strip()
                        if not line:
                            continue
                        is_header = _is_header_line(line)
                        structured_content.append({
                            "type": "header" if is_header else "paragraph",
                            "content": normalize_vi(line),
                            "section": line if is_header else "",
                            "page": page_num,
                        })
        raw_text = "\n".join([x["content"] for x in structured_content])
        meta = {
            "file_path": pdf_path,
            "file_name": Path(pdf_path).name,
            "file_size": os.path.getsize(pdf_path),
            "extraction_time": datetime.now().isoformat(),
            "total_sections": len([x for x in structured_content if x["type"] == "header"]),
        }
        return {"raw_text": raw_text, "structured_content": structured_content, "metadata": meta}
    except Exception as e:
        return {
            "raw_text": "",
            "structured_content": [],
            "metadata": {"file_path": pdf_path, "file_name": Path(pdf_path).name, "error": str(e)},
        }


# ======================
# 2) Section-Aware Semantic Chunking
# ======================

def normalize_vi(text: str) -> str:
    """Normalize Vietnamese text"""
    if text is None: return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def strip_accents_vi(text: str) -> str:
    """Strip Vietnamese accents for better matching"""
    if text is None: return ""
    text = unicodedata.normalize('NFD', text)
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn')
    return text.replace('đ','d').replace('Đ','D')

# Tokenizer tiếng Việt (ưu tiên underthesea; fallback regex)
_underthesea_ok = False
_tokens_info = "regex"
try:
    from underthesea import word_tokenize as _ut_word_tokenize
    _underthesea_ok = True
    _tokens_info = "underthesea"
except Exception:
    _underthesea_ok = False
    _tokens_info = "regex"

def tokenize_vi(text: str) -> List[str]:
    """Token hoá tiếng Việt cho BM25:
       - Nếu có underthesea: dùng word_tokenize trên bản không dấu.
       - Nếu không: fallback regex a-z0-9 trên bản không dấu.
    """
    nd = strip_accents_vi(text).lower()
    if _underthesea_ok:
        try:
            toks = _ut_word_tokenize(nd)
            # underthesea trả chuỗi/tokens; ép về list[str]
            if isinstance(toks, str):
                toks = toks.split()
            return [t for t in toks if t and not t.isspace()]
        except Exception:
            pass
    # Fallback
    return re.findall(r"[a-z0-9]+", nd)

def section_aware_chunking(structured: List[Dict[str, Any]],
                           max_chars: int = CFG.CHUNK_MAX_CHARS,
                           overlap_chars: int = CFG.CHUNK_OVERLAP_CHARS) -> List[Dict[str, Any]]:
    """
    Gộp các 'paragraph' thành chunk có độ dài mục tiêu theo KÝ TỰ (ổn định hơn 'đếm item').
    - Khi gặp header: cập nhật current_section.
    - Mỗi chunk: prefix heading để tăng tín hiệu chủ đề.
    - Overlap tính theo ký tự cuối của chunk trước.
    """
    chunks, cur, cur_len, cur_section = [], [], 0, ""
    for item in structured:
        t = normalize_vi(item["content"])
        if item["type"] == "header":
            # Cập nhật section hiện tại
            cur_section = t
            continue
        if not t:
            continue
        l = len(t)

        # Nếu thêm vào sẽ vượt max_chars -> finalize chunk
        if cur_len + l + 1 > max_chars and cur:
            text = " ".join(x["content"] for x in cur)
            if cur_section:
                text = f"{cur_section}\n{text}"
            chunks.append({
                "text": text,
                "meta": {
                    "section": cur_section or "N/A",
                    "page": cur[0]["page"],
                    "items_count": len(cur),
                },
            })
            # Overlap theo ký tự
            keep, acc = [], 0
            for x in reversed(cur):
                if acc >= overlap_chars:
                    break
                keep.append(x)
                acc += len(x["content"])
            cur = list(reversed(keep))
            cur_len = sum(len(x["content"]) for x in cur)

        cur.append({"content": t, "page": item["page"]})
        cur_len += l + 1

    if cur:
        text = " ".join(x["content"] for x in cur)
        if cur_section:
            text = f"{cur_section}\n{text}"
        chunks.append({
            "text": text,
            "meta": {"section": cur_section or "N/A", "page": cur[0]["page"], "items_count": len(cur)},
        })
    return chunks


# ======================
# 3) Professional Embedding & Indexing
# ======================

class ProfessionalDenseIndexer:
    """Professional dense indexer with BGE-M3 support"""
    
    def __init__(self, model_name: str = CFG.EMBEDDING_MODEL):
        self.model_name = model_name
        print(f"Loading embedding model: {model_name}")
        self.emb = SentenceTransformer(model_name)
        self.index = None
        self.doc_matrix = None
        self.docs: List[Dict[str, Any]] = []

    def encode_passages(self, texts: List[str]) -> np.ndarray:
        """Encode passages with proper prefixes"""
        # BGE-M3 uses different prefixes for different tasks
        prefixed_texts = ["passage: " + t for t in texts]
        vecs = self.emb.encode(prefixed_texts, normalize_embeddings=True, show_progress_bar=True)
        return np.asarray(vecs, dtype="float32")

    def encode_query(self, text: str) -> np.ndarray:
        """Encode query with proper prefix"""
        prefixed_text = "query: " + text
        vec = self.emb.encode(prefixed_text, normalize_embeddings=True)
        return np.asarray(vec, dtype="float32")

    def fit(self, chunks: List[Dict[str, Any]]):
        """Fit the indexer with chunks"""
        self.docs = []
        texts = []
        
        for ch in tqdm(chunks, desc="Processing chunks"):
            # Normalize text
            normalized_text = normalize_vi(ch["text"])
            
            # Create document with metadata
            doc = {
                "text": normalized_text,
                "text_nodiac": strip_accents_vi(normalized_text),
                "meta": ch.get("meta", {}),
                "chunk_id": len(self.docs)
            }
            self.docs.append(doc)
            texts.append(normalized_text)
        
        # Generate embeddings
        print("Generating embeddings...")
        self.doc_matrix = self.encode_passages(texts)
        
        # Create FAISS index
        dim = self.doc_matrix.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # Inner product for normalized embeddings
        self.index.add(self.doc_matrix)
        
        print(f"Index created with {len(self.docs)} documents, dimension {dim}")

    def save(self, index_dir: str):
        """Save index and metadata"""
        os.makedirs(index_dir, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(Path(index_dir, "index.faiss")))
        np.save(str(Path(index_dir, "embeddings.npy")), self.doc_matrix)
        
        # Save documents
        with open(Path(index_dir, "docs.jsonl"), "w", encoding="utf-8") as f:
            for d in self.docs:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")
        
        # Save config
        cfg = {
            "model_name": self.model_name,
            "created_at": time.time(),
            "total_docs": len(self.docs),
            "dimension": self.doc_matrix.shape[1]
        }
        Path(index_dir, "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        
        print(f"Index saved to {index_dir}")

    @staticmethod
    def load(index_dir: str) -> "ProfessionalDenseIndexer":
        """Load existing index"""
        index_dir = Path(index_dir)
        
        # Load config
        cfg = json.loads(index_dir.joinpath("config.json").read_text(encoding="utf-8"))
        obj = ProfessionalDenseIndexer(model_name=cfg.get("model_name", "BAAI/bge-m3"))
        
        # Load FAISS index
        obj.index = faiss.read_index(str(index_dir.joinpath("index.faiss")))
        obj.doc_matrix = np.load(str(index_dir.joinpath("embeddings.npy")))
        
        # Load documents
        obj.docs = []
        with open(index_dir.joinpath("docs.jsonl"), "r", encoding="utf-8") as f:
            for line in f:
                obj.docs.append(json.loads(line))
        
        print(f"Loaded index with {len(obj.docs)} documents")
        return obj


# ======================
# 4) Professional BM25
# ======================

def build_professional_bm25(index_dir: str, docs: List[Dict[str, Any]]):
    """Build professional BM25 index with Vietnamese support"""
    print("Building BM25 index...")
    
    # Create directory if not exists
    os.makedirs(index_dir, exist_ok=True)
    
    # Tokenize documents
    corpus_tokens = []
    for doc in tqdm(docs, desc="Tokenizing for BM25"):
        # Use both original and accent-stripped text
        original_tokens = tokenize_vi(doc["text"])
        stripped_tokens = tokenize_vi(doc["text_nodiac"])
        combined_tokens = original_tokens + stripped_tokens
        corpus_tokens.append(combined_tokens)
    
    # Create BM25 index
    bm25 = BM25Okapi(corpus_tokens)
    
    # Save BM25 index
    with open(Path(index_dir, "bm25.pkl"), "wb") as f:
        pickle.dump({
            "corpus_tokens": corpus_tokens,
            "bm25": bm25,
            "total_docs": len(docs)
        }, f)
    
    print(f"BM25 index built with {len(docs)} documents")

def load_professional_bm25(index_dir: str) -> Tuple[BM25Okapi, List[List[str]]]:
    """Load professional BM25 index"""
    p = Path(index_dir, "bm25.pkl")
    if not p.exists():
        raise FileNotFoundError("BM25 index not found. Run build first.")
    
    with open(p, "rb") as f:
        obj = pickle.load(f)
    
    return obj["bm25"], obj["corpus_tokens"]


# ======================
# 5) Hybrid Retrieval with RRF
# ======================

@dataclass
class ProfessionalSearchResult:
    text: str
    meta: Dict[str, Any]
    dense_score: float
    bm25_score: float
    rrf_score: float
    final_score: float

def reciprocal_rank_fusion(dense_scores: Dict[int, float], 
                          bm25_scores: Dict[int, float], 
                          k: int = 60) -> Dict[int, float]:
    """Reciprocal Rank Fusion for combining rankings"""
    all_docs = set(dense_scores.keys()) | set(bm25_scores.keys())
    rrf_scores = {}
    
    for doc_id in all_docs:
        rrf_score = 0.0
        
        # Dense ranking contribution
        if doc_id in dense_scores:
            dense_rank = sorted(dense_scores.values(), reverse=True).index(dense_scores[doc_id]) + 1
            rrf_score += 1.0 / (k + dense_rank)
        
        # BM25 ranking contribution
        if doc_id in bm25_scores:
            bm25_rank = sorted(bm25_scores.values(), reverse=True).index(bm25_scores[doc_id]) + 1
            rrf_score += 1.0 / (k + bm25_rank)
        
        rrf_scores[doc_id] = rrf_score
    
    return rrf_scores

def mmr_select(query_vec: np.ndarray, cand_ids: List[int], doc_matrix: np.ndarray,
               lambda_mult: float = 0.8, top_n: int = 12) -> List[int]:
    """MMR: chọn top_n kết quả đa dạng (maximize rel - (1-lambda)*div)."""
    selected = []
    cand = cand_ids[:]
    while len(selected) < min(top_n, len(cand)):
        best, best_score = None, -1e9
        for i in cand:
            rel = float(doc_matrix[i] @ query_vec)
            div = 0.0 if not selected else max(float(doc_matrix[i] @ doc_matrix[j]) for j in selected)
            score = lambda_mult * rel - (1 - lambda_mult) * div
            if score > best_score:
                best, best_score = i, score
        selected.append(best)
        cand.remove(best)
    return selected

def professional_hybrid_search(
    dense_indexer: ProfessionalDenseIndexer,
    bm25: Optional[BM25Okapi],
    query: str,
    topk_dense: int = CFG.TOPK_DENSE,
    topk_bm25: int = CFG.TOPK_BM25,
    top_n: int = CFG.TOP_N_AFTER_MMR,
    rrf_k: int = CFG.RRF_K
) -> List[ProfessionalSearchResult]:
    """Professional hybrid search with RRF + MMR"""
    
    # Normalize query
    q_norm = normalize_vi(query)
    
    # Dense search
    q_vec = dense_indexer.encode_query(q_norm).astype("float32")
    D, I = dense_indexer.index.search(q_vec.reshape(1, -1), topk_dense)
    dense_scores = {int(i): float(s) for i, s in zip(I[0].tolist(), D[0].tolist())}
    
    # BM25 search
    bm25_scores = {}
    if bm25 is not None:
        q_tokens = tokenize_vi(q_norm)
        scores = bm25.get_scores(q_tokens)
        top_bm25_indices = np.argsort(scores)[::-1][:topk_bm25].tolist()
        bm25_scores = {int(i): float(scores[int(i)]) for i in top_bm25_indices}
    
    # RRF fusion
    rrf_scores = reciprocal_rank_fusion(dense_scores, bm25_scores, k=rrf_k)
    
    # Chọn pre_top rồi MMR
    pre_top_ids = [doc_id for doc_id, _ in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:max(50, top_n * 5)]]
    mmr_ids = mmr_select(q_vec, pre_top_ids, dense_indexer.doc_matrix, lambda_mult=0.8, top_n=top_n)
    
    # Create results
    results = []
    for i in mmr_ids:
        if i < len(dense_indexer.docs):
            doc = dense_indexer.docs[i]
            results.append(ProfessionalSearchResult(
                text=doc["text"],
                meta=doc["meta"],
                dense_score=dense_scores.get(i, 0.0),
                bm25_score=bm25_scores.get(i, 0.0),
                rrf_score=rrf_scores.get(i, 0.0),
                final_score=rrf_scores.get(i, 0.0)
            ))
    
    return results


# ======================
# 6) Cross-Encoder Reranking
# ======================

class ProfessionalReranker:
    """Professional reranker with cross-encoder"""
    
    def __init__(self, model_name: str = CFG.RERANK_MODEL):
        self.model_name = model_name
        print(f"Loading reranker: {model_name}")
        from sentence_transformers import CrossEncoder
        self.reranker = CrossEncoder(model_name)
    
    def rerank(self, query: str, results: List[ProfessionalSearchResult], top_k: int = CFG.RERANK_TOPK) -> List[ProfessionalSearchResult]:
        """Rerank results using cross-encoder"""
        if not results:
            return results
        
        # Prepare pairs for reranking
        pairs = [(query, result.text) for result in results]
        
        # Get rerank scores
        rerank_scores = self.reranker.predict(pairs)
        
        # Update results with rerank scores
        for i, (result, score) in enumerate(zip(results, rerank_scores)):
            result.final_score = float(score)
        
        # Sort by rerank score and return top_k
        reranked_results = sorted(results, key=lambda x: x.final_score, reverse=True)
        return reranked_results[:top_k]


# ======================
# 7) Professional Answer Generation
# ======================

PROFESSIONAL_PROMPT_TEMPLATE = """Bạn là trợ lý AI chuyên nghiệp, trả lời bằng tiếng Việt dựa trên ngữ cảnh được cung cấp.

NGỮ CẢNH:
{context}

CÂU HỎI:
{question}

YÊU CẦU:
- Trả lời chính xác, ngắn gọn, dựa hoàn toàn vào ngữ cảnh
- Nếu không đủ thông tin, hãy trả lời: "Tôi không tìm thấy thông tin liên quan trong tài liệu"
- Luôn trích nguồn cụ thể (trang, phần, tài liệu)
- Sử dụng ngôn ngữ tự nhiên, dễ hiểu
- Nếu có số liệu/quy định, nêu rõ và đính kèm nguồn

TRÍCH NGUỒN:
"""

def build_professional_context(results: List[ProfessionalSearchResult]) -> str:
    """Build professional context with metadata"""
    context_parts = []
    
    for i, result in enumerate(results, 1):
        # Extract metadata
        section = result.meta.get('section', 'Không xác định')
        page = result.meta.get('page', '?')
        
        # Format text with metadata
        text_snippet = textwrap.shorten(result.text, width=300, placeholder="...")
        context_parts.append(f"[{i}] {text_snippet}\n   📄 Nguồn: {section} (Trang {page})")
    
    return "\n\n".join(context_parts)

def call_ollama_professional(ollama_url: str, model: str, system_msg: str, user_prompt: str,
                           temperature: float = CFG.OLLAMA_TEMPERATURE, num_ctx: int = CFG.OLLAMA_NUM_CTX) -> str:
    """Professional Ollama call with better error handling"""
    try:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False,
            "options": {
                "temperature": float(temperature),
                "num_ctx": int(num_ctx),
                "top_p": 0.9,
                "repeat_penalty": 1.1
            }
        }
        
        response = requests.post(
            f"{ollama_url.rstrip('/')}/api/chat", 
            json=payload, 
            timeout=120
        )
        response.raise_for_status()
        
        data = response.json()
        
        # Extract content from response
        if isinstance(data, dict):
            if "message" in data and isinstance(data["message"], dict):
                return data["message"].get("content", "").strip()
            if "content" in data:
                return str(data["content"]).strip()
            if "response" in data:
                return str(data["response"]).strip()
        
        return str(data).strip()
        
    except requests.exceptions.Timeout:
        return "[LỖI] Ollama timeout - vui lòng thử lại"
    except requests.exceptions.ConnectionError:
        return "[LỖI] Không thể kết nối đến Ollama"
    except Exception as e:
        return f"[LỖI] Lỗi khi gọi Ollama: {str(e)}"

def generate_professional_answer(results: List[ProfessionalSearchResult], 
                                question: str,
                                ollama_url: str = CFG.OLLAMA_URL,
                                ollama_model: str = CFG.OLLAMA_MODEL,
                                temperature: float = CFG.OLLAMA_TEMPERATURE,
                                num_ctx: int = CFG.OLLAMA_NUM_CTX) -> str:
    """Generate professional answer with context"""
    
    # Build context
    context = build_professional_context(results)
    
    # Create prompt
    prompt = PROFESSIONAL_PROMPT_TEMPLATE.format(context=context, question=question)
    
    # System message
    system_msg = "Bạn là trợ lý AI chuyên nghiệp, trả lời chính xác dựa trên ngữ cảnh được cung cấp."
    
    # Call Ollama
    return call_ollama_professional(ollama_url, ollama_model, system_msg, prompt, temperature, num_ctx)


# ======================
# 8) Professional RAG Product
# ======================

class ProfessionalRAGProduct:
    """Professional RAG Product with complete functionality"""
    
    def __init__(self, index_dir: str = str(CFG.INDEX_DIR)):
        self.index_dir = index_dir
        self.dense_indexer = None
        self.bm25 = None
        self.reranker = None
        self.load_system()
    
    def load_system(self):
        """Load RAG system"""
        try:
            if Path(self.index_dir).exists():
                self.dense_indexer = ProfessionalDenseIndexer.load(self.index_dir)
                try:
                    self.bm25, _ = load_professional_bm25(self.index_dir)
                except:
                    self.bm25 = None
                try:
                    self.reranker = ProfessionalReranker()
                except:
                    self.reranker = None
                print("✅ RAG system loaded successfully")
            else:
                print("⚠️ No index found, system not loaded")
        except Exception as e:
            print(f"❌ Error loading system: {e}")
    
    def import_pdf(self, files, mode):
        """Import PDF files and rebuild index"""
        if not files:
            return "❌ No files selected", "System not loaded"
        
        try:
            # Create data directory
            data_dir = CFG.DATA_DIR
            data_dir.mkdir(exist_ok=True)
            
            # Clear if replace mode
            if mode == "replace":
                for file in data_dir.glob("*.pdf"):
                    file.unlink()
            
            # Copy files
            imported_files = []
            for file_path in files:
                if os.path.exists(file_path):
                    filename = Path(file_path).name
                    target_path = data_dir / filename
                    shutil.copy2(file_path, target_path)
                    imported_files.append(filename)
            
            if not imported_files:
                return "❌ No valid files imported", "System not loaded"
            
            # Rebuild index
            print(f"🔄 Rebuilding index with {len(imported_files)} files...")
            
            # Process files
            all_chunks = []
            for pdf_file in data_dir.glob("*.pdf"):
                pdf_data = extract_pdf_with_metadata(str(pdf_file))
                if pdf_data['raw_text']:
                    chunks = section_aware_chunking(pdf_data['structured_content'])
                    for chunk in chunks:
                        chunk['meta']['file_name'] = pdf_file.name
                    all_chunks.extend(chunks)
            
            if not all_chunks:
                return "❌ No valid content extracted", "System not loaded"
            
            # Build new index
            self.dense_indexer = ProfessionalDenseIndexer()
            self.dense_indexer.fit(all_chunks)
            self.dense_indexer.save(self.index_dir)
            
            # Build BM25
            build_professional_bm25(self.index_dir, self.dense_indexer.docs)
            
            # Load BM25
            try:
                self.bm25, _ = load_professional_bm25(self.index_dir)
            except:
                self.bm25 = None
            
            # Load reranker
            try:
                self.reranker = ProfessionalReranker()
            except:
                self.reranker = None
            
            success_msg = f"✅ Successfully imported {len(imported_files)} files\n📊 Created {len(all_chunks)} chunks\n🧠 Index rebuilt"
            status_msg = f"System Status:\n- Files: {len(imported_files)}\n- Chunks: {len(all_chunks)}\n- Model: BGE-M3\n- Status: Ready"
            
            return success_msg, status_msg
            
        except Exception as e:
            return f"❌ Error importing files: {str(e)}", "System not loaded"
    
    def chat_with_rag(self, message, history):
        """Chat with RAG system"""
        if not self.dense_indexer:
            error_msg = "❌ RAG system not loaded. Please import PDF files first."
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return "", history
        
        try:
            # Search
            results = professional_hybrid_search(
                self.dense_indexer, self.bm25, message,
                topk_dense=CFG.TOPK_DENSE, topk_bm25=CFG.TOPK_BM25, top_n=CFG.TOP_N_AFTER_MMR
            )
            
            # Rerank if available
            if self.reranker and results:
                results = self.reranker.rerank(message, results, top_k=CFG.RERANK_TOPK)
            
            # Generate answer
            answer = generate_professional_answer(results, message)
            
            # Add sources
            if results:
                answer += "\n\n**📚 Nguồn tham khảo:**\n"
                for i, result in enumerate(results[:3], 1):
                    section = result.meta.get('section', 'N/A')
                    page = result.meta.get('page', 'N/A')
                    answer += f"{i}. {section} (Trang {page})\n"
            
            # Update history
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": answer})
            
            return "", history
            
        except Exception as e:
            error_msg = f"❌ Error processing query: {str(e)}"
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": error_msg})
            return "", history
    
    def get_status(self):
        """Get system status"""
        if not self.dense_indexer:
            return "❌ RAG system not loaded"
        
        status = f"""
**🤖 Professional RAG System Status**

**📊 System Information:**
- Total Chunks: {len(self.dense_indexer.docs)}
- Model: {self.dense_indexer.model_name}
- BM25: {'✅ Loaded' if self.bm25 else '❌ Not Available'}
- Reranker: {'✅ Loaded' if self.reranker else '❌ Not Available'}

**🔧 Configuration:**
- Index Directory: {self.index_dir}
- Embedding Model: BGE-M3
- Reranking Model: BGE-Reranker-v2-M3

**📈 Performance:**
- Hybrid Search: ✅ Dense + BM25 + RRF
- Section-Aware Chunking: ✅
- Cross-Encoder Reranking: {'✅' if self.reranker else '❌'}
- Multi-language Support: ✅

**🕒 Last Updated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        return status


# ======================
# 9) Gradio Interface
# ======================

def create_professional_gradio_interface():
    """Create professional Gradio interface"""
    
    # Initialize RAG interface
    rag_interface = ProfessionalRAGProduct()
    
    with gr.Blocks(title="Professional RAG System", theme=gr.themes.Soft()) as interface:
        
        # Header
        gr.Markdown("""
        # 🤖 Professional RAG System
        **Hệ thống RAG chuyên nghiệp với chunking và ranking tối ưu**
        
        **Features**: Section-aware chunking, Hybrid retrieval, RRF fusion, Cross-encoder reranking
        """)
        
        with gr.Tabs():
            
            # Tab 1: PDF Import
            with gr.Tab("📚 Import PDF Files"):
                gr.Markdown("### Import PDF files for professional processing")
                
                pdf_files = gr.File(
                    label="Select PDF files",
                    file_count="multiple",
                    file_types=[".pdf"]
                )
                
                import_mode = gr.Radio(
                    choices=["append", "replace"],
                    value="replace",
                    label="Import Mode",
                    info="Append: Add to existing data | Replace: Clear and replace all data"
                )
                
                import_btn = gr.Button("🚀 Import & Process", variant="primary")
                import_output = gr.Markdown("No files imported yet.")
                system_status = gr.Markdown("System not loaded.")
                
                # Import event
                import_btn.click(
                    fn=rag_interface.import_pdf,
                    inputs=[pdf_files, import_mode],
                    outputs=[import_output, system_status]
                )
            
            # Tab 2: Professional Chat
            with gr.Tab("💬 Professional Chat"):
                gr.Markdown("### Chat with professional RAG system")
                
                chatbot = gr.Chatbot(
                    label="Professional Chat History",
                    height=500,
                    type="messages"
                )
                
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask about the imported documents...",
                        lines=2
                    )
                    send_btn = gr.Button("📤 Send", variant="primary")
                
                # Chat events
                msg_input.submit(
                    fn=rag_interface.chat_with_rag,
                    inputs=[msg_input, chatbot],
                    outputs=[msg_input, chatbot]
                )
                
                send_btn.click(
                    fn=rag_interface.chat_with_rag,
                    inputs=[msg_input, chatbot],
                    outputs=[msg_input, chatbot]
                )
            
            # Tab 3: System Status
            with gr.Tab("📊 System Status"):
                gr.Markdown("### Professional RAG system status")
                
                status_btn = gr.Button("🔄 Refresh Status", variant="primary")
                professional_status = gr.Markdown("System not loaded.")
                
                # Status event
                status_btn.click(
                    fn=rag_interface.get_status,
                    outputs=[professional_status]
                )
        
        # Footer
        gr.Markdown("""
        ---
        **🎯 Professional Features:**
        - ✅ Section-aware semantic chunking
        - ✅ Hybrid retrieval (Dense + BM25)
        - ✅ RRF (Reciprocal Rank Fusion)
        - ✅ Cross-encoder reranking
        - ✅ Multi-language support
        - ✅ Professional PDF processing
        
        **📚 Workflow:**
        1. **Import**: Upload PDF files
        2. **Process**: Section-aware chunking
        3. **Index**: Dense + BM25 indexing
        4. **Search**: Hybrid retrieval with RRF
        5. **Rerank**: Cross-encoder reranking
        6. **Answer**: Professional answer generation
        """)
    
    return interface


# ======================
# 10) Main Function
# ======================

def main():
    """Main function to run professional RAG product"""
    parser = argparse.ArgumentParser(description="Professional RAG Product")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--share", action="store_true", help="Share publicly")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    
    args = parser.parse_args()
    
    print("🚀 Starting Professional RAG Product...")
    print(f"🌐 Host: {args.host}")
    print(f"🔌 Port: {args.port}")
    print(f"🌍 Share: {args.share}")
    print(f"🐛 Debug: {args.debug}")
    
    try:
        # Create Gradio interface
        interface = create_professional_gradio_interface()
        
        # Launch
        interface.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            debug=args.debug
        )
        
    except Exception as e:
        print(f"❌ Error starting system: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
