# 🤖 Professional RAG System (VI)

## 📦 **Product Overview**

**Professional RAG System** là một hệ thống RAG (Retrieval-Augmented Generation) chuyên nghiệp được tối ưu hóa cho tiếng Việt với các tính năng nâng cao:

- ✅ **Page-aware PDF processing** - Trích xuất PDF theo trang với metadata chính xác
- ✅ **Character-based chunking** - Chunking theo ký tự (1800 chars) ổn định hơn token
- ✅ **MMR diversification** - Maximum Marginal Relevance để đa dạng hóa kết quả
- ✅ **Advanced Vietnamese tokenization** - Hỗ trợ underthesea cho tiếng Việt
- ✅ **Hybrid retrieval** - Kết hợp Dense + BM25 + RRF
- ✅ **Cross-encoder reranking** - Reranking chuyên nghiệp
- ✅ **Centralized configuration** - Quản lý cấu hình tập trung
- ✅ **Professional Gradio interface** - Giao diện web thân thiện

## 🚀 **Quick Start**

### **1. Cài đặt**
```bash
pip install -r requirements.txt
```

### **2. Chạy hệ thống**
```bash
# Cách 1: Sử dụng script
./run.sh

# Cách 2: Chạy trực tiếp
python rag_product.py --host 0.0.0.0 --port 7860 --share
```

### **3. Truy cập**
```
🌐 Local: http://localhost:7860
🌍 Public: https://[gradio-share-url]
```

## 📁 **Product Structure**

```
RAG/
├── rag_product.py          # Main product file (upgraded)
├── requirements.txt         # Dependencies
├── run.sh                  # Launch script
├── README.md               # Documentation
├── PRODUCT_INFO.md         # Product information
├── data/                   # PDF files directory
│   └── imported/           # Imported PDF files
└── indexes/                # RAG indexes
    └── professional/       # Professional index files
```

## 🎯 **Key Features**

### **1. Page-Aware PDF Processing**
- Extract text với page metadata đầy đủ
- Section detection (headers, paragraphs, tables)
- Structure preservation với page numbers

### **2. Character-Based Chunking**
- Chunking theo ký tự (1800 chars) thay vì token
- Metadata rich (section, page, chunk_type)
- Smart overlap (200 characters)

### **3. MMR Diversification**
- Maximum Marginal Relevance để đa dạng hóa kết quả
- Tránh 5 đoạn giống nhau cùng 1 trang
- Lambda parameter để cân bằng relevance vs diversity

### **4. Advanced Vietnamese Support**
- Underthesea tokenization cho tiếng Việt
- Fallback mechanism nếu underthesea không có
- Better BM25 performance cho tiếng Việt

### **5. Hybrid Retrieval**
- **Dense**: BGE-M3 embedding (1024 dimensions)
- **Sparse**: BM25 với Vietnamese tokenization
- **Fusion**: RRF (Reciprocal Rank Fusion) với k=60
- **MMR**: Diversification để đa dạng kết quả

### **6. Cross-Encoder Reranking**
- BGE-Reranker-v2-M3 cho reranking chuyên nghiệp
- Top-100 → Top-20 với độ chính xác cao
- Multi-language support

### **7. Professional Answer Generation**
- Context building với metadata
- Source attribution chi tiết
- Ollama integration với error handling

## 🔧 **Configuration**

### **CFG Configuration Class**
```python
class CFG:
    # Đường dẫn mặc định
    DATA_DIR = Path("data/imported")
    INDEX_DIR = Path("indexes/professional")
    
    # Mô hình embedding & reranker
    EMBEDDING_MODEL = "BAAI/bge-m3"
    RERANK_MODEL = "BAAI/bge-reranker-v2-m3"
    
    # LLM (Ollama)
    OLLAMA_URL = "http://localhost:11434"
    OLLAMA_MODEL = "gpt-oss:20b"
    
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
    RERANK_THRESHOLD = 0.05
```

## 📊 **Performance**

### **Chunking Quality**
- ✅ Character-based: Chunks theo ký tự ổn định
- ✅ Page-aware: Đầy đủ thông tin nguồn
- ✅ Context preservation: Overlap thông minh

### **Retrieval Quality**
- ✅ Hybrid search: Kết hợp dense + sparse
- ✅ RRF fusion: Ổn định, không cần chuẩn hóa
- ✅ MMR diversification: Đa dạng kết quả
- ✅ Cross-encoder reranking: Độ chính xác cao

### **Answer Quality**
- ✅ Professional answers: Chính xác, có nguồn
- ✅ Source attribution: Trang, section, tài liệu
- ✅ Multi-language: Tiếng Việt ưu tiên

## 🚨 **Troubleshooting**

### **Lỗi thường gặp**

1. **"No such file or directory: bm25.pkl"**
   - Giải pháp: Đảm bảo thư mục index tồn tại

2. **"Ollama connection error"**
   - Giải pháp: Kiểm tra Ollama đang chạy tại http://localhost:11434

3. **"Model loading error"**
   - Giải pháp: Kiểm tra internet connection cho model download

4. **"Gradio launch error"**
   - Giải pháp: Kiểm tra port 7860 không bị chiếm

## 📈 **Expected Results**

### **Chunking Performance**
- Chunking time: ~1-2 seconds per PDF
- Metadata extraction: 100% accuracy
- Page detection: 100% accuracy

### **Retrieval Performance**
- Indexing time: ~10-30 seconds per 100 chunks
- Search time: ~1-3 seconds per query
- Reranking time: ~2-5 seconds per query

### **Answer Quality**
- Faithfulness: ≥0.8 (RAGAS)
- Answer relevance: ≥0.85 (RAGAS)
- Source attribution: 100% accuracy

## 🚀 **Deployment**

### **Local Deployment**
```bash
python rag_product.py --host 0.0.0.0 --port 7860
```

### **Public Deployment**
```bash
python rag_product.py --host 0.0.0.0 --port 7860 --share
```

### **Production Deployment**
```bash
# With process manager
nohup python rag_product.py --host 0.0.0.0 --port 7860 > rag.log 2>&1 &
```

## 📞 **Support**

- **Documentation**: README.md
- **Issues**: Check logs for error details
- **Performance**: Monitor system status tab
- **Updates**: Check for new versions

---

**Tác giả**: RAG Product Team  
**Phiên bản**: 2.0.0 (Upgraded)  
**Cập nhật**: 2024-10-24  
**Status**: ✅ Production Ready