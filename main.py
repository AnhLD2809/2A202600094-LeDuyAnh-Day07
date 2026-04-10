import os
import re
from pathlib import Path
from dotenv import load_dotenv

from src.models import Document
from src.chunking import ChunkingStrategyComparator, RecursiveChunker
from src.store import EmbeddingStore
from src.agent import KnowledgeBaseAgent
from src.embeddings import OpenAIEmbedder, _mock_embed

# ==========================================
# 1. TÙY CHỈNH CHUNKER (TỪ FILE BENCHMARK)
# ==========================================
class LawDocumentChunker:
    """
    Custom Chunker dành riêng cho tài liệu Luật Pháp (Legal Documents).
    Tách văn bản dựa trên các header "Điều X." trong Markdown để bảo toàn trọn vẹn ngữ cảnh của một Điều Luật.
    """
    def chunk(self, text: str) -> list[str]:
        # Dùng regex lookahead để tách ngay trước các dấu # Điều (giữ lại chữ Điều ở chunk mới)
        chunks = re.split(r'\n(?=#{1,4}\s+Điều\s+\d+\.)', text)
        result = []
        for c in chunks:
            if len(c.strip()) > 30: # Bỏ qua các đoạn quá rác/ngắn
                result.append(c.strip())
        return result

def real_llm_fn(prompt: str) -> str:
    """Gọi LLM thật từ OpenAI nếu có key, nếu không thì dùng mock."""
    load_dotenv(override=False)
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            import openai
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Trả lời tóm tắt dựa vào Context. Mọi thông tin không có trong Context thì PHẢI từ chối trả lời bằng câu: 'Dựa trên ngữ cảnh cung cấp, không có thông tin để trả lời.'"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0
            )
            return response.choices[0].message.content
        except ImportError:
            pass
    # Fallback default
    return "Vui lòng cài đặt 'openai' và khóa OPENAI_API_KEY để xem Agent trả lời."

# ==========================================
# 2. RUN BASELINE ANALYSIS (MỤC TÍNH TOÁN BASELINE)
# ==========================================
def run_baseline(text: str):
    print("\n" + "="*50)
    print("PHẦN 1: TÍNH TOÁN CHUNKING BASELINE (REPORT MỤC 3)")
    print("="*50)
    
    comparator = ChunkingStrategyComparator()
    # Chạy trên toàn bộ luật để đếm số liệu
    results = comparator.compare(text, chunk_size=200)
    
    for strategy_name, metrics in results.items():
        print(f"Strategy: {strategy_name}")
        print(f"  - Chunk Count: {metrics['count']}")
        print(f"  - Avg Length: {metrics['avg_length']:.2f}")
    
    print(f"\nStrategy: Custom (LawDocumentChunker)")
    custom_chunker = LawDocumentChunker()
    custom_chunks = custom_chunker.chunk(text)
    custom_avg = sum(len(c) for c in custom_chunks) / len(custom_chunks) if custom_chunks else 0
    print(f"  - Chunk Count: {len(custom_chunks)}")
    print(f"  - Avg Length: {custom_avg:.2f}")

# ==========================================
# 3. RUN BENCHMARK AGENT (MỤC KIỂM TRA BENCHMARK)
# ==========================================
def run_benchmark(text: str):
    print("\n" + "="*50)
    print("PHẦN 2: CHẠY BENCHMARK VECTOR STORE & AGENT (REPORT MỤC 6)")
    print("="*50)

    queries = [
        "Bộ luật Lao động năm 2019 (Luật số 45/2019/QH14) chính thức có hiệu lực thi hành kể từ ngày tháng năm nào?",
        "Theo Bộ luật Lao động 2019, hợp đồng lao động được phân loại thành mấy loại chính? Đó là những loại nào?",
        "Quy định pháp luật không cho phép áp dụng thời gian thử việc đối với trường hợp người lao động giao kết loại hợp đồng lao động nào?",
        "Theo quy định, thời gian thử việc tối đa đối với công việc của người quản lý doanh nghiệp là bao nhiêu ngày?",
        "Trong dịp lễ Quốc khánh 02/9, người lao động được nghỉ làm việc và hưởng nguyên lương tổng cộng bao nhiêu ngày?",
        "Lộ trình điều chỉnh tuổi nghỉ hưu đối với người lao động làm việc trong điều kiện lao động bình thường được thực hiện cho đến khi đạt mức độ tuổi nào đối với nam và nữ?"
    ]

    print("Đang phân mảnh (Chunking) tài liệu bằng LawDocumentChunker...")
    chunker = LawDocumentChunker()
    text_chunks = chunker.chunk(text)
    
    docs = []
    for i, c in enumerate(text_chunks):
        docs.append(Document(id=f"chunk_{i}", content=c, metadata={"source": "luat_lao_dong.md"}))

    print("Đang khởi tạo Vector Store...")
    load_dotenv(override=False)
    api_key = os.getenv("OPENAI_API_KEY")
    
    if api_key:
        print("Sử dụng: OpenAI Embedder (Vector thực tế)")
        embedder = OpenAIEmbedder(model_name="text-embedding-3-small")
    else:
        print("CẢNH BÁO: Không tìm thấy OPENAI_API_KEY, dùng MockEmbedder (kết quả sẽ bị sai ngữ nghĩa).")
        embedder = _mock_embed

    store = EmbeddingStore(collection_name="benchmark_store", embedding_fn=embedder)
    store.add_documents(docs)
    
    agent = KnowledgeBaseAgent(store=store, llm_fn=real_llm_fn)

    for i, q in enumerate(queries, start=1):
        print(f"\n[Q{i}] Query: {q}")
        
        search_results = store.search(q, top_k=3)
        if search_results:
            top_1 = search_results[0]
            preview = top_1['content'][:150].replace('\n', ' ')
            print(f" -> Top-1 Retrieved Chunk: {preview}...")
            print(f" -> Top-1 Score: {top_1['score']:.4f}")
            
            print(" -> [Các chunk khác trong Top 3]:")
            for j, c in enumerate(search_results[1:], start=2):
                short_c = c['content'][:150].replace("\n", " ") + "..."
                print(f"      Top {j}: {short_c} (Score: {c.get('score', 0.0):.4f})")
        else:
            print(" -> [Search Results]: Không tìm thấy.")

        answer = agent.answer(q, top_k=2)
        print(f" -> Agent Answer: {answer}")
        print("-" * 50)

# ==========================================
# MAIN ROUTINE
# ==========================================
def main() -> int:
    file_path = "data/luat_lao_dong.md"
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {file_path}")
        return 1

    run_baseline(text)
    run_benchmark(text)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())