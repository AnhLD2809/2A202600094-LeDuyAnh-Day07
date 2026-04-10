# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** Lê Duy Anh
**Nhóm:** C401-F1
**Ngày:** 10/4/2026

---

## 1. Warm-up (5 điểm)

### Cosine Similarity (Ex 1.1)

**High cosine similarity nghĩa là gì?**
> High cosine similarity nghĩa là hai vector có hướng gần giống nhau, tức là hai câu có ý nghĩa tương tự nhau.

**Ví dụ HIGH similarity:**
- Sentence A: "Tôi thích ăn phở"
- Sentence B: "Tôi thích ăn bún"
- Tại sao tương đồng: Cả hai câu đều nói về việc thích ăn một món ăn, chỉ khác nhau ở loại món ăn.

**Ví dụ LOW similarity:**
- Sentence A: "Tôi thích ăn phở"
- Sentence B: "Tôi thích học toán"
- Tại sao khác: Một câu nói về sở thích ăn uống, câu còn lại nói về sở thích học tập.

**Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?**
> Cosine similarity đo lường góc giữa hai vector, không phụ thuộc vào độ lớn của vector. Trong khi đó, Euclidean distance đo lường khoảng cách giữa hai vector, phụ thuộc vào độ lớn của vector. Do đó, cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings.

### Chunking Math (Ex 1.2)

**Document 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Số chunks = (10000 - 50) / (500 - 50) + 1 = 19 + 1 = 20 chunks*
> *Đáp án: 20 chunks*

**Nếu overlap tăng lên 100, chunk count thay đổi thế nào? Tại sao muốn overlap nhiều hơn?**
> Nếu overlap tăng lên 100, chunk count sẽ giảm xuống còn (10000 - 100) / (500 - 100) + 1 = 24 + 1 = 25 chunks
> Muốn overlap nhiều hơn để tăng khả năng giữ context giữa các chunks.

---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Luật lao động Việt Nam

**Tại sao nhóm chọn domain này?**
> Luật lao động Việt Nam là một lĩnh vực phức tạp với nhiều quy định khác nhau. Việc áp dụng RAG cho domain này sẽ giúp người dùng dễ dàng tra cứu thông tin và giải đáp các thắc mắc liên quan đến luật lao động.

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 |Bộ luật lao động 2019 | https://datafiles.chinhphu.vn/cpp/files/vbpq/2019/12/45.signed.pdf | 193202 | Không có |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| | | | |
| | | | |

---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 2-3 tài liệu:

| Tài liệu | Strategy | Chunk Count | Avg Length | Preserves Context? |
|-----------|----------|-------------|------------|-------------------|
| Luật lao động Việt Nam | FixedSizeChunker (`fixed_size`) | 1074 | 199.87 | Không |
| Luật lao động Việt Nam | SentenceChunker (`by_sentences`) | 554 | 346.92 | Có |
| Luật lao động Việt Nam | RecursiveChunker (`recursive`) | 1652 | 115.27 | Có |

### Strategy Của Tôi

**Loại:** Custom Strategy (Regex Based Chunking)

**Mô tả cách hoạt động:**
> Sử dụng regex để tách văn bản dựa trên các header "Điều X." trong Markdown để bảo toàn trọn vẹn ngữ cảnh của một Điều Luật.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> Luật lao động Việt Nam có cấu trúc phân cấp rõ ràng với các Chương, Mục và Điều. Việc tách văn bản dựa trên các header "Điều X." sẽ giúp bảo toàn trọn vẹn ngữ cảnh của một Điều Luật, từ đó tăng cường khả năng retrieval.

**Code snippet (nếu custom):**
```python
import re
from src.embeddings import OpenAIEmbedder

class LawDocumentChunker:
    """
    Custom Chunker dành riêng cho tài liệu Luật Pháp (Legal Documents).
    Tách văn bản dựa trên các header "Điều X." trong Markdown để bảo toàn trọn vẹn ngữ cảnh của một Điều Luật.
    """
    def chunk(self, text: str) -> list[str]:
        # Dùng regex lookahead để tách ngay trước các dấu # Điều (giữ lại chữ Điều ở chunk mới)
        # Pattern: \n(?=#{1,4}\s+Điều\s+\d+\.)
        chunks = re.split(r'\n(?=#{1,4}\s+Điều\s+\d+\.)', text)
        
        result = []
        for c in chunks:
            if len(c.strip()) > 30: # Bỏ qua các đoạn quá ngắn
                result.append(c.strip())
        return result
```

### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| Luật lao động Việt Nam | best baseline (Recursive) | 1652 | 115.27 | Trung bình |
| Luật lao động Việt Nam | **của tôi** | 214 | 900.82 | Xuất sắc |

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | Custom Strategy (Regex Based Chunking) | 8.5 | Bảo toàn ngữ cảnh tốt | Khi điều luật quá dài, đoạn chunk sinh ra sẽ vượt qua giới hạn context window. Hao phí khi embedding. Sự thừa thãi khi truy xuất.  |
| [Tên] | | | | |
| [Tên] | | | | |

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:*

---

## 4. My Approach — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi implement các phần chính trong package `src`.

### Chunking Functions

**`SentenceChunker.chunk`** — approach:
> Sử dụng `re.split` với biểu thức chính quy (regex) để cắt câu dựa trên các dấu câu (`. `, `! `, `? `, `.\n`) nhưng giữ lại luôn các dấu đó. Sau đó tiến hành gộp (join) các câu lại để đảm bảo mỗi chunk không vượt quá `max_sentences_per_chunk`. Edge case như khoảng trắng thừa đều được `.strip()` xử lý hiệu quả.

**`RecursiveChunker.chunk` / `_split`** — approach:
> Thuật toán đệ quy vòng tự chia nhỏ. Base case là khi độ dài đoạn văn nhỏ hơn `chunk_size` hoặc khi hết các ký tự separators (dấu ngắt ưu tiên). Nếu sau một lần split mà chuỗi con vẫn dài và vượt qua giới hạn `chunk_size`, thuật toán sẽ giữ lại trong stack nối chuỗi và chạy đệ quy gọi separator tiếp theo xuống các lớp sâu hơn.

### EmbeddingStore

**`add_documents` + `search`** — approach:
> Quản lý hai chế độ thông qua cờ `_use_chroma` (sử dụng library ChromaDB hay in-memory List). Ở in-memory, các document chunking được push tuần tự vào biến `self._store`. Với `search` của in-memory, em lặp qua record, tính điểm `compute_similarity` chuẩn hoá (dot product theo base formula) rồi sort danh sách giảm dần để bóc Top K.

**`search_with_filter` + `delete_document`** — approach:
> Với luồng filter trước, em lặp qua metadata dict so sánh từng key với filter_query rồi thu gom list thoả mãn điều kiện TRƯỚC KHI đem đi calculate similarity để tránh lãng phí vòng truy vấn tính toán nặng. Hàm delete sử dụng list comprehension duyệt và huỷ reference record để Python tự garbage collection dọn dẹp các chunk theo class matching.

### KnowledgeBaseAgent

**`answer`** — approach:
> Mảnh ghép cuối của luồng RAG. Bước đầu là móc lại hàm `search()` trong store để rút trích các tài liệu liên luân (top K chunk) nhất. Biến các context thô này thành template RAG prompt hệ thống, gộp kèm format input của user, và cuối cùng bơm sang LLM callback trả API Result.

### Test Results

```plain
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.0.3, pluggy-1.6.0
collected 42 items

... (42 file check logs skip) ...
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED [100%]

============================= 42 passed in 0.08s ==============================
```

**Số tests pass:** 42 / 42

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Trời hôm nay rất đẹp | Hôm nay thời tiết tốt | high | 0.663 | Đúng |
| 2 | Tôi thích học lập trình | Tôi đam mê viết code | high | 0.696 | Đúng |
| 3 | Mèo là loài động vật đáng yêu | Con chó nhà tôi rất khôn | low | 0.378 | Đúng |
| 4 | Bầu trời màu xanh | Sáng nay tôi ăn phở | low | 0.337 | Đúng |
| 5 | Học máy đang là xu hướng | AI và Machine Learning phát triển | high | 0.500 | Đúng |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**
> Kết quả dự đoán của tôi khá chính xác, tuy nhiên có một số trường hợp dự đoán sai. Điều này cho thấy rằng embeddings biểu diễn nghĩa không hoàn toàn tuyến tính và có thể bị ảnh hưởng bởi nhiều yếu tố khác nhau.
---

## 6. Results — Cá nhân (10 điểm)

Chạy 6 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. **6 queries phải trùng với các thành viên cùng nhóm.**

### Benchmark Queries & Gold Answers (nhóm thống nhất)

| # | Query | Gold Answer |
|---|-------|-------------|
| 1 | Bộ luật Lao động năm 2019 (Luật số 45/2019/QH14) chính thức có hiệu lực thi hành kể từ ngày tháng năm nào? | 1/1/2021 |
| 2 | Theo Bộ luật Lao động 2019, hợp đồng lao động được phân loại thành mấy loại chính? Đó là những loại nào? | Gồm 02 loại chính: Hợp đồng lao động không xác định thời hạn và Hợp đồng lao động xác định thời hạn (thời hạn không quá 36 tháng). (Lưu ý: Đã bỏ Hợp đồng lao động theo mùa vụ hoặc theo một công việc nhất định có thời hạn dưới 12 tháng so với bộ luật cũ). |
| 3 | Quy định pháp luật không cho phép áp dụng thời gian thử việc đối với trường hợp người lao động giao kết loại hợp đồng lao động nào? | Không áp dụng thử việc đối với người lao động giao kết hợp đồng lao động có thời hạn dưới 01 tháng.|
| 4 | Theo quy định, thời gian thử việc tối đa đối với công việc của người quản lý doanh nghiệp (theo quy định của Luật Doanh nghiệp, Luật Quản lý, sử dụng vốn nhà nước đầu tư vào sản xuất, kinh doanh tại doanh nghiệp) là bao nhiêu ngày? | Không quá 180 ngày. |
| 5 | Trong dịp lễ Quốc khánh 02/9, người lao động được nghỉ làm việc và hưởng nguyên lương tổng cộng bao nhiêu ngày? | 02 ngày (Bao gồm ngày 02 tháng 9 dương lịch và 01 ngày liền kề trước hoặc sau ngày 02 tháng 9). |
| 6 |  Lộ trình điều chỉnh tuổi nghỉ hưu đối với người lao động làm việc trong điều kiện lao động bình thường được thực hiện cho đến khi đạt mức độ tuổi nào đối với nam và nữ? | Nam đạt đủ 62 tuổi (vào năm 2028) và Nữ đạt đủ 60 tuổi (vào năm 2035). |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Bộ luật Lao động năm 2019 (Luật số 45/2019/QH14) chính thức có hiệu lực thi hành kể từ ngày tháng năm nào? | #### Điều 220. Hiệu lực thi hành... | 0.6279 | Có | Chính thức có hiệu lực thi hành kể từ ngày 01 tháng 01 năm 2021. |
| 2 | Theo Bộ luật Lao động 2019, hợp đồng lao động được phân loại thành mấy loại chính? Đó là những loại nào? | ### Điều 20. Loại hợp đồng lao động... | 0.5872 | Có | 2 loại chính: hợp đồng lao động không xác định thời hạn và xác định thời hạn. |
| 3 | Quy định pháp luật không cho phép áp dụng thời gian thử việc đối với trường hợp người lao động giao kết loại hợp đồng lao động nào? | #### Điều 24. Thử việc... | 0.6392 | Có | Không áp dụng thử việc đối với người lao động giao kết HĐLĐ dưới 01 tháng. |
| 4 | Theo quy định, thời gian thử việc tối đa đối với công việc của người quản lý doanh nghiệp là bao nhiêu ngày? | #### Điều 25. Thời gian thử việc... | 0.7421 | Có | Thời gian thử việc tối đa đối với người quản lý doanh nghiệp là 180 ngày. |
| 5 | Trong dịp lễ Quốc khánh 02/9, người lao động được nghỉ làm việc và hưởng nguyên lương tổng cộng bao nhiêu ngày? | #### Điều 112. Nghỉ lễ, tết... | 0.6271 | Có | Nghỉ làm việc và hưởng nguyên lương tổng cộng 02 ngày. |
| 6 | Lộ trình điều chỉnh tuổi nghỉ hưu đối với người lao động làm việc trong điều kiện lao động bình thường được thực hiện cho đến khi đạt mức độ tuổi nào đối với nam và nữ? | #### Điều 113. Nghỉ hằng năm... | 0.5469 | Không | Dựa trên ngữ cảnh cung cấp, không có thông tin để trả lời. |

**Bao nhiêu queries trả về chunk relevant trong top-3?** 5 / 6

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> 

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> *Viết 2-3 câu:*

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Kết hợp thêm recursive chunking để chia nhỏ các đoạn văn bản dài, giúp tăng khả năng truy xuất thông tin chính xác.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | / 5 |
| Document selection | Nhóm | / 10 |
| Chunking strategy | Nhóm | / 15 |
| My approach | Cá nhân | / 10 |
| Similarity predictions | Cá nhân | / 5 |
| Results | Cá nhân | / 10 |
| Core implementation (tests) | Cá nhân | / 30 |
| Demo | Nhóm | / 5 |
| **Tổng** | | **/ 100** |
