# So sánh Kết quả Thực nghiệm với Các Nghiên cứu Tốt nhất Hiện nay

> **Hướng dẫn sử dụng:** Chèn từng đoạn phân tích vào **cuối** mục tương ứng trong chapter5.tex.  
> BibTeX cần được thêm vào `references.bib`. Citation `\cite{}` dùng key đã ghi.

---

## MỤC 4.1 — Kết quả Prompting

---

### 4.1.A: So sánh SOTA — Essays Dataset (Prompting: Zero-shot, Few-shot, CoT)

#### Đoạn phân tích (chèn vào cuối mục 4.1.1 hoặc 4.1.6)

> **Cách chèn LaTeX:**
> ```latex
> \paragraph{So sánh với các nghiên cứu tốt nhất hiện nay.}
> ```

Trong toàn bộ các cấu hình prompting được đánh giá trên Essays Dataset,
GPT-4.1-mini Few-shot là cấu hình tốt nhất với Accuracy trung bình $57{,}46\%$ và
Macro-F1 $54{,}27\%$. Để định vị kết quả này trong bối cảnh nghiên cứu hiện tại,
luận văn đối chiếu với công trình của Yang et al.~\cite{yang2023psycot}, vốn đánh
giá GPT-3.5 theo ba chiến lược trên cùng bộ dữ liệu Essays: standard prompting
(zero-shot) đạt Accuracy $58{,}70\%$ và Macro-F1 $54{,}20\%$; Zero-shot-CoT đạt
Accuracy $57{,}58\%$ và Macro-F1 $53{,}05\%$; và phương pháp đề xuất PsyCoT---tích
hợp questionnaire 44-item Big Five Inventory làm chain-of-thought qua multi-turn
dialogue---đạt Accuracy $59{,}64\%$ và Macro-F1 $58{,}43\%$. Đối chiếu trực tiếp,
kết quả Few-shot GPT-4.1-mini của luận văn (F1 $=54{,}27\%$) tương đương với
standard prompting GPT-3.5 zero-shot (F1 $=54{,}20\%$) của Yang et al., chênh lệch
chỉ $0{,}07$ điểm---một khoảng cách không có ý nghĩa thực tế. Tuy nhiên, khoảng
cách so với PsyCoT là $4{,}16$ điểm F1, phản ánh hiệu quả của việc cấu trúc hóa
chuỗi suy luận bằng kiến thức tâm lý học tường minh. Sự chênh lệch này xuất phát
từ hai yếu tố: (i) PsyCoT sử dụng giao diện multi-turn với 44 bước đánh giá riêng
cho từng khía cạnh trait, cho phép mô hình phân tích hành vi ngôn ngữ từng phần
trước khi tổng hợp; trong khi luận văn sử dụng single-turn few-shot buộc mô hình
tổng hợp toàn bộ năm chiều OCEAN trong một bước; (ii) GPT-4.1-mini là mô hình
nhỏ hơn và được tối ưu hóa cho độ trễ thấp, khác với GPT-3.5-turbo dùng trong
PsyCoT, nên hiệu năng có thể được cải thiện nếu áp dụng PsyCoT với GPT-4.

Về Chain-of-Thought, kết quả của luận văn cho thấy xu hướng suy giảm nhất quán:
GPT-4.1-mini CoT chỉ đạt Macro-F1 $44{,}76\%$---thấp hơn $9{,}51$ điểm so với
cấu hình Few-shot của cùng mô hình. Xu hướng này đồng thuận với Yang et
al.~\cite{yang2023psycot}: Zero-shot-CoT (``Let's think step by step'') không cải
thiện mà còn giảm hiệu năng so với standard prompting ($53{,}05\%$ vs
$54{,}20\%$). Điều này xác nhận rằng CoT không có cấu trúc tâm lý học cụ thể
không phù hợp với bài toán APR trên Essays, nơi nhãn tính cách không suy ra được
bằng chuỗi logic tường minh. Ganesan et al.~\cite{ganesan2023systematic} cũng ghi
nhận rằng zero-shot GPT-3 trên tập Facebook posts chỉ đạt Macro-F1 $45{,}4\%$
ngay cả khi tiêm thêm ITEMDESC knowledge---thấp hơn SotA lexical baseline (WT-LEX,
F1 $=51{,}8\%$)---gợi ý rằng bài toán nhận diện tính cách từ văn bản ngắn đòi hỏi
tích hợp tri thức tâm lý học tường minh, chứ không chỉ năng lực ngôn ngữ chung.

#### Bảng so sánh 4.1.A

| Công trình | Venue | Dataset | Model | Phương pháp | Metric | Kết quả | Chênh lệch so với luận văn |
|---|---|---|---|---|---|---|---|
| Yang et al.~\cite{yang2023psycot} | EMNLP Findings 2023 | Essays (N≈247/trait) | GPT-3.5 | PsyCoT (CoT + questionnaire) | Macro-F1 | 58.43% | −4.16% |
| Yang et al.~\cite{yang2023psycot} | EMNLP Findings 2023 | Essays | GPT-3.5 | Standard prompting (zero-shot) | Macro-F1 | 54.20% | +0.07% |
| Yang et al.~\cite{yang2023psycot} | EMNLP Findings 2023 | Essays | GPT-3.5 | Zero-shot-CoT | Macro-F1 | 53.05% | +1.22% |
| Ganesan et al.~\cite{ganesan2023systematic} | ACL WASSA 2023 | Facebook posts (N=142) | GPT-3 | Zero-shot ITEMDESC | Macro-F1 | 45.4% | +8.87% |
| **Luận văn (tốt nhất)** | — | Essays (N=493) | GPT-4.1-mini | Few-shot | Macro-F1 | **54.27%** | — |

> **Lưu ý:** Ganesan et al. \cite{ganesan2023systematic} dùng tập Facebook posts (N=142), khác cấu trúc với Essays dataset. Kết quả chỉ mang tính tham khảo về năng lực zero-shot của GPT-3.

---

### 4.1.B: So sánh SOTA — MBTI PersonalityCafe Dataset (Prompting)

#### Đoạn phân tích (chèn vào cuối phần MBTI trong 4.1.1 hoặc 4.1.6)

Trên MBTI PersonalityCafe Dataset, cấu hình tốt nhất của luận văn là
Mistral-Small-24B Zero-shot với Binary Macro-F1 $=75{,}90\%$ và Macro-Accuracy
$=79{,}08\%$. Để định vị kết quả này, luận văn đối chiếu với bảng tổng hợp toàn
diện nhất hiện có trên cùng bộ dữ liệu Kaggle/PersonalityCafe, được công bố bởi
Cao et al.~\cite{cao2026perdet} tại AAAI 2026. Theo đó, ChatGPT (gpt-3.5-turbo,
zero-shot) đạt Binary Macro-F1 $=63{,}89\%$ và Qwen-plus (zero-shot) đạt
$76{,}43\%$. Mistral-Small-24B của luận văn ($75{,}90\%$) xấp xỉ Qwen-plus với
khoảng cách chỉ $0{,}53$ điểm F1, bất chấp thực tế Qwen-plus là flagship model
của Alibaba với quy mô tham số lớn hơn đáng kể. Điều đáng chú ý là kết quả
zero-shot prompting của luận văn ($75{,}90\%$) vượt trội D-DGCN ($71{,}35\%$,
Yang et al.~\cite{yang2023ddgcn}, AAAI 2023)---mô hình fine-tuned dựa trên
domain-adapted BERT kết hợp Dynamic Graph Convolutional Network---cho thấy sức
mạnh của zero-shot prompting với các LLM hiện đại có thể vượt qua fine-tuning
truyền thống trên task MBTI.

Khoảng cách so với PerDet-R1 ($80{,}57\%$,
Cao et al.~\cite{cao2026perdet})---SOTA hiện tại sử dụng Qwen2.5-7B với
SFT+GRPO+NDCG ranking---là $4{,}67$ điểm. Khoảng cách này phản ánh lợi thế của
việc cập nhật tham số theo tín hiệu domain-specific: PerDet-R1 không chỉ fine-tune
mô hình với supervised data mà còn áp dụng Reinforcement Learning với phần thưởng
dựa trên NDCG, giúp tối ưu hóa trực tiếp vị trí xếp hạng thay vì chỉ phân loại
nhị phân. Trong khi đó, prompting không thể thay đổi biểu diễn nội tại của mô hình
để học các đặc trưng domain-specific của văn bản forum PersonalityCafe.

Cần lưu ý rằng PsyCoT~\cite{yang2023psycot} trên Kaggle MBTI chỉ được đánh giá
trên 200 mẫu ngẫu nhiên từ tập test, không phải toàn bộ test set, nên kết quả
$65{,}22\%$ F1 không đáp ứng tiêu chí so sánh đầy đủ và chỉ được liệt kê để tham
khảo.

#### Bảng so sánh 4.1.B

| Công trình | Venue | Dataset | Model | Phương pháp | Binary Macro-F1 | Chênh lệch |
|---|---|---|---|---|---|---|
| Cao et al.~\cite{cao2026perdet} | AAAI 2026 | Kaggle (N=8675, full) | Qwen2.5-7B | SFT+GRPO (SOTA) | 80.57% | −4.67% |
| Cao et al.~\cite{cao2026perdet} | AAAI 2026 | Kaggle (full) | Qwen-plus | Zero-shot prompting | 76.43% | −0.53% |
| Bi et al.~\cite{bi2025etm} | AAAI 2025 | Kaggle (full) | LLM-enhanced | Fine-tuning (ETM) | 77.79% | −1.89% |
| Yang et al.~\cite{yang2023ddgcn} | AAAI 2023 | Kaggle (full) | BERT+DGCN | Fine-tuning (D-DGCN) | 71.35% | +4.55% ↑ |
| Cao et al.~\cite{cao2026perdet} | AAAI 2026 | Kaggle (full) | ChatGPT (gpt-3.5) | Zero-shot | 63.89% | +12.01% ↑ |
| Yang et al.~\cite{yang2023psycot} | EMNLP Findings 2023 | Kaggle (N=200\*) | GPT-3.5 | PsyCoT (CoT) | 65.22%\* | +10.68% ↑ |
| **Luận văn (tốt nhất)** | — | PersonalityCafe (N=8669) | Mistral-Small-24B | Zero-shot | **75.90%** | — |

> \* PsyCoT trên Kaggle MBTI chỉ được test trên 200 mẫu ngẫu nhiên (không phải full test set), không đáp ứng tiêu chí so sánh chuẩn; giá trị chỉ mang tính tham khảo.

---

## MỤC 4.2.1 & 4.2.2 — Mistral-7B và Qwen2.5-3B trên Essays Dataset (Fine-tuning)

### Đoạn phân tích (chèn vào cuối 4.2.1 và 4.2.2, mỗi mục một phiên bản)

#### Phiên bản cho 4.2.1 (Mistral-7B)

Mistral-7B-Instruct sau QLoRA fine-tuning trên Essays Dataset đạt Accuracy
$55{,}50\%$ và Macro-F1 trung bình $53{,}97\%$ (SFT) và $51{,}46\%$ (IFT). Để
định vị kết quả này, luận văn đối chiếu với các phương pháp fine-tuning được báo
cáo trong Yang et al.~\cite{yang2023psycot} trên cùng bộ dữ liệu Essays với quy
trình train/val/test 80-10-10. BERT-base fine-tuned (full fine-tuning) đạt Accuracy
$57{,}91\%$ và Macro-F1 $57{,}13\%$; RoBERTa-base fine-tuned đạt Accuracy
$58{,}38\%$ và Macro-F1 $57{,}64\%$. Kết quả SFT của Mistral-7B thấp hơn RoBERTa
khoảng $2{,}88\%$ Accuracy và $3{,}67\%$ Macro-F1.

Khoảng cách này có nguồn gốc từ sự khác biệt kiến trúc và phương pháp huấn luyện.
BERT và RoBERTa là mô hình encoder-only được fine-tuned theo phương pháp full
fine-tuning với classification head chuyên biệt, cho phép toàn bộ tham số cập nhật
theo tín hiệu cross-entropy loss trực tiếp. Ngược lại, QLoRA chỉ cập nhật một phần
nhỏ tham số thông qua low-rank adapter (r=8 đến 16), hạn chế khả năng điều chỉnh
biểu diễn nội tại để học đặc trưng tính cách từ Essays. Quan trọng hơn, tập huấn
luyện Essays chỉ cung cấp khoảng $395$ mẫu/lớp/trait---quá ít so với năng lực
của mô hình 7B tham số---dẫn đến gradient không đủ để cập nhật adapter có hiệu
quả. Điều này nhất quán với quan sát của Mục~\ref{sec:so-sanh}: tập dữ liệu nhỏ
không đủ để khai thác tiềm năng của fine-tuning LLM quy mô lớn.

Đáng chú ý, phương pháp PsyCoT~\cite{yang2023psycot} (GPT-3.5 zero-shot với CoT
structured bằng questionnaire) đạt Macro-F1 $58{,}43\%$---vượt cả BERT lẫn RoBERTa
fine-tuned, và cao hơn Mistral-7B SFT khoảng $4{,}46$ điểm---mà không cần bất kỳ
tham số nào được cập nhật. Kết quả này xác nhận luận điểm rằng với dataset nhỏ như
Essays, prompting có cấu trúc từ LLM mạnh có thể vượt trội fine-tuning trên mô hình
vừa.

Về chiều Gemma, luận văn không có dữ liệu thực nghiệm với họ mô hình này. Theo tìm
hiểu tài liệu, chưa tìm thấy công bố phù hợp đánh giá Gemma SFT/IFT trên Essays
Dataset với toàn bộ test set theo quy trình chuẩn, nên không thực hiện so sánh cho
hạng mục này.

#### Phiên bản cho 4.2.2 (Qwen2.5-3B)

Qwen2.5-3B-Instruct sau QLoRA fine-tuning trên Essays đạt Accuracy $55{,}50\%$ và
Macro-F1 $52{,}65\%$ (SFT) và $51{,}31\%$ (IFT)---kết quả tương đương Mistral-7B
mặc dù nhỏ hơn về số tham số. So sánh với BERT ($57{,}13\%$ F1) và RoBERTa
($57{,}64\%$ F1) fine-tuned~\cite{yang2023psycot}, Qwen2.5-3B SFT thấp hơn lần
lượt $4{,}48$ và $4{,}99$ điểm Macro-F1. Khoảng cách này phản ánh cùng nguyên nhân
đã phân tích ở Mục~\ref{subsec:ket_qua_mistral_essays}: quy mô dữ liệu Essays
không đủ để fine-tuning LLM quy mô lớn vượt trội encoder model được huấn luyện
full fine-tuning chuyên biệt.

#### Bảng so sánh 4.2.1–4.2.2 (Essays Fine-tuning)

| Công trình | Venue | Dataset | Model | Phương pháp | Accuracy | Macro-F1 | Chênh lệch F1 (vs Mistral SFT) |
|---|---|---|---|---|---|---|---|
| Yang et al.~\cite{yang2023psycot} | EMNLP Findings 2023 | Essays (N≈247/trait) | GPT-3.5 | PsyCoT (zero-shot) | 59.64% | 58.43% | −4.46% |
| Yang et al.~\cite{yang2023psycot} | EMNLP Findings 2023 | Essays | RoBERTa-base | Full fine-tuning | 58.38% | 57.64% | −3.67% |
| Yang et al.~\cite{yang2023psycot} | EMNLP Findings 2023 | Essays | BERT-base | Full fine-tuning | 57.91% | 57.13% | −3.16% |
| **Luận văn (Mistral-7B SFT)** | — | Essays (N=493) | Mistral-7B | QLoRA SFT | 55.50% | **53.97%** | — |
| **Luận văn (Qwen2.5-3B SFT)** | — | Essays (N=493) | Qwen2.5-3B | QLoRA SFT | 55.50% | **52.65%** | −1.32% |
| **Luận văn (Mistral-7B IFT)** | — | Essays (N=493) | Mistral-7B | QLoRA IFT | 52.50% | **51.46%** | −2.51% |
| **Luận văn (Qwen2.5-3B IFT)** | — | Essays (N=493) | Qwen2.5-3B | QLoRA IFT | 52.50% | **51.31%** | −2.66% |

---

## MỤC 4.2.3 — Mistral-7B trên MBTI PersonalityCafe (Fine-tuning)

### Đoạn phân tích (chèn vào cuối 4.2.3)

Mistral-7B-Instruct sau QLoRA fine-tuning trên MBTI PersonalityCafe đạt Binary
Macro-F1 $65{,}10\%$ (SFT) và $65{,}24\%$ (IFT). Kết quả này cao hơn ChatGPT
zero-shot ($63{,}89\%$, Cao et al.~\cite{cao2026perdet}) $1{,}21$ điểm, nhưng thấp
hơn D-DGCN ($71{,}35\%$, Yang et al.~\cite{yang2023ddgcn}) khoảng $6{,}11$ điểm.
Khoảng cách với ETM ($77{,}79\%$, Bi et al.~\cite{bi2025etm}) và PerDet-R1
($80{,}57\%$, Cao et al.~\cite{cao2026perdet}) lần lượt là $12{,}55$ và
$15{,}33$ điểm.

Việc Mistral-7B QLoRA SFT không vượt được D-DGCN---mô hình encoder-based từ
2023---có thể được giải thích bởi một số yếu tố. Thứ nhất, D-DGCN áp dụng
domain-adapted BERT kết hợp với Dynamic Graph Convolutional Network để mô hình hóa
mối quan hệ theo cấu trúc giữa các post của cùng một user; phương pháp mô hình hóa
cấu trúc graph phù hợp hơn với đặc thù dữ liệu forum nhiều post/user so với single-sequence encoding của causal LM. Thứ hai, QLoRA chỉ cập nhật adapter trong
decoder, trong khi task MBTI classification đòi hỏi biểu diễn toàn cục của nhiều
post (mỗi user có tới 50 post), điều mà causal LM xử lý kém hơn encoder với pooled
representation. Thứ ba, phương pháp ETM~\cite{bi2025etm} và TAE~\cite{hu2024tae}
tận dụng khả năng sinh văn bản của LLM để tạo ra biểu diễn ngữ nghĩa phong phú hơn
(semantic, sentiment, linguistic analysis) trước khi đưa vào classifier---một hướng
kết hợp LLM và fine-tuning chuyên biệt mà luận văn chưa khám phá.

So sánh với kết quả ablation trong PerDet-R1~\cite{cao2026perdet}: Qwen2.5-7B SFT
đơn thuần (không có GRPO) đạt $65{,}03\%$ Binary Macro-F1---xấp xỉ Mistral-7B SFT
của luận văn ($65{,}10\%$)---cho thấy kết quả của luận văn nhất quán với kết quả
SFT đơn thuần của các mô hình cùng quy mô, xác nhận tính hợp lệ của baseline.
Khoảng cách lớn giữa Mistral-7B SFT ($65{,}10\%$) và Qwen2.5-3B SFT ($79{,}80\%$)
của luận văn cho thấy kiến trúc và pre-training data của Qwen2.5 phù hợp hơn với
task MBTI từ văn bản forum tiếng Anh, và là yếu tố quyết định hơn quy mô tham số
trong trường hợp này.

Về Gemma SFT/IFT trên MBTI: **Chưa tìm thấy công bố phù hợp** đánh giá họ mô hình
Gemma trên MBTI PersonalityCafe với toàn bộ test set theo quy trình chuẩn (không
phải toy benchmark). Không thực hiện so sánh cho hạng mục này.

Về Mistral SFT/IFT cụ thể trên MBTI: Ngoài kết quả của luận văn, **chưa tìm thấy
công bố phù hợp** báo cáo Mistral-7B SFT/IFT trên MBTI PersonalityCafe với full
test set. PerDet-R1 chỉ báo cáo LLaMA3-8B SFT+GRPO ($72{,}57\%$) và không có
baseline Mistral riêng biệt.

#### Bảng so sánh 4.2.3 (Mistral-7B on MBTI)

| Công trình | Venue | Dataset | Model | Phương pháp | Binary Macro-F1 | Chênh lệch |
|---|---|---|---|---|---|---|
| Cao et al.~\cite{cao2026perdet} | AAAI 2026 | Kaggle (N=8675, full) | Qwen2.5-7B | SFT+GRPO (SOTA) | 80.57% | −15.33% |
| Bi et al.~\cite{bi2025etm} | AAAI 2025 | Kaggle (full) | LLM-enhanced | ETM fine-tuning | 77.79% | −12.55% |
| Hu et al.~\cite{hu2024tae} | AAAI 2024 | Kaggle (full) | LLM-augmented | TAE fine-tuning | 72.07% | −6.83% |
| Yang et al.~\cite{yang2023ddgcn} | AAAI 2023 | Kaggle (full) | BERT+DGCN | D-DGCN fine-tuning | 71.35% | −6.11% |
| Cao et al.~\cite{cao2026perdet} | AAAI 2026 | Kaggle (full) | Qwen2.5-7B (ablation) | SFT only (no GRPO) | 65.03% | +0.07% |
| Cao et al.~\cite{cao2026perdet} | AAAI 2026 | Kaggle (full) | ChatGPT (gpt-3.5) | Zero-shot | 63.89% | +1.21% ↑ |
| **Luận văn (Mistral-7B SFT)** | — | PersonalityCafe (N=1734) | Mistral-7B | QLoRA SFT | **65.10%** | — |
| **Luận văn (Mistral-7B IFT)** | — | PersonalityCafe (N=1734) | Mistral-7B | QLoRA IFT | **65.24%** | +0.14% |

---

## MỤC 4.2.4 — Qwen2.5-3B trên MBTI PersonalityCafe (Fine-tuning)

### Đoạn phân tích (chèn vào cuối 4.2.4)

Qwen2.5-3B-Instruct SFT đạt Binary Macro-F1 $=79{,}80\%$---kết quả nổi bật nhất
trong toàn bộ thực nghiệm của luận văn và cạnh tranh trực tiếp với nghiên cứu
state-of-the-art hiện tại. So sánh với PerDet-R1 ($80{,}57\%$,
Cao et al.~\cite{cao2026perdet})---SOTA trên Kaggle sử dụng Qwen2.5-7B với
SFT+GRPO+NDCG ranking---kết quả của luận văn chỉ thấp hơn $0{,}77$ điểm F1. Đáng
chú ý hơn, Qwen2.5-3B SFT của luận văn sử dụng mô hình nhỏ hơn $2{,}3\times$ về
số tham số (3B vs 7B) và phương pháp đơn giản hơn đáng kể (QLoRA classification
SFT vs SFT+GRPO ranking). Kết quả này vượt trội ETM ($77{,}79\%$,
Bi et al.~\cite{bi2025etm}) $+2{,}01$ điểm, TAE ($72{,}07\%$,
Hu et al.~\cite{hu2024tae}) $+7{,}73$ điểm, và D-DGCN ($71{,}35\%$,
Yang et al.~\cite{yang2023ddgcn}) $+8{,}45$ điểm.

Có thể giải thích kết quả này từ ba góc độ. Thứ nhất, Qwen2.5 được pre-trained
trên corpus tiếng Anh phong phú bao gồm nhiều loại diễn đàn cộng đồng, tạo biểu
diễn ngữ nghĩa phù hợp hơn với phong cách viết của PersonalityCafe so với Mistral-7B. Thứ hai, task classification SFT với BCE loss tối ưu hóa trực tiếp theo nhãn
binary của bốn chiều MBTI, phù hợp về objective với quy trình đánh giá; trong khi
các phương pháp graph-based (D-DGCN) tối ưu hóa qua graph convolution với nhiều
bước gián tiếp hơn. Thứ ba, dữ liệu PersonalityCafe sau loại bỏ label leakage vẫn
cung cấp khoảng $1{,}387$ training samples/trait---đủ để fine-tuning mô hình 3B
tham số đạt hiệu năng cao.

Khoảng cách $0{,}77$ điểm so với PerDet-R1 phản ánh lợi thế của hai yếu tố bổ
sung: (i) quy mô mô hình lớn hơn (7B vs 3B); và (ii) GRPO reinforcement learning
với NDCG reward, giúp mô hình học mối quan hệ giữa 16 loại MBTI như một bài toán
ranking thay vì bốn binary classification độc lập, từ đó nắm bắt được tương tác
giữa các chiều tính cách. Nếu áp dụng GRPO tương tự cho Qwen2.5-3B hoặc nâng cấp
lên Qwen2.5-7B, kết quả nhiều khả năng sẽ cạnh tranh hoặc vượt qua SOTA hiện tại.

#### Bảng so sánh 4.2.4 (Qwen2.5-3B on MBTI)

| Công trình | Venue | Dataset | Model | Phương pháp | Binary Macro-F1 | Chênh lệch |
|---|---|---|---|---|---|---|
| Cao et al.~\cite{cao2026perdet} | AAAI 2026 | Kaggle (N=8675, full) | Qwen2.5-7B | SFT+GRPO (SOTA) | 80.57% | −0.77% |
| Bi et al.~\cite{bi2025etm} | AAAI 2025 | Kaggle (full) | LLM-enhanced | ETM fine-tuning | 77.79% | **+2.01% ↑** |
| Hu et al.~\cite{hu2024tae} | AAAI 2024 | Kaggle (full) | LLM-augmented | TAE fine-tuning | 72.07% | **+7.73% ↑** |
| Yang et al.~\cite{yang2023ddgcn} | AAAI 2023 | Kaggle (full) | BERT+DGCN | D-DGCN fine-tuning | 71.35% | **+8.45% ↑** |
| Cao et al.~\cite{cao2026perdet} | AAAI 2026 | Kaggle (full) | ChatGPT (gpt-3.5) | Zero-shot | 63.89% | **+15.91% ↑** |
| **Luận văn (Qwen2.5-3B SFT)** | — | PersonalityCafe (N=1734) | Qwen2.5-3B | QLoRA SFT | **79.80%** | — |
| **Luận văn (Qwen2.5-3B IFT)** | — | PersonalityCafe (N=1734) | Qwen2.5-3B | QLoRA IFT | **68.20%** | −11.60% |

---

## BIBTEX ENTRIES (thêm vào references.bib)

```bibtex
@inproceedings{yang2023psycot,
  title     = {{PsyCoT}: Psychological Questionnaire as Powerful
               Chain-of-Thought for Personality Detection},
  author    = {Yang, Tao and Shi, Tianyuan and Wan, Fanqi and
               Quan, Xiaojun and Wang, Qifan and Wu, Bingzhe and
               Wu, Jiaxiang},
  booktitle = {Findings of the Association for Computational
               Linguistics: {EMNLP} 2023},
  pages     = {3305--3320},
  year      = {2023},
  address   = {Singapore},
  publisher = {Association for Computational Linguistics}
}

@inproceedings{yang2023ddgcn,
  title     = {Orders Are Unwanted: Dynamic Deep Graph Convolutional
               Network for Personality Detection},
  author    = {Yang, Tao and Deng, Jinghao and Quan, Xiaojun and
               Wang, Qifan},
  booktitle = {Proceedings of the 37th {AAAI} Conference on
               Artificial Intelligence},
  volume    = {37},
  pages     = {13896--13904},
  year      = {2023}
}

@inproceedings{ganesan2023systematic,
  title     = {Systematic Evaluation of {GPT}-3 for Zero-Shot
               Personality Estimation},
  author    = {Ganesan, Adithya V and Lal, Yash Kumar and
               Nilsson, August H{\aa}kan and Schwartz, H Andrew},
  booktitle = {Proceedings of the 13th Workshop on Computational
               Approaches to Subjectivity, Sentiment, \& Social
               Media Analysis},
  pages     = {390--400},
  year      = {2023},
  publisher = {Association for Computational Linguistics}
}

@inproceedings{cao2026perdet,
  title     = {From Classification to Ranking: Enhancing {LLM}
               Reasoning Capabilities for {MBTI} Personality
               Detection},
  author    = {Cao, Yuan and Liu, Feixiang and Wang, Xinyue and
               Zhu, Yihan and Xu, Hui and Wang, Zheng and Qiu,
               Qiang},
  booktitle = {Proceedings of the 40th {AAAI} Conference on
               Artificial Intelligence},
  year      = {2026}
}

@inproceedings{bi2025etm,
  title     = {Leveraging the Dual Capabilities of {LLM}:
               {LLM}-Enhanced Text Mapping Model for Personality
               Detection},
  author    = {Bi, Wenqi and Kou, Fangfang and Shi, Lei and Li,
               Ying and Li, Hong and Chen, Jing and Xu, Mingliang},
  booktitle = {Proceedings of the 39th {AAAI} Conference on
               Artificial Intelligence},
  pages     = {23487--23495},
  year      = {2025}
}

@inproceedings{hu2024tae,
  title     = {{LLM} vs Small Model? Large Language Model Based Text
               Augmentation Enhanced Personality Detection Model},
  author    = {Hu, Linmei and He, Hongfei and Wang, Duokang and
               Zhao, Zhuoman and Shao, Yinglong and Nie, Liqiang},
  booktitle = {Proceedings of the 38th {AAAI} Conference on
               Artificial Intelligence},
  pages     = {18234--18242},
  year      = {2024}
}
```

---

## IEEE CITATIONS (dùng trong phần tài liệu tham khảo kiểu IEEE)

```
[1] T. Yang, T. Shi, F. Wan, X. Quan, Q. Wang, B. Wu, and J. Wu,
    "PsyCoT: Psychological Questionnaire as Powerful Chain-of-Thought
    for Personality Detection," in Findings of the Association for
    Computational Linguistics: EMNLP 2023, Singapore, 2023,
    pp. 3305–3320.

[2] T. Yang, J. Deng, X. Quan, and Q. Wang, "Orders Are Unwanted:
    Dynamic Deep Graph Convolutional Network for Personality Detection,"
    in Proc. 37th AAAI Conf. Artif. Intell., 2023, vol. 37,
    pp. 13896–13904.

[3] A. V. Ganesan, Y. K. Lal, A. H. Nilsson, and H. A. Schwartz,
    "Systematic Evaluation of GPT-3 for Zero-Shot Personality
    Estimation," in Proc. 13th Workshop Comput. Approaches
    Subjectivity Sentiment Social Media Anal. (WASSA), 2023,
    pp. 390–400.

[4] Y. Cao, F. Liu, X. Wang, Y. Zhu, H. Xu, Z. Wang, and Q. Qiu,
    "From Classification to Ranking: Enhancing LLM Reasoning
    Capabilities for MBTI Personality Detection," in Proc. 40th AAAI
    Conf. Artif. Intell., 2026.

[5] W. Bi, F. Kou, L. Shi, Y. Li, H. Li, J. Chen, and M. Xu,
    "Leveraging the Dual Capabilities of LLM: LLM-Enhanced Text
    Mapping Model for Personality Detection," in Proc. 39th AAAI
    Conf. Artif. Intell., 2025, pp. 23487–23495.

[6] L. Hu, H. He, D. Wang, Z. Zhao, Y. Shao, and L. Nie,
    "LLM vs Small Model? Large Language Model Based Text Augmentation
    Enhanced Personality Detection Model," in Proc. 38th AAAI Conf.
    Artif. Intell., 2024, pp. 18234–18242.
```

---

## GHI CHÚ QUAN TRỌNG

### Các hạng mục "Chưa tìm thấy công bố phù hợp"

| Hạng mục | Lý do |
|---|---|
| Gemma SFT/IFT trên Essays | Không có bài báo công bố kết quả Gemma trên Essays dataset với full test set |
| Gemma SFT/IFT trên MBTI | Không có bài báo công bố kết quả Gemma trên PersonalityCafe với full test set |
| Mistral SFT/IFT trên MBTI (từ các tác giả khác) | PerDet-R1 chỉ báo cáo LLaMA3-8B và Qwen2.5-7B; không có baseline Mistral độc lập |
| CoT tốt nhất trên MBTI | Không có công bố CoT trên PersonalityCafe với full test set (PsyCoT chỉ test 200 mẫu) |

### Lưu ý về so sánh dataset

- **Essays**: Luận văn dùng test split 20% (N=493/trait), PsyCoT dùng 10% (N≈247/trait).
  Kết quả vẫn so sánh được vì cùng dataset nguồn.
- **MBTI**: Luận văn dùng PersonalityCafe sau de-leaking (N=1734 test, ~8675 total).
  PerDet-R1, D-DGCN, ETM, TAE đều dùng full Kaggle/PersonalityCafe với de-leaking tương tự.
  So sánh là hợp lệ.
- Qwen-plus prompting (76.43%) trong bảng 4.1.B được test trên full Kaggle theo
  Cao et al. 2026 — so sánh trực tiếp với luận văn là hợp lệ.
