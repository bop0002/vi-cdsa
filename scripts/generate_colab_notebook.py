"""Script to generate notebooks/03_phobert_colab.ipynb with complete self-contained code."""
from pathlib import Path
import nbformat as nbf

nb = nbf.v4.new_notebook()

cells = []

# Cell 1: Markdown Title
cells.append(nbf.v4.new_markdown_cell("""# BÀI TẬP LỚN MÔN HỌC MÁY
## ĐỀ TÀI: PHÂN TÍCH CẢM XÚC TIẾNG VIỆT THEO HƯỚNG CROSS-DOMAIN
### THÀNH VIÊN 3: HUẤN LUYỆN VÀ ĐÁNH GIÁ MÔ HÌNH PHOBERT & PHÂN TÍCH LỖI (ERROR ANALYSIS)

---
- **Mục tiêu:**
  1. Fine-tuning mô hình Transformer tiếng Việt **PhoBERT** (`vinai/phobert-base-v2`) trên 2 miền:
     - **Miền A:** Đánh giá Sách / Thương mại điện tử (Tiki Books).
     - **Miền B:** Đánh giá Nhà hàng / Khách sạn (Foody).
  2. So sánh giữa **Cross-Entropy Loss** và **Focal Loss** trên phân bố dữ liệu tự nhiên (Natural Distribution).
  3. Đánh giá 4 kịch bản thực nghiệm:
     - **In-domain:** $A \\to A$, $B \\to B$
     - **Cross-domain:** $A \\to B$, $B \\to A$
  4. Tính toán mức độ suy giảm hiệu năng (**Performance Drop**) và trực quan hóa **Confusion Matrix**.
  5. Thực hiện **Phân tích lỗi (Error Analysis)** theo các nhóm lỗi quy định tại Mục 16 của tài liệu kế hoạch.
"""))

# Cell 2: Code Check GPU
cells.append(nbf.v4.new_code_cell("""# 1. Kiểm tra cấu hình phần cứng GPU trên Google Colab
!nvidia-smi

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Device: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
"""))

# Cell 3: Code Install Libraries
cells.append(nbf.v4.new_code_cell("""# 2. Cài đặt các thư viện cần thiết
!pip install -q transformers pyvi accelerate seaborn matplotlib pandas scikit-learn
"""))

# Cell 4: Markdown Project Setup
cells.append(nbf.v4.new_markdown_cell("""### 3. Tự động Kéo Mã nguồn và Dữ liệu từ GitHub về Google Colab
Đoạn code dưới đây sẽ tự động clone toàn bộ thư mục dự án (bao gồm toàn bộ mã nguồn `src/` và dữ liệu sạch `.parquet` trong `data/processed/full/`) và chuyển thư mục làm việc vào `vi-cdsa`.
"""))

# Cell 5: Code Mount & Path
cells.append(nbf.v4.new_code_cell("""import os
import sys
from pathlib import Path

# Kiểm tra nếu đang chạy trên Google Colab
if 'google.colab' in sys.modules or os.path.exists('/content'):
    if not os.path.exists('/content/vi-cdsa'):
        print("Đang kéo toàn bộ thư mục dự án (mã nguồn + dữ liệu) từ GitHub...")
        !git clone -b member3-phobert https://github.com/bop0002/vi-cdsa.git /content/vi-cdsa
    
    # Chuyển thư mục làm việc vào vi-cdsa
    %cd /content/vi-cdsa
    if '/content/vi-cdsa' not in sys.path:
        sys.path.insert(0, '/content/vi-cdsa')

print(f"Thư mục làm việc hiện tại: {os.getcwd()}")
!ls -la
"""))

# Cell 6: Markdown Load Data
cells.append(nbf.v4.new_markdown_cell("""### 4. Nạp Dữ Liệu Tự Nhiên (Natural Distribution - Variant `full`)
Tập dữ liệu đã được tiền xử lý và tách từ tiếng Việt chuẩn cho PhoBERT (`text_dl`).
"""))

# Cell 7: Code Load Data
cells.append(nbf.v4.new_code_cell("""import pandas as pd
from src.dataset import load_split

# Nạp 6 tập dữ liệu tự nhiên
splits = {
    "A_train": load_split("A", "train", variant="full"),
    "A_val": load_split("A", "val", variant="full"),
    "A_test": load_split("A", "test", variant="full"),
    "B_train": load_split("B", "train", variant="full"),
    "B_val": load_split("B", "val", variant="full"),
    "B_test": load_split("B", "test", variant="full"),
}

print("=== Thống kê quy mô tập dữ liệu ===")
for name, df in splits.items():
    counts = df["label"].value_counts().to_dict()
    print(f"{name:8s}: {len(df):6d} dòng | Phân bố nhãn: 0 (Neg)={counts.get(0,0)}, 1 (Neu)={counts.get(1,0)}, 2 (Pos)={counts.get(2,0)}")
"""))

# Cell 8: Markdown Dataset & DataLoader
cells.append(nbf.v4.new_markdown_cell("""### 5. Xây dựng DataLoader với Dynamic Padding
PhoBERT sử dụng `DynamicPaddingCollator` để chỉ đệm (pad) độ dài theo mẫu dài nhất trong từng batch riêng lẻ, giúp tiết kiệm tối đa VRAM và tăng tốc độ xử lý hơn 40%.
"""))

# Cell 9: Code DataLoader
cells.append(nbf.v4.new_code_cell("""from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from src.phobert import DEFAULT_MODEL_NAME, DynamicPaddingCollator, ReviewDataset, compute_class_weights

MODEL_NAME = DEFAULT_MODEL_NAME
BATCH_SIZE = 32
MAX_LENGTH = 128

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
collator = DynamicPaddingCollator(tokenizer=tokenizer, max_length=MAX_LENGTH)

# Tính class weights cho Focal Loss từ tập train A và B
weights_A = compute_class_weights(splits["A_train"]["label"])
weights_B = compute_class_weights(splits["B_train"]["label"])

print(f"Class Weights Miền A (Neg, Neu, Pos): {weights_A.tolist()}")
print(f"Class Weights Miền B (Neg, Neu, Pos): {weights_B.tolist()}")
"""))

# Cell 10: Markdown Training
cells.append(nbf.v4.new_markdown_cell("""### 6. Huấn luyện Mô hình PhoBERT
Chúng ta huấn luyện mô hình cho cả 2 miền (Domain A và Domain B) với 2 loại hàm mất mát:
1. **Cross-Entropy Loss (Chuẩn)**
2. **Focal Loss (Tối ưu cho dữ liệu mất cân bằng nhãn - lớp Neutral ~8%)**
"""))

# Cell 11: Code Train Loop
cells.append(nbf.v4.new_code_cell("""from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.phobert import SentimentPhoBERT, ReviewDataset, train_phobert, evaluate_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Thư mục lưu checkpoint và kết quả
Path("models").mkdir(exist_ok=True)
Path("results/metrics").mkdir(parents=True, exist_ok=True)
Path("results/figures").mkdir(parents=True, exist_ok=True)

NUM_EPOCHS = 3
LR = 2e-5

def train_domain_model(domain_code: str, loss_type: str, class_weights=None):
    train_df = splits[f"{domain_code}_train"]
    val_df = splits[f"{domain_code}_val"]

    train_ds = ReviewDataset(train_df["text_dl"], train_df["label"])
    val_ds = ReviewDataset(val_df["text_dl"], val_df["label"])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator)

    model = SentimentPhoBERT(model_name=MODEL_NAME, num_classes=3)
    save_path = f"models/phobert_{domain_code}_{loss_type}.pt"

    print(f"\\n>>> Bắt đầu huấn luyện PhoBERT [{domain_code}] với {loss_type.upper()}...")
    train_phobert(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=NUM_EPOCHS,
        lr=LR,
        loss_type=loss_type,
        class_weights=class_weights if loss_type == "focal" else None,
        save_path=save_path,
        device=device,
        progress_callback=lambda ep, tl, vm: print(f"  Epoch {ep}/{NUM_EPOCHS} - Train Loss: {tl:.4f} | Val Loss: {vm['loss']:.4f} | Val Macro-F1: {vm['macro_f1']:.4f}")
    )
    return save_path

# Huấn luyện các mô hình (có thể comment bớt nếu chỉ muốn chạy thử nghiệm)
ckpt_A_ce = train_domain_model("A", "cross_entropy")
ckpt_A_focal = train_domain_model("A", "focal", weights_A)

ckpt_B_ce = train_domain_model("B", "cross_entropy")
ckpt_B_focal = train_domain_model("B", "focal", weights_B)
"""))

# Cell 12: Markdown Evaluation
cells.append(nbf.v4.new_markdown_cell("""### 7. Thực nghiệm In-Domain & Cross-Domain (4 Kịch bản)
Đánh giá mô hình trên cả 4 kịch bản:
- **A -> A** (In-domain Tiki Sách)
- **A -> B** (Cross-domain: Train Sách -> Test Nhà hàng Foody)
- **B -> B** (In-domain Foody Nhà hàng)
- **B -> A** (Cross-domain: Train Nhà hàng -> Test Sách)
"""))

# Cell 13: Code Evaluation
cells.append(nbf.v4.new_code_cell("""import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torch.utils.data import DataLoader
from src.phobert import LABEL_NAMES, ReviewDataset, SentimentPhoBERT, evaluate_model

test_loaders = {
    "A": DataLoader(ReviewDataset(splits["A_test"]["text_dl"], splits["A_test"]["label"]), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator),
    "B": DataLoader(ReviewDataset(splits["B_test"]["text_dl"], splits["B_test"]["label"]), batch_size=BATCH_SIZE, shuffle=False, collate_fn=collator),
}

all_eval_results = []
test_predictions = {}

experiments = [
    ("A", "cross_entropy", "models/phobert_A_cross_entropy.pt"),
    ("A", "focal", "models/phobert_A_focal.pt"),
    ("B", "cross_entropy", "models/phobert_B_cross_entropy.pt"),
    ("B", "focal", "models/phobert_B_focal.pt"),
]

for train_domain, loss_type, ckpt_path in experiments:
    model = SentimentPhoBERT(model_name=MODEL_NAME, num_classes=3)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()

    for test_domain in ["A", "B"]:
        scenario = f"{train_domain} -> {test_domain}"
        sc_type = "In-domain" if train_domain == test_domain else "Cross-domain"
        
        res = evaluate_model(model=model, dataloader=test_loaders[test_domain], device=device, return_details=True)
        key = f"{scenario}_{loss_type}"
        test_predictions[key] = res["preds"]

        record = {
            "model": "PhoBERT",
            "loss_type": loss_type,
            "scenario": scenario,
            "type": sc_type,
            "macro_f1": res["macro_f1"],
            "accuracy": res["accuracy"],
            "f1_negative": res["f1_neg"],
            "f1_neutral": res["f1_neu"],
            "f1_positive": res["f1_pos"],
            "confusion_matrix": res["confusion_matrix"]
        }
        all_eval_results.append(record)

df_results = pd.DataFrame(all_eval_results)
df_results.to_csv("results/metrics/phobert_results.csv", index=False)
with open("results/metrics/phobert_results.json", "w", encoding="utf-8") as f:
    json.dump(all_eval_results, f, ensure_ascii=False, indent=2)

print("=== BẢNG TỔNG HỢP KẾT QUẢ THỰC NGHIỆM ===")
display(df_results[["scenario", "loss_type", "type", "macro_f1", "accuracy", "f1_neutral"]])
"""))

# Cell 14: Markdown Confusion Matrix & Performance Drop
cells.append(nbf.v4.new_markdown_cell(r"""### 8. Trực quan hóa: Confusion Matrix & Performance Drop ($\Delta$)"""))

# Cell 15: Code Visualization
cells.append(nbf.v4.new_code_cell("""# 1. Vẽ Confusion Matrix cho mô hình Focal Loss
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
focal_results = [r for r in all_eval_results if r["loss_type"] == "focal"]

for ax, r in zip(axes.flatten(), focal_results):
    cm = r["confusion_matrix"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, ax=ax)
    ax.set_title(f"PhoBERT Focal: {r['scenario']} ({r['type']})\\nMacro-F1: {r['macro_f1']:.4f}")
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Nhãn thật")

plt.tight_layout()
plt.savefig("results/figures/phobert_confusion_matrices.png", dpi=300)
plt.show()

# 2. Tính toán và vẽ Performance Drop
def calc_drop(df_sub):
    sub = df_sub.set_index("scenario")
    drop_A_to_B = sub.loc["A -> A", "macro_f1"] - sub.loc["A -> B", "macro_f1"]
    drop_B_to_A = sub.loc["B -> B", "macro_f1"] - sub.loc["B -> A", "macro_f1"]
    return drop_A_to_B, drop_B_to_A

drop_ce_A_B, drop_ce_B_A = calc_drop(df_results[df_results["loss_type"] == "cross_entropy"])
drop_fl_A_B, drop_fl_B_A = calc_drop(df_results[df_results["loss_type"] == "focal"])

print(f"Performance Drop A -> B (Cross-Entropy): {drop_ce_A_B:.4f}")
print(f"Performance Drop A -> B (Focal Loss)   : {drop_fl_A_B:.4f}")
print(f"Performance Drop B -> A (Cross-Entropy): {drop_ce_B_A:.4f}")
print(f"Performance Drop B -> A (Focal Loss)   : {drop_fl_B_A:.4f}")
"""))

# Cell 16: Markdown Error Analysis
cells.append(nbf.v4.new_markdown_cell("""### 9. Phân tích lỗi (Error Analysis - Mục 16 của Đề tài)
Tự động trích xuất các câu đoán sai tiêu biểu ở kịch bản Cross-Domain và phân loại theo các nhóm:
- *Domain-specific vocabulary* (Từ vựng đặc thù của miền)
- *Phủ định phức tạp* (Negation)
- *Mixed Sentiment* (Vừa khen vừa chê)
- *Sarcasm* (Mỉa mai)
- *Review quá ngắn*
"""))

# Cell 17: Code Error Analysis
cells.append(nbf.v4.new_code_cell("""from src.error_analysis import extract_error_cases, format_error_table_markdown

# Lấy các trường hợp dự đoán sai của PhoBERT Focal ở kịch bản A -> B
df_test_B = splits["B_test"].copy()
df_test_B["pred"] = test_predictions["A -> B_focal"]

errors_A_to_B = extract_error_cases(
    df=df_test_B,
    source_domain="A",
    target_domain="B",
    model_name="PhoBERT (Focal Loss)",
    max_cases_per_type=2,
    total_cases=10
)

# Lấy các trường hợp dự đoán sai ở kịch bản B -> A
df_test_A = splits["A_test"].copy()
df_test_A["pred"] = test_predictions["B -> A_focal"]

errors_B_to_A = extract_error_cases(
    df=df_test_A,
    source_domain="B",
    target_domain="A",
    model_name="PhoBERT (Focal Loss)",
    max_cases_per_type=2,
    total_cases=10
)

df_all_errors = pd.concat([errors_A_to_B, errors_B_to_A], ignore_index=True)
df_all_errors.to_csv("results/metrics/error_analysis.csv", index=False)

md_table = format_error_table_markdown(df_all_errors)
with open("results/metrics/error_analysis.md", "w", encoding="utf-8") as f:
    f.write("# ERROR ANALYSIS REPORT\\n\\n")
    f.write(md_table)
    f.write("\\n")

print("=== BẢNG PHÂN TÍCH LỖI MẪU CHO CHƯƠNG 5 BÁO CÁO ===")
print(md_table)
"""))

# Cell 18: Download results
cells.append(nbf.v4.new_markdown_cell("""### 10. Nén và Tải Toàn bộ Kết quả về Máy tính
Chạy ô bên dưới để tự động đóng gói thư mục `results/` (bao gồm các bảng số liệu CSV, file JSON và biểu đồ Confusion Matrix PNG) thành file ZIP và tải về máy tính của bạn.
"""))

cells.append(nbf.v4.new_code_cell("""# Nén toàn bộ kết quả thực nghiệm và tự động tải về
!zip -r /content/results_phobert.zip results/
try:
    from google.colab import files
    files.download('/content/results_phobert.zip')
    print("✅ Đã khởi chạy tải file results_phobert.zip về máy!")
except Exception as e:
    print(f"File zip đã lưu tại: /content/results_phobert.zip (Lỗi Colab files: {e})")
"""))

nb.cells = cells

out_path = Path("notebooks/03_phobert_colab.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    nbf.write(nb, f)

print(f"Created notebook at {out_path}")
