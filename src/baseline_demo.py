"""Local desktop demo: python -m src.baseline_demo."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
from tkinter import ttk

from src.baseline_predict import predict_comment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="Download VnCoreNLP resources")
    args = parser.parse_args()
    if args.prepare:
        from src.segment import _ensure_model_files
        _ensure_model_files()
        print("VnCoreNLP resources ready.")
        return

    root = tk.Tk()
    root.title("Đánh giá cảm xúc bình luận")
    root.geometry("740x470")
    root.minsize(620, 440)
    root.configure(bg="#f3f5f9")
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TFrame", background="#f3f5f9")
    style.configure("TLabel", background="#f3f5f9", font=("Segoe UI", 11))
    style.configure("TRadiobutton", background="#f3f5f9", font=("Segoe UI", 10))
    style.configure("TButton", font=("Segoe UI", 11), padding=10)
    frame = ttk.Frame(root, padding=24)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Bình luận này mang cảm xúc gì?", font=("Segoe UI", 19, "bold")).pack(anchor="w")
    ttk.Label(frame, text="Chọn mô hình đã huấn luyện và nhập bình luận tiếng Việt.").pack(anchor="w", pady=(6, 18))
    model = tk.StringVar(value="linear_svc")
    domain = tk.StringVar(value="A")
    options = ttk.Frame(frame)
    options.pack(fill="x")
    controls = []
    for title, value in (("LinearSVC", "linear_svc"), ("Logistic Regression", "logistic_regression")):
        control = ttk.Radiobutton(options, text=title, variable=model, value=value)
        control.pack(side="left", padx=(0, 20))
        controls.append(control)
    domains = ttk.Frame(frame)
    domains.pack(fill="x", pady=12)
    for title, value in (("Train A · Sách Tiki", "A"), ("Train B · Nhà hàng", "B")):
        control = ttk.Radiobutton(domains, text=title, variable=domain, value=value)
        control.pack(side="left", padx=(0, 20))
        controls.append(control)
    comment = ttk.Entry(frame, font=("Segoe UI", 12))
    comment.pack(fill="x", ipady=9, pady=(0, 12))
    controls.append(comment)
    result = tk.StringVar(value="Kết quả sẽ xuất hiện tại đây.")
    detail = tk.StringVar(value="")
    executor = ThreadPoolExecutor(max_workers=1)
    pending = None

    def poll():
        if not pending.done():
            root.after(100, poll)
            return
        try:
            prediction = pending.result()
            result.set(f"{prediction['label_name']} · Nhãn {prediction['label']}")
            name = "LinearSVC" if prediction["model"] == "linear_svc" else "Logistic Regression"
            detail.set(f"{name} · Train {prediction['domain']}")
        except Exception as exc:
            result.set("Chưa thể đánh giá")
            detail.set(str(exc))
        for control in controls:
            control.configure(state="normal")

    def submit(event=None):
        nonlocal pending
        if pending is not None and not pending.done():
            return
        text = comment.get()
        if not text.strip():
            result.set("Vui lòng nhập bình luận.")
            detail.set("")
            return
        for control in controls:
            control.configure(state="disabled")
        result.set("Đang đánh giá…")
        detail.set("Lần đầu có thể lâu hơn do khởi tạo bộ tách từ.")
        pending = executor.submit(predict_comment, text, model.get(), domain.get())
        root.after(100, poll)

    button = ttk.Button(frame, text="Đánh giá bình luận", command=submit)
    button.pack(anchor="w")
    controls.append(button)
    ttk.Label(frame, textvariable=result, font=("Segoe UI", 20, "bold")).pack(anchor="w", pady=(24, 6))
    description = ttk.Label(frame, textvariable=detail, wraplength=670)
    description.pack(anchor="w", fill="x")
    frame.bind("<Configure>", lambda event: description.configure(wraplength=max(200, event.width - 48)))
    comment.bind("<Return>", submit)
    comment.focus_set()

    def close():
        executor.shutdown(wait=False, cancel_futures=True)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    root.mainloop()


if __name__ == "__main__":
    main()
