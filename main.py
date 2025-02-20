import cv2
import numpy as np
import mss
import time
import tkinter as tk
from tkinter import ttk, filedialog
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def load_overlay():
    file_path = filedialog.askopenfilename(filetypes=[("PNG Images", "*.png")])
    if file_path:
        global overlay_img, overlay_alpha, overlay_rgb
        overlay_img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if overlay_img is None:
            print("오류: 선택한 파일을 불러올 수 없습니다!")
            os.system("pause")
            return
        if overlay_img.shape[2] == 4:
            overlay_alpha = overlay_img[:, :, 3] / 255.0
            overlay_rgb = overlay_img[:, :, :3]
        else:
            overlay_alpha = np.ones(overlay_img.shape[:2], dtype=np.float32)
            overlay_rgb = overlay_img

overlay_img = None
overlay_alpha = None
overlay_rgb = None

gui = tk.Tk()
gui.title("FTM")
gui.geometry("400x450")
gui.configure(bg="#2C3E50")

sct = mss.mss()
monitors = sct.monitors[1:] if len(sct.monitors) > 1 else sct.monitors
monitor_options = [f"Monitor {i+1}: {m['width']}x{m['height']}" for i, m in enumerate(monitors)]
monitor_var = tk.StringVar(value=monitor_options[0] if monitor_options else "Monitor 없음")

def update_monitor():
    global monitor
    selected_index = monitor_options.index(monitor_var.get()) if monitor_options else 0
    monitor = monitors[selected_index] if monitor_options else sct.monitors[0]
update_monitor()

scale_factor = 0.5
face_scaleFactor = 1.2
face_minNeighbors = 4
show_fps = tk.BooleanVar(value=True)
exit_key = tk.StringVar(value="27")

overlay_cache = {}

def update_values():
    global scale_factor, face_scaleFactor, face_minNeighbors
    scale_factor = scale_factor_var.get()
    face_scaleFactor = face_scaleFactor_var.get()
    face_minNeighbors = int(face_minNeighbors_var.get())
    scale_label.config(text=f"{scale_factor:.1f}")
    scaleFactor_label.config(text=f"{face_scaleFactor:.2f}")
    minNeighbors_label.config(text=f"{face_minNeighbors}")

style = ttk.Style()
style.configure("TLabel", foreground="white", background="#2C3E50", font=("Arial", 12))
style.configure("TButton", font=("Arial", 12))

scale_factor_var = tk.DoubleVar(value=scale_factor)
face_scaleFactor_var = tk.DoubleVar(value=face_scaleFactor)
face_minNeighbors_var = tk.IntVar(value=face_minNeighbors)

frame = tk.Frame(gui, bg="#2C3E50")
frame.pack(pady=10)

ttk.Label(frame, text="모니터 선택").pack()
monitor_dropdown = ttk.Combobox(frame, values=monitor_options, textvariable=monitor_var, state="readonly")
monitor_dropdown.pack()
monitor_dropdown.bind("<<ComboboxSelected>>", lambda e: update_monitor())

ttk.Button(frame, text="오버레이 변경", command=load_overlay).pack(pady=5)

ttk.Label(frame, text="화면 크기 조절").pack()
scale_slider = ttk.Scale(frame, from_=0.3, to=1.5, variable=scale_factor_var, orient=tk.HORIZONTAL, command=lambda e: update_values())
scale_slider.pack()
scale_label = ttk.Label(frame, text=f"{scale_factor:.1f}")
scale_label.pack()

ttk.Label(frame, text="얼굴 검출 정확도 (scaleFactor)").pack()
scaleFactor_slider = ttk.Scale(frame, from_=1.01, to=1.5, variable=face_scaleFactor_var, orient=tk.HORIZONTAL, command=lambda e: update_values())
scaleFactor_slider.pack()
scaleFactor_label = ttk.Label(frame, text=f"{face_scaleFactor:.2f}")
scaleFactor_label.pack()

ttk.Label(frame, text="얼굴 검출 민감도 (minNeighbors)").pack()
minNeighbors_slider = ttk.Scale(frame, from_=3, to=10, variable=face_minNeighbors_var, orient=tk.HORIZONTAL, command=lambda e: update_values())
minNeighbors_slider.pack()
minNeighbors_label = ttk.Label(frame, text=f"{face_minNeighbors}")
minNeighbors_label.pack()

ttk.Checkbutton(frame, text="FPS 표시", variable=show_fps).pack(pady=5)

ttk.Label(frame, text="종료 키 (ASCII 코드) 입력").pack()
exit_entry = ttk.Entry(frame, textvariable=exit_key)
exit_entry.pack(pady=5)

def run_detection():
    global scale_factor, face_scaleFactor, face_minNeighbors, monitor
    if overlay_rgb is None or overlay_alpha is None:
        print("오류: 오버레이 이미지를 선택하세요!")
        os.system("pause")
        return

    while True:
        start_time = time.time()
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot, dtype=np.uint8)[:, :, :3]
        frame = cv2.resize(frame, (0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_NEAREST)
        frame_h, frame_w, _ = frame.shape
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=face_scaleFactor, minNeighbors=face_minNeighbors, minSize=(30, 30))
        display_frame = frame.copy()

        for (x, y, w, h) in faces:
            overlay_resized = cv2.resize(overlay_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
            overlay_alpha_resized = cv2.resize(overlay_alpha, (w, h), interpolation=cv2.INTER_LINEAR)
            for c in range(3):
                display_frame[y:y+h, x:x+w, c] = (overlay_alpha_resized * overlay_resized[:, :, c] +
                    (1 - overlay_alpha_resized) * display_frame[y:y+h, x:x+w, c]).astype(np.uint8)

        fps = 1.0 / (time.time() - start_time)
        if show_fps.get():
            text_size = cv2.getTextSize(f'FPS: {fps:.2f}', cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = frame_w - text_size[0] - 10
            cv2.putText(display_frame, f'FPS: {fps:.2f}', (text_x, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Overlay Face Detection", display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == int(exit_key.get()):
            break
    cv2.destroyAllWindows()

ttk.Button(gui, text="실행", command=run_detection).pack(pady=10)
gui.mainloop()
