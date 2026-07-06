"""
多模态情感分析系统 - Web 界面
用法: python app.py
"""
import os, sys, time, cv2

if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path: sys.path.insert(0, _PROJ)

import gradio as gr
import pandas as pd
import numpy as np
from PIL import Image

from main.pipeline import EmotionAnalysisPipeline

# 人脸检测模块（自带，不依赖 zz/）
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_FD_DIR = os.path.join(_APP_DIR, "face_detect")
if _FD_DIR not in sys.path: sys.path.insert(0, _FD_DIR)
from face_detect import find_face, get_face, draw_box, init_detector
from cv.vision_toolkit import predict_emotion_from_array

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
_USER_DATA = os.environ.get("EMOTION_USER_DATA", _APP_DIR)
FACES_DIR = os.path.join(_USER_DATA, "faces")
HISTORY_CSV = os.path.join(_USER_DATA, "history.csv")
os.makedirs(FACES_DIR, exist_ok=True)

EMOTIONS_EN = ["angry","disgust","fear","happy","sad","surprise","neutral"]
SENTIMENTS_EN = ["negative","neutral","positive"]
FUSION_EN = ["happy","angry","sad","surprise","sarcasm"]
ALPHA_COL = "门控alpha"

CSV_COLS = (["样本ID","图片路径","文本内容","模式","状态",
             "CV预测","CV置信度"] + [f"CV_{e}" for e in EMOTIONS_EN]
            + ["NLP预测","NLP置信度"] + [f"NLP_{e}" for e in SENTIMENTS_EN]
            + ["融合预测","融合置信度","情感强度", ALPHA_COL]
            + [f"融合_{e}" for e in FUSION_EN])

DISPLAY_COLS = ["样本ID","图片路径","文本内容","CV预测","CV置信度",
                "NLP预测","NLP置信度","融合预测","融合置信度","情感强度", ALPHA_COL]

# 抑制第三方库的进度条输出
os.environ.setdefault("TQDM_DISABLE", "1")
import transformers
transformers.logging.set_verbosity_error()

from main.pipeline import FUSION_REGISTRY, FUSION_NAMES

pipeline = EmotionAnalysisPipeline(fusion_strategy="gated")
pipeline.load_all()

# 切换融合策略
def switch_fusion(strategy):
    pipeline.load_fusion(strategy=strategy)
    return f"### 当前策略: {FUSION_NAMES[strategy]}"

if not os.path.exists(HISTORY_CSV) or os.path.getsize(HISTORY_CSV) < 10:
    pd.DataFrame(columns=CSV_COLS).to_csv(HISTORY_CSV, index=False, encoding="utf-8-sig")


def _read_csv():
    try:
        df = pd.read_csv(HISTORY_CSV, encoding="utf-8-sig")
        return df if len(df.columns) > 0 else pd.DataFrame(columns=CSV_COLS)
    except:
        return pd.DataFrame(columns=CSV_COLS)

def _str(v, default=""):
    if pd.isna(v) or v is None: return default
    return str(v)

def _pct(s):
    try: return float(str(s).replace("%",""))/100
    except: return 0


# ==================== 分析 ====================
def analyze(image, text):
    text = (text or "").strip()
    img_rel, img_abs = "", ""
    if image is not None:
        fname = f"{int(time.time()*1000)%100000}.jpg"
        img_abs = os.path.join(FACES_DIR, fname)
        Image.fromarray(image).save(img_abs)
        img_rel = f"faces/{fname}"

    cv_r = _cv(img_abs)
    nlp_r = _nlp(text)
    fusion_r = _fusion(img_abs, text)

    row = {c: "" for c in CSV_COLS}
    row["样本ID"] = int(time.time()*1000)%100000
    row["图片路径"] = img_rel
    row["文本内容"] = text[:100]
    row["模式"] = "auto"
    row["状态"] = "ok"

    if cv_r and "error" not in cv_r:
        row["CV预测"] = _str(cv_r["label_name"])
        row["CV置信度"] = f"{cv_r['confidence']*100:.1f}%"
        for i,e in enumerate(EMOTIONS_EN): row[f"CV_{e}"] = f"{cv_r['probabilities'][i]*100:.1f}%"
    if nlp_r and "error" not in nlp_r:
        row["NLP预测"] = _str(nlp_r["label_name"])
        row["NLP置信度"] = f"{nlp_r['confidence']*100:.1f}%"
        p = nlp_r["probabilities"]
        for i,e in enumerate(SENTIMENTS_EN): row[f"NLP_{e}"] = f"{list(p.values())[i]*100:.1f}%" if isinstance(p,dict) else f"{p[i]*100:.1f}%"
    if fusion_r and "error" not in fusion_r:
        row["融合预测"] = _str(fusion_r["label_name"])
        row["融合置信度"] = f"{fusion_r['confidence']*100:.1f}%"
        row["情感强度"] = str(fusion_r["intensity"])
        row[ALPHA_COL] = f"{fusion_r['alpha']:.3f}"
        p = fusion_r["probabilities"]
        for i,e in enumerate(FUSION_EN): row[f"融合_{e}"] = f"{list(p.values())[i]*100:.1f}%" if isinstance(p,dict) else f"{p[i]*100:.1f}%"

    df = _read_csv()
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(HISTORY_CSV, index=False, encoding="utf-8-sig")

    return (_bars(cv_r, EMOTIONS_EN), _bars(nlp_r, SENTIMENTS_EN),
            _bars(fusion_r, FUSION_EN), _summary(fusion_r,cv_r,nlp_r), load_history())

def _cv(p):
    if not p: return None
    try: return pipeline.predict_cv_only(p)
    except: return None
def _nlp(t):
    if not t: return None
    try: return pipeline.predict_nlp_only(t)
    except: return None
def _fusion(p,t):
    if not p or not t: return None
    try: return pipeline.predict_fusion(p,t)
    except: return None

def _bars(r, labels):
    if r is None: return {l:0 for l in labels}
    if "error" in r: return {}
    p = r.get("probabilities",{})
    if isinstance(p,list): return {labels[i]:p[i] for i in range(min(len(labels),len(p)))}
    return p

def _summary(f,cv,nlp):
    if f and "error" not in f:
        bar = "█" * f["intensity"] + "░" * (10 - f["intensity"])
        lines = [
            f"### Fusion: **{f['label_name']}** ({f['confidence']*100:.1f}%)  "
            f"强度: {f['intensity']}/10 {bar}",
            f"CV→{cv['label_name']}({cv['confidence']*100:.0f}%) | "
            f"NLP→{nlp['label_name']}({nlp['confidence']*100:.0f}%) | "
            f"α={f['alpha']:.3f}",
        ]
        # 智能冲突判定：脸和文字真的矛盾才报警
        cv_pos = cv["label_name"] in ("happy", "surprise")
        cv_neg = cv["label_name"] in ("angry", "sad", "fear", "disgust")
        nlp_pos = nlp["label_name"] == "positive"
        nlp_neg = nlp["label_name"] == "negative"
        if (cv_pos and nlp_neg) or (cv_neg and nlp_pos):
            lines.append(f"\n⚠️ 检测到模态冲突 (冲突度: {f['conflict_score']:.2f})，"
                         f"可能与讽刺或反差表达有关")
        return "\n\n".join(lines)
    if cv and "error" not in cv: return f"### CV: **{cv['label_name']}** ({cv['confidence']*100:.0f}%)"
    if nlp and "error" not in nlp: return f"### NLP: **{nlp['label_name']}** ({nlp['confidence']*100:.0f}%)"
    return "请上传图片或输入文本"


# ==================== 历史 ====================
def load_history():
    df = _read_csv()
    if len(df)==0: return gr.update(value=pd.DataFrame({"提示":["暂无"]}))
    cols = [c for c in DISPLAY_COLS if c in df.columns]
    return gr.update(value=df[cols].tail(50).iloc[::-1])

def clear_history():
    pd.DataFrame(columns=CSV_COLS).to_csv(HISTORY_CSV, index=False, encoding="utf-8-sig")
    return gr.update(value=pd.DataFrame({"提示":["已清空"]}))

def _csv_bars(row, prefix, labels):
    d = {}
    for l in labels:
        v = row.get(f"{prefix}_{l}","")
        if pd.notna(v) and str(v) not in ("","nan"): d[l] = _pct(v)
    return d


def _camera_backends():
    if sys.platform.startswith("win"):
        return [getattr(cv2, "CAP_DSHOW", None), getattr(cv2, "CAP_MSMF", None), getattr(cv2, "CAP_ANY", None)]
    if sys.platform == "darwin":
        return [getattr(cv2, "CAP_AVFOUNDATION", None), getattr(cv2, "CAP_ANY", None)]
    return [getattr(cv2, "CAP_V4L2", None), getattr(cv2, "CAP_ANY", None)]


def _open_camera(index=0):
    for backend in _camera_backends():
        try:
            cam = cv2.VideoCapture(index) if backend is None else cv2.VideoCapture(index, backend)
            if cam.isOpened():
                return cam
            cam.release()
        except Exception:
            continue
    return None


# ==================== 页面 ====================
with gr.Blocks(title="多模态情感分析系统") as demo:
    gr.Markdown("# 多模态情感分析系统")
    gr.Markdown("CV (ResNet18) + NLP (RoBERTa) → Gated Fusion")

    with gr.Tabs():
        with gr.TabItem("实时分析"):
            with gr.Row(equal_height=True):
                with gr.Column(scale=1):
                    img_input = gr.Image(label="上传图片", type="numpy", height=260)
                    btn = gr.Button("开始分析", variant="primary", size="lg", scale=1)
                with gr.Column(scale=1):
                    text_input = gr.Textbox(label="输入文本", placeholder="太搞笑了哈哈哈！", lines=3)
                    fusion_sel = gr.Dropdown(
                        choices=[(FUSION_NAMES[k], k) for k in ["early","late","gated"]],
                        value="gated", label="融合策略",
                        interactive=True, scale=1)
            fusion_status = gr.Markdown("### 当前策略: 门控注意力")
            fusion_sel.change(fn=switch_fusion, inputs=[fusion_sel], outputs=[fusion_status])
            gr.Markdown("---")
            with gr.Row():
                with gr.Column(): cv_plot = gr.Label(label="CV (7 表情)", num_top_classes=7)
                with gr.Column(): nlp_plot = gr.Label(label="NLP (3 倾向)", num_top_classes=3)
                with gr.Column(): fusion_plot = gr.Label(label="Fusion (5 情感)", num_top_classes=5)
            fusion_text = gr.Markdown("")

        with gr.TabItem("摄像头"):
            gr.Markdown("点「开始」→ 摄像头实时分析 → 绿框+表情 → 点「停止」释放")
            with gr.Row():
                cam_start = gr.Button("开始", variant="primary", size="lg")
                cam_stop = gr.Button("停止", variant="stop", size="lg")
            cam_output = gr.Image(label="实时画面", height=400)
            with gr.Row():
                cam_bars = gr.Label(label="全部表情概率", num_top_classes=7)
            cam_status = gr.Markdown("")
            _active = gr.State(False)

            # 全局摄像头句柄 + 缓存
            _cam = None
            _last_result = None
            _last_infer = 0

            def _cam_tick(active):
                global _cam, _last_result, _last_infer
                if not active:
                    return None, {}, "已停止"

                try:
                    if _cam is None:
                        _cam = _open_camera(0)
                        if _cam is not None:
                            _cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                            _cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                            _cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                            _cam.set(cv2.CAP_PROP_FPS, 30)
                        _last_result = None
                        _last_infer = 0
                    if _cam is None or not _cam.isOpened():
                        if _cam is not None:
                            _cam.release()
                            _cam = None
                        return None, {}, "摄像头打开失败"

                    ret, frame = _cam.read()
                    if not ret:
                        return None, {}, "读取失败"

                    frame = cv2.flip(frame, 1)

                    # 每 2 秒推理一次（高清预览不卡）
                    now = time.time()
                    if now - _last_infer > 2.0:
                        _last_infer = now
                        h, w = frame.shape[:2]
                        init_detector(w, h)
                        face = find_face(frame)
                        if face is not None:
                            face_crop = get_face(frame)
                            r = predict_emotion_from_array(face_crop, pipeline.cv_classifier)
                            label = f"{r['label_name']} {r['confidence']*100:.0f}%"
                            _last_result = (face, label, r)
                        else:
                            _last_result = None

                    bars = {}
                    status = "未检测到人脸"
                    if _last_result is not None:
                        face, label, r = _last_result
                        frame = draw_box(frame, face, label, (0, 255, 0))
                        status = f"**{label}**"
                        bars = {EMOTIONS_EN[i]: r['probabilities'][i] for i in range(7)}

                    # 缩小显示尺寸减少网络传输
                    display = cv2.resize(frame, (640, 360))
                    return cv2.cvtColor(display, cv2.COLOR_BGR2RGB), bars, status
                except Exception as e:
                    return None, {}, f"错误: {e}"

            def _start_cam():
                global _cam
                if _cam is not None:
                    _cam.release()
                    _cam = None
                return True

            def _stop_cam():
                global _cam
                if _cam is not None:
                    _cam.release()
                    _cam = None
                return None, {}, "已停止"

            cam_start.click(fn=_start_cam, outputs=[_active])
            cam_stop.click(fn=_stop_cam, outputs=[cam_output, cam_bars, cam_status])

            t = gr.Timer(0.1)
            t.tick(fn=_cam_tick, inputs=[_active], outputs=[cam_output, cam_bars, cam_status])

        with gr.TabItem("历史记录"):
            with gr.Row():
                refresh_btn = gr.Button("刷新", size="sm")
                del_sel_btn = gr.Button("删除选中", variant="stop", size="sm")
                clear_btn = gr.Button("清空全部", size="sm")
            del_msg = gr.Markdown("")
            selected_id = gr.State(None)
            history_table = gr.Dataframe(label="点击行查看详情", interactive=False, wrap=True)
            gr.Markdown("---")
            with gr.Row():
                hist_img = gr.Image(label="图片", height=200)
                hist_summary = gr.Markdown("")
            with gr.Row():
                hist_cv = gr.Label(label="CV", num_top_classes=7)
                hist_nlp = gr.Label(label="NLP", num_top_classes=3)
                hist_fusion = gr.Label(label="Fusion", num_top_classes=5)

            def delete_selected(sid):
                if sid is None: return load_history(), None, "请先点击表格中的行"
                df = _read_csv()
                before = len(df)
                df = df[df["样本ID"].astype(str) != str(sid)]
                if len(df) == before: return load_history(), None, "删除失败"
                df.to_csv(HISTORY_CSV, index=False, encoding="utf-8-sig")
                return load_history(), None, f"已删除样本#{sid} (剩余{len(df)}条)"

            def on_select(evt: gr.SelectData):
                df = _read_csv()
                if len(df)==0: return (None,{},{},{},None,"")
                idx = len(df)-1-evt.index[0]
                if idx<0 or idx>=len(df): return (None,{},{},{},None,"")
                row = df.iloc[idx]
                sid = _str(row.get("样本ID",""))
                img = None
                p = _str(row.get("图片路径",""))
                if p:
                    ap = os.path.join(_USER_DATA, p)
                    if os.path.exists(ap): img = ap
                cv = _csv_bars(row,"CV",EMOTIONS_EN)
                nlp = _csv_bars(row,"NLP",SENTIMENTS_EN)
                fusion = _csv_bars(row,"融合",FUSION_EN)
                txt = _str(row.get("文本内容",""))
                s = f"**#{sid}** | {txt[:60]}"
                if cv: s += f"\n\nCV: **{_str(row.get('CV预测','?'))}** ({_str(row.get('CV置信度',''))})"
                if nlp: s += f"\nNLP: **{_str(row.get('NLP预测','?'))}** ({_str(row.get('NLP置信度',''))})"
                if fusion: s += f"\nFusion: **{_str(row.get('融合预测','?'))}** ({_str(row.get('融合置信度',''))}) α={_str(row.get(ALPHA_COL,''))}"
                return (img, cv, nlp, fusion, sid, s)

            refresh_btn.click(fn=load_history, outputs=[history_table])
            clear_btn.click(fn=clear_history, outputs=[history_table])
            del_sel_btn.click(fn=delete_selected, inputs=[selected_id], outputs=[history_table, selected_id, del_msg])
            history_table.select(fn=on_select, outputs=[hist_img, hist_cv, hist_nlp, hist_fusion, selected_id, hist_summary])

    btn.click(fn=analyze, inputs=[img_input, text_input],
              outputs=[cv_plot, nlp_plot, fusion_plot, fusion_text, history_table])

    # 历史自动刷新（2秒一次）+ 页面加载时触发
    hist_timer = gr.Timer(2)
    hist_timer.tick(fn=load_history, outputs=[history_table])
    demo.load(fn=load_history, outputs=[history_table])


if __name__ == "__main__":
    print("  ─────────────────────────────────────")
    print("  浏览器将自动打开，或访问 http://127.0.0.1:7860")
    print()
    demo.launch(share=False, inbrowser=True, allowed_paths=[FACES_DIR])
