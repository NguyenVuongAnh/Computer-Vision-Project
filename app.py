import base64
import html
import io
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

from model import Predictor, FeatureMapExtractor, GradCAM
from style import inject_style, THEMES

NUM_CLASSES = 2
IMG_SIZE = 512
WEIGHTS_PATH = "Resnet50_best/model.weights.h5"
CLASS_NAMES = {
    0: "Không bị tiểu đường",
    1: "Bị tiểu đường"
}

st.set_page_config(
    page_title="RetinaScan · Phân loại mức độ bệnh võng mạc tiểu đường",
    page_icon="🩺",
    layout="wide",
)

if "theme" not in st.session_state:
    st.session_state.theme = "light"

# Resolve the toggle click before building any theme-dependent output below, so
# the new theme applies on this same rerun. st.button() already triggers its own
# natural rerun on click — calling st.rerun() here would additionally reset
# later widgets (e.g. the file uploader) on that pass, so it's intentionally
# not used.
with st.container(key="theme_toggle"):
    if st.button("🌓", key="theme_toggle_btn"):
        st.session_state.theme = "dark" if st.session_state.theme == "light" else "light"

theme_colors = THEMES[st.session_state.theme]
plt.rcParams.update({
    "figure.facecolor": theme_colors["panel"],
    "axes.facecolor": theme_colors["panel"],
    "axes.edgecolor": theme_colors["panel-border"],
    "axes.labelcolor": theme_colors["ink"],
    "text.color": theme_colors["ink"],
    "xtick.color": theme_colors["muted"],
    "ytick.color": theme_colors["muted"],
})

inject_style(st.session_state.theme)

# fixed header, hides on scroll down
st.markdown(
    """
    <div id="app-header">
        <div class="brand">Retina<span>Scan</span></div>
        <div class="nav">
            <a href="#demo-section">Demo</a>
            <a href="#workflow-section">Quy trình</a>
            <a href="#about-section">Giới thiệu</a>
        </div>
    </div>
    <div class="header-spacer"></div>
    """,
    unsafe_allow_html=True,
)

# injected via an iframe so it can reach the parent document (st.markdown scripts don't run)
st.iframe(
    """
    <script>
    (function () {
        function init() {
            var doc = window.parent.document;
            var header = doc.getElementById('app-header');
            if (!header) { return; }
            var container = doc.querySelector('section.main')
                          || doc.querySelector('[data-testid="stAppViewContainer"]')
                          || doc.scrollingElement
                          || doc.documentElement;
            var lastY = 0;
            function onScroll() {
                var y = container.scrollTop || doc.documentElement.scrollTop || 0;
                if (y > lastY && y > 90) {
                    header.classList.add('header-hidden');
                } else {
                    header.classList.remove('header-hidden');
                }
                lastY = y;
            }
            container.addEventListener('scroll', onScroll, { passive: true });
            window.parent.addEventListener('scroll', onScroll, { passive: true });
        }
        setTimeout(init, 300);
    })();
    </script>
    """,
    height=1,
)

st.markdown('<div id="demo-section"></div>', unsafe_allow_html=True)
st.markdown(
    """
    <div class="hero-eyebrow">FUNDUS IMAGE ANALYSIS — RESNET50</div>
    <div class="hero-title">Phân loại mức độ bệnh võng mạc tiểu đường</div>
    <div class="hero-sub">
        Tải lên ảnh đáy mắt (fundus image) để mô hình ResNet50 dự đoán mức độ
        bệnh võng mạc do tiểu đường, đồng thời trực quan hoá vùng ảnh mô hình
        chú ý (Grad-CAM) và các đặc trưng nội bộ được trích xuất qua từng lớp.
    </div>
    <br>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_model():
    inputs = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    base_model = ResNet50(include_top=False, weights=None, input_tensor=inputs)
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)
    x = Dense(2048, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(1, activation="sigmoid", name="final_output")(x)
    model = Model(inputs, outputs)
    model.load_weights(WEIGHTS_PATH)
    return model


@st.cache_resource(show_spinner=False)
def load_tools(_model):
    return Predictor(_model), FeatureMapExtractor(_model), GradCAM(_model)


def conv_layer_names(model):
    return [layer.name for layer in model.layers if len(layer.output.shape) == 4]


def default_index(options, preferred, fallback=0):
    return options.index(preferred) if preferred in options else fallback


try:
    model = load_model()
except Exception as e:
    model = None
    model_error = str(e)

with st.container(key="workstation"):
    if model is None:
        st.error(
            f"Không thể tải trọng số mô hình từ `{WEIGHTS_PATH}`.\n\n"
            f"Kiểm tra file có tồn tại đúng đường dẫn hay không.\n\nChi tiết: {model_error}"
        )
        st.markdown(
            """
            <div class="scan-panel">
                <div class="panel-label">Ảnh tải lên</div>
                <div class="placeholder-box">
                    <div class="placeholder-text">
                        Chưa có mô hình khả dụng — không thể chạy dự đoán cho đến khi
                        tìm thấy file trọng số.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        predictor, extractor, gradcam = load_tools(model)
        conv_layers = conv_layer_names(model)

        st.markdown('<div class="panel-label">Ảnh tải lên</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Tải lên ảnh đáy mắt", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
        )

        if uploaded_file is None:
            st.markdown(
                """
                <div class="scan-panel">
                    <div class="placeholder-box">
                        <div class="placeholder-text">
                            Tải lên một ảnh đáy mắt (fundus image) ở trên để bắt đầu phân tích.
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            original_image = Image.open(uploaded_file).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            image_array = np.array(original_image, dtype=np.float32)
            model_input = preprocess_input(image_array.copy())

            result = predictor.predict_class(model_input)
            predicted_class = result["class_id"]
            probs = result["probabilities"]

            tab_predict, tab_features, tab_gradcam, tab_summary = st.tabs(
                ["Dự đoán", "Bản đồ đặc trưng", "Grad-CAM", "Tổng quan mô hình"]
            )

            with tab_predict:
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown('<div class="panel-label">Ảnh đầu vào</div>', unsafe_allow_html=True)
                    st.image(original_image, width="stretch")
                with col2:
                    st.markdown('<div class="panel-label">Kết quả dự đoán</div>', unsafe_allow_html=True)

                    has_dr = predicted_class != 0
                    verdict_text = "CÓ DẤU HIỆU BỆNH" if has_dr else "KHÔNG CÓ DẤU HIỆU BỆNH"
                    verdict_icon = "⚠️" if has_dr else "✅"
                    verdict_color = "var(--danger)" if has_dr else "var(--accent-teal)"

                    st.markdown(
                        f"""
                        <div class="verdict-card" style="border-left-color:{verdict_color};">
                            <div class="verdict-line" style="color:{verdict_color};">
                                {verdict_icon} {verdict_text}
                            </div>
                            <div class="verdict-sub">Độ tin cậy: <b>{result['confidence']:.2%}</b></div>
                            <div class="verdict-sub">
                                Mức độ dự đoán: <b style="color:{verdict_color}">{CLASS_NAMES[predicted_class]}</b>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            with tab_features:
                if "selected_layer" not in st.session_state or st.session_state.selected_layer not in conv_layers:
                    st.session_state.selected_layer = conv_layers[default_index(conv_layers, "conv2_block1_out")]

                def select_layer(lname):
                    st.session_state.selected_layer = lname

                # bound via key= so this selectbox's value persists across reruns
                layer_name = st.selectbox("Layer", conv_layers, key="selected_layer")

                feature_map = extractor.extract_layer(model_input, layer_name)
                num_channels = feature_map.shape[-1]

                if "selected_channel" not in st.session_state or st.session_state.selected_channel > num_channels - 1:
                    st.session_state.selected_channel = min(10, num_channels - 1)

                def select_channel(ch):
                    st.session_state.selected_channel = ch

                def step_channel(delta):
                    idx = st.session_state.selected_channel
                    st.session_state.selected_channel = max(0, min(num_channels - 1, idx + delta))

                # bound via key= so the prev/next arrows and filmstrip clicks
                # below can update this slider's value for the next rerun
                channel = st.slider("Channel", 0, num_channels - 1, key="selected_channel")

                st.markdown(
                    f'<div class="filmstrip-caption">{html.escape(layer_name)} '
                    f'&nbsp;·&nbsp; channel {channel}</div>',
                    unsafe_allow_html=True,
                )

                # lightbox-style viewer: prev arrow · big image · next arrow
                prev_col, main_col, next_col = st.columns([1, 10, 1])
                with prev_col:
                    st.markdown("<div style='height:80px'></div>", unsafe_allow_html=True)
                    st.button(
                        "‹", key="prev_channel", on_click=step_channel, args=(-1,),
                        disabled=channel == 0, width="stretch",
                    )
                with main_col:
                    with st.container(key="main_feature_plot"):
                        fig, ax = plt.subplots(figsize=(4.7, 4.7))
                        im = ax.imshow(feature_map[:, :, channel], cmap="viridis")
                        ax.axis("off")
                        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
                        # st.pyplot defaults to width="stretch", which forces the
                        # figure to fill the column regardless of figsize — that's
                        # what was blowing this up to a huge, blocky/pixelated image
                        st.pyplot(fig, width="content")
                with next_col:
                    st.markdown("<div style='height:80px'></div>", unsafe_allow_html=True)
                    st.button(
                        "›", key="next_channel", on_click=step_channel, args=(1,),
                        disabled=channel == num_channels - 1, width="stretch",
                    )

                def fig_to_data_uri(fig):
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", bbox_inches="tight", dpi=130)
                    plt.close(fig)
                    buf.seek(0)
                    return "data:image/png;base64," + base64.b64encode(buf.read()).decode()

                # filmstrip: 4 before, current (highlighted), 4 after — one row,
                # browsing nearby channels of the current layer (mirrors the
                # Channel slider above). Clickable thumbnails are rendered as a
                # button whose own background-image IS the plot, so the click
                # target and the picture are the same element (an overlay
                # button on a separate image is unreliable — Streamlit's own
                # image hover layer can sit above it and swallow the click).
                window = range(max(0, channel - 4), min(num_channels, channel + 5))
                thumb_styles = []
                # wrapped in a keyed container so the CSS below (which makes the
                # columns shrink-wrap around each fixed-size thumbnail instead of
                # stretching to fill the row) only applies to this filmstrip
                with st.container(key="filmstrip_row"):
                    strip_cols = st.columns(len(window))
                    for col, ch in zip(strip_cols, window):
                        is_current = ch == channel
                        fig, ax = plt.subplots(figsize=(0.9, 0.9))
                        ax.imshow(feature_map[:, :, ch], cmap="viridis")
                        ax.axis("off")
                        uri = fig_to_data_uri(fig)
                        with col:
                            if is_current:
                                with st.container(key="thumb_selected"):
                                    # same aspect-ratio + background-image technique as
                                    # the clickable thumbnails below, so it matches
                                    # their size exactly instead of a plain <img> that
                                    # rendered smaller than its box
                                    st.markdown(
                                        f'<div style="width:100%;aspect-ratio:1/1;'
                                        f'background-image:url(\'{uri}\');'
                                        f'background-size:cover;background-position:center;'
                                        f'border-radius:4px;"></div>',
                                        unsafe_allow_html=True,
                                    )
                            else:
                                btn_key = f"pick_{layer_name}_{ch}"
                                with st.container(key=f"thumb_dim_{layer_name}_{ch}"):
                                    st.button(
                                        " ", key=btn_key, on_click=select_channel, args=(ch,),
                                        width="stretch",
                                    )
                                thumb_styles.append(f"""
                                    .st-key-{btn_key} button {{
                                        background-image:url('{uri}') !important;
                                    }}
                                """)

                st.markdown(f"<style>{''.join(thumb_styles)}</style>", unsafe_allow_html=True)

            with tab_gradcam:
                layer_name_cam = st.selectbox(
                    "Layer",
                    conv_layers,
                    index=default_index(conv_layers, "conv5_block3_out", len(conv_layers) - 1),
                    key="gradcam_layer",
                )
                target_class = st.selectbox(
                    "Lớp mục tiêu",
                    options=list(CLASS_NAMES.keys()),
                    format_func=lambda i: CLASS_NAMES[i],
                    index=predicted_class,
                )
                heatmap = gradcam.generate(model_input, layer_name_cam, target_class=target_class)

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(image_array.astype("uint8"))
                axes[0].set_title("Ảnh gốc")
                axes[0].axis("off")

                axes[1].imshow(heatmap, cmap="jet")
                axes[1].set_title("Grad-CAM Heatmap")
                axes[1].axis("off")

                axes[2].imshow(image_array.astype("uint8"))
                axes[2].imshow(heatmap, cmap="jet", alpha=0.4)
                axes[2].set_title(f"Overlay - {CLASS_NAMES[target_class]}")
                axes[2].axis("off")

                st.pyplot(fig)

            with tab_summary:
                rows = "".join(
                    f"<tr><td>{info['index']}</td>"
                    f"<td>{html.escape(info['name'])}</td>"
                    f"<td>{html.escape(info['type'])}</td>"
                    f"<td>{html.escape(str(info['output_shape']))}</td></tr>"
                    for info in extractor.layer_info()
                )
                st.markdown(
                    f"""
                    <div class="model-summary-table-wrap">
                        <table class="model-summary-table">
                            <thead>
                                <tr><th>Index</th><th>Layer Name</th><th>Type</th><th>Output Shape</th></tr>
                            </thead>
                            <tbody>{rows}</tbody>
                        </table>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

st.markdown("<div style='height:64px'></div>", unsafe_allow_html=True)

# workflow section (header's "Quy trình" link scrolls here)
st.markdown('<div id="workflow-section"></div>', unsafe_allow_html=True)
st.markdown('<div class="hero-eyebrow">PIPELINE</div>', unsafe_allow_html=True)
st.markdown("## Quy trình hoạt động")

WORKFLOW_STEPS = [
    ("1", "Tải ảnh lên", "Người dùng tải lên ảnh đáy mắt (fundus image) định dạng JPG/PNG."),
    ("2", "Tiền xử lý", "Ảnh được resize về 256×256 và chuẩn hoá theo chuẩn ResNet50 (preprocess_input)."),
    ("3", "Mô hình ResNet50", "Ảnh đi qua mạng CNN đã fine-tune để trích xuất đặc trưng và tính xác suất từng mức độ bệnh."),
    ("4", "Kết quả & giải thích", "Hiển thị mức độ dự đoán, biểu đồ xác suất, Grad-CAM và feature maps để diễn giải quyết định của mô hình."),
]

workflow_cols = st.columns([4, 1, 4, 1, 4, 1, 4])
step_cols, arrow_cols = workflow_cols[0::2], workflow_cols[1::2]

for col, (num, title, desc) in zip(step_cols, WORKFLOW_STEPS):
    with col:
        st.markdown(
            f"""
            <div class="workflow-step">
                <div class="workflow-num">{num}</div>
                <div class="workflow-title">{title}</div>
                <div class="workflow-desc">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
for col in arrow_cols:
    with col:
        st.markdown('<div class="workflow-arrow">→</div>', unsafe_allow_html=True)

st.markdown("<div style='height:64px'></div>", unsafe_allow_html=True)

# about section (header's "Giới thiệu" link scrolls here)
st.markdown('<div id="about-section" class="about-section"></div>', unsafe_allow_html=True)
st.markdown("## Về dự án")
st.markdown(
    """
    **Mục tiêu:** Hỗ trợ phát hiện sớm bệnh võng mạc do tiểu đường (diabetic
    retinopathy) từ ảnh đáy mắt, bằng cách phân loại mức độ bệnh (No DR, Mild,
    Moderate, Severe, Proliferative DR) sử dụng mạng **ResNet50** được huấn
    luyện lại (fine-tuned) trên bộ dữ liệu chuyên biệt. Bên cạnh dự đoán, ứng
    dụng còn trực quan hoá **Grad-CAM** để chỉ ra vùng ảnh mô hình dựa vào khi
    đưa ra quyết định, và **feature maps** để quan sát cách mô hình trích xuất
    đặc trưng qua từng lớp tích chập.
    """
)
