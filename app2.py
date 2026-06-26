import os
import json
import base64
from io import BytesIO
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from sklearn.cluster import KMeans
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from streamlit_image_coordinates import streamlit_image_coordinates
import math
import matplotlib.pyplot as plt





st.markdown("""
<style>

/* 버튼 글자색 강제 고정 */
.stButton > button,
.stButton > button p,
.stButton > button span,
.stButton > button div {
    color: #FEFCF7 !important;
}

.stButton > button {
    background-color: #23445F !important;
    border: 1px solid #23445F !important;
}

.stButton > button:hover {
    background-color: #151618 !important;
    border-color: #151618 !important;
}

/* 비활성화 버튼 */
.stButton > button:disabled,
.stButton > button:disabled p,
.stButton > button:disabled span {
    color: #FEFCF7 !important;
    background-color: #9AA5AA !important;
    border-color: #9AA5AA !important;
    opacity: 0.65;
}

/* info/success/warning 알림창 색상 덮어쓰기 */
div[data-testid="stAlert"] {
    background-color: #FEFCF7 !important;
    border: 1px solid #23445F !important;
    color: #23445F !important;
}

div[data-testid="stAlert"] p,
div[data-testid="stAlert"] div,
div[data-testid="stAlert"] span {
    color: #23445F !important;
}

/* success 알림창 초록색 제거 */
div[data-testid="stAlert"][kind="success"],
div[data-testid="stAlert"][data-baseweb="notification"] {
    background-color: #FEFCF7 !important;
    border-color: #23445F !important;
}

/* 슬라이더 빨간색 제거 */
[data-baseweb="slider"] div[role="slider"] {
    background-color: #23445F !important;
    border-color: #23445F !important;
}

[data-baseweb="slider"] div {
    color: #23445F !important;
}

/* 라디오 버튼 빨간색 제거 */
[data-baseweb="radio"] div[aria-checked="true"] {
    background-color: #23445F !important;
    border-color: #23445F !important;
}

[data-baseweb="radio"] div[aria-checked="false"] {
    border-color: #9AA5AA !important;
}

/* 토글 ON/OFF 색상 */
[data-baseweb="switch"] input:checked + div,
[data-testid="stToggle"] button[aria-checked="true"] {
    background-color: #23445F !important;
}

[data-baseweb="switch"] input:not(:checked) + div,
[data-testid="stToggle"] button[aria-checked="false"] {
    background-color: #C9C8C3 !important;
}

/* 토글 동그라미 */
[data-baseweb="switch"] div div {
    background-color: #FEFCF7 !important;
}

</style>
""", unsafe_allow_html=True)


if "page" not in st.session_state:
    st.session_state.page = "create"


def set_page(page_name):
    st.session_state.page = page_name


st.markdown("""
<style>
.nav-title {
    color: #FEFCF7;
    font-size: 30px;
    font-weight: 800;
    white-space: nowrap;
}

div[data-testid="stHorizontalBlock"]:has(.nav-title) {
    background-color: #151618;
    padding: 22px 60px;
    border-radius: 0 0 18px 18px;
    margin-bottom: 42px;
    width: 100vw;
    margin-left: calc(-50vw + 50%);
    box-shadow: 0 8px 24px rgba(0,0,0,0.14);
}

div[data-testid="stHorizontalBlock"]:has(.nav-title) .stButton > button {
    background-color: transparent !important;
    border: none !important;
    color: #FEFCF7 !important;
    box-shadow: none !important;
    font-weight: 650;
    font-size: 16px !important;
    white-space: nowrap !important;
}

div[data-testid="stHorizontalBlock"]:has(.nav-title) .stButton > button:hover {
    opacity: 0.7;
}
</style>
""", unsafe_allow_html=True)


col_logo, col_create, col_gallery, col_about = st.columns(
    [14, 2, 2, 1.2]
)

with col_logo:
    st.markdown(
        '<div class="nav-title">Quantum Craft Studio</div>',
        unsafe_allow_html=True
    )

with col_create:
    st.button("Create", key="nav_create", on_click=set_page, args=("create",))

with col_gallery:
    st.button("Gallery", key="nav_gallery", on_click=set_page, args=("gallery",))

with col_about:
    st.button("Info", key="nav_about", on_click=set_page, args=("about",))

page = st.session_state.page

GALLERY_FILE = "gallery_data.json"


def image_to_base64(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_image(image_base64):
    image_bytes = base64.b64decode(image_base64)
    return Image.open(BytesIO(image_bytes))


def load_gallery():
    if not os.path.exists(GALLERY_FILE):
        return []

    with open(GALLERY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_gallery(gallery_items):
    with open(GALLERY_FILE, "w", encoding="utf-8") as f:
        json.dump(gallery_items, f, ensure_ascii=False)

def build_grover_circuit(num_items, target_index):
    num_qubits = math.ceil(math.log2(num_items))
    total_states = 2 ** num_qubits

    qc = QuantumCircuit(num_qubits)

    # uniform superposition
    qc.h(range(num_qubits))

    # oracle: target_index만 phase flip
    target_bits = format(target_index, f"0{num_qubits}b")

    for qubit, bit in enumerate(reversed(target_bits)):
        if bit == "0":
            qc.x(qubit)

    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)

    for qubit, bit in enumerate(reversed(target_bits)):
        if bit == "0":
            qc.x(qubit)

    # diffuser
    qc.h(range(num_qubits))
    qc.x(range(num_qubits))

    qc.h(num_qubits - 1)
    qc.mcx(list(range(num_qubits - 1)), num_qubits - 1)
    qc.h(num_qubits - 1)

    qc.x(range(num_qubits))
    qc.h(range(num_qubits))

    return qc

def hex_to_rgb(hex_color):
    return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))


def rgb_to_hex(color):
    return "#{:02x}{:02x}{:02x}".format(
        int(color[0]), int(color[1]), int(color[2])
    )

def make_quantum_features(pixels):
    features = []

    for pixel in pixels:
        rgb_feature = np.array(pixel) / 255.0

        quantum_feature = quantum_feature_map(pixel)

        combined_feature = np.concatenate([
            rgb_feature,
            quantum_feature
        ])

        features.append(combined_feature)

    return np.array(features)

def quantum_feature_map(rgb):
    rgb = np.array(rgb) / 255.0
    r, g, b = rgb

    qc = QuantumCircuit(3)

    qc.ry(np.pi * r, 0)
    qc.ry(np.pi * g, 1)
    qc.ry(np.pi * b, 2)

    qc.cx(0, 1)
    qc.cx(1, 2)

    qc.rz(np.pi * r * g, 0)
    qc.rz(np.pi * g * b, 1)
    qc.rz(np.pi * b * r, 2)

    state = Statevector.from_instruction(qc)
    probs = np.abs(state.data) ** 2

    return probs





st.markdown("""
<style>
/* 전체 Streamlit 본문 폭을 화면 전체로 확장 */
section.main > div {
    max-width: 100% !important;
    padding-left: 0rem !important;
    padding-right: 0rem !important;
}

/* 일반 컨텐츠는 다시 적당한 폭으로 중앙 정렬 */
section.main > div > div:not(:first-child) {
    max-width: 1200px !important;
    margin-left: auto !important;
    margin-right: auto !important;
}

/* 상단바가 들어있는 첫 번째 horizontal block만 전체 폭 */
div[data-testid="stHorizontalBlock"]:has(.nav-title) {
    width: 100vw !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
    border-radius: 0 0 18px 18px !important;
}

/* 상단바 내부 폭 확보 */
div[data-testid="stHorizontalBlock"]:has(.nav-title) {
    padding-left: 64px !important;
    padding-right: 64px !important;
}

/* 메뉴 버튼 간격 확보 */
div[data-testid="stHorizontalBlock"]:has(.nav-title) .stButton > button {
    min-width: max-content !important;
    white-space: nowrap !important;
}
</style>
""", unsafe_allow_html=True)


if page == "create":

    st.title("Quantum Craft Studio")

    st.markdown("""
    ### Transform Images into Craft Patterns with Quantum Computing

    Quantum Craft Studio converts ordinary images into Pixel Art, Bead Art, and Cross-Stitch patterns.

    Unlike conventional pattern generators that rely only on RGB values, this application applies Quantum Feature Mapping and Grover Search to perform quantum-inspired color processing and palette selection.
    """)
    with st.expander("Why Quantum?"):

        st.markdown("""
    ### Why use Quantum Computing?

    Traditional craft pattern generators compare colors only in RGB space.

    However, RGB values alone may not fully capture relationships between color channels.

    This application improves color representation by:

    - Encoding RGB values into quantum states using **Quantum Feature Mapping**
    - Capturing color relationships through **entanglement**
    - Generating quantum feature vectors
    - Applying **Grover Search** to select the optimal palette color

    This provides a richer representation of colors than using RGB values alone.
    """)

    st.markdown("### Learn More")

    st.markdown("""
        - [Qiskit Documentation](https://quantum.cloud.ibm.com/docs/en/guides/tools-intro)
        - [Grover's Algorithm](https://learning.quantum.ibm.com/course/fundamentals-of-quantum-algorithms/grovers-algorithm)
        - [Quantum Feature Maps (Qiskit)](https://www.quandela.com/resources/quantum-computing-glossary/quantum-feature-map/, https://qiskit.org/documentation/machine-learning/)
        """)

    uploaded_file = st.file_uploader(
        "Upload image",
        type=["png", "jpg", "jpeg"],
        help="Upload an image to convert it into a craft pattern."
    )

    display_style = st.selectbox(
        "Craft display style",
        ["Square Pixel", "Round Bead", "Cross Stitch X"],
        help="Choose how the final craft pattern will be displayed."
    )

    mode = st.radio(
        "Palette mode",
        [
            "Option 1: Pick colors from image",
            "Option 2: Auto-generate colors"
        ],
        help="""
    Option 1:
    Manually select palette colors from the image.

    Option 2:
    Automatically generate palette colors using Quantum Feature Mapping, KMeans clustering, and Grover Search.
    """
    )

    if "picked_colors" not in st.session_state:
        st.session_state.picked_colors = []





    def render_craft_image(pixel_art, style, pattern=None, show_numbers=False, cell_size=12):
        h, w, _ = pixel_art.shape

        canvas = Image.new("RGB", (w * cell_size, h * cell_size), "white")
        draw = ImageDraw.Draw(canvas)

        for y in range(h):
            for x in range(w):
                color = tuple(pixel_art[y, x])
                x0 = x * cell_size
                y0 = y * cell_size
                x1 = x0 + cell_size - 1
                y1 = y0 + cell_size - 1

                if "Square Pixel" in style:
                    draw.rectangle([x0, y0, x1, y1], fill=color)
                    draw.rectangle([x0, y0, x1, y1], outline="lightgray")

                elif "Round Bead" in style:
                    margin = max(1, cell_size // 8)
                    draw.ellipse(
                        [x0 + margin, y0 + margin, x1 - margin, y1 - margin],
                        fill=color,
                        outline="gray"
                    )

                elif "Cross Stitch X" in style:
                    draw.rectangle([x0, y0, x1, y1], fill="white")

                    line_width = max(1, cell_size // 5)

                    draw.line(
                        [x0 + 3, y0 + 3, x1 - 3, y1 - 3],
                        fill=color,
                        width=line_width
                    )

                    draw.line(
                        [x0 + 3, y1 - 3, x1 - 3, y0 + 3],
                        fill=color,
                        width=line_width
                    )

                    draw.rectangle([x0, y0, x1, y1], outline="lightgray")

                if show_numbers and pattern is not None:
                    number = str(int(pattern[y, x]) + 1)

                    font_size = max(6, int(cell_size * 0.55))

                    try:
                        font = ImageFont.truetype("arial.ttf", font_size)
                    except:
                        font = ImageFont.load_default()

                    bbox = draw.textbbox((0, 0), number, font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]

                    while (text_w > cell_size * 0.85 or text_h > cell_size * 0.85) and font_size > 5:
                        font_size -= 1
                        try:
                            font = ImageFont.truetype("arial.ttf", font_size)
                        except:
                            font = ImageFont.load_default()

                        bbox = draw.textbbox((0, 0), number, font=font)
                        text_w = bbox[2] - bbox[0]
                        text_h = bbox[3] - bbox[1]

                    brightness = int(color[0]) + int(color[1]) + int(color[2])
                    text_color = "black" if brightness > 420 else "white"

                    text_x = x0 + (cell_size - text_w) / 2
                    text_y = y0 + (cell_size - text_h) / 2 - 1

                    draw.text((text_x, text_y), number, fill=text_color, font=font)

        return canvas


    def make_color_usage_table(pattern, palette):
        color_ids, counts = np.unique(pattern, return_counts=True)

        rows = []

        total = int(np.sum(counts))

        for color_id, count in zip(color_ids, counts):
            color = palette[color_id]

            rows.append({
                "Color No.": int(color_id) + 1,
                "HEX": rgb_to_hex(color),
                "RGB": f"({int(color[0])}, {int(color[1])}, {int(color[2])})",
                "Bead / Stitch Count": int(count),
                "Percentage": round(int(count) / total * 100, 2)
            })

        return pd.DataFrame(rows)


    def pixelize_auto_palette(image, pixel_width, pixel_height, num_colors):
        image = image.convert("RGB")
        small = image.resize((pixel_width, pixel_height), Image.Resampling.BILINEAR)

        pixels = np.array(small)
        flat_pixels = pixels.reshape(-1, 3)

        quantum_features = make_quantum_features(flat_pixels)

        kmeans = KMeans(
            n_clusters=num_colors,
            random_state=0,
            n_init=10
        )

        initial_labels = kmeans.fit_predict(quantum_features)

        palette = []

        for i in range(num_colors):
            group = flat_pixels[initial_labels == i]

            if len(group) == 0:
                palette.append([0, 0, 0])
            else:
                palette.append(np.median(group, axis=0))

        palette = np.array(palette).astype(np.uint8)

        palette_features = make_quantum_features(palette)

        labels = assign_labels_with_grover(
            quantum_features,
            palette_features
        )

        pixel_art = palette[labels].reshape(pixel_height, pixel_width, 3)
        pattern = labels.reshape(pixel_height, pixel_width)

        return pixel_art, palette, pattern

    def recommend_num_colors(
        image,
        pixel_width,
        pixel_height,
        min_k=2,
        max_k=32,
        improvement_threshold=0.08
    ):
        image = image.convert("RGB")
        small = image.resize((pixel_width, pixel_height), Image.Resampling.BILINEAR)

        pixels = np.array(small)
        flat_pixels = pixels.reshape(-1, 3)
        features = flat_pixels / 255.0

        errors = []

        for k in range(min_k, max_k + 1):
            kmeans = KMeans(
                n_clusters=k,
                random_state=0,
                n_init=10
            )

            labels = kmeans.fit_predict(features)

            palette = []

            for i in range(k):
                group = flat_pixels[labels == i]

                if len(group) == 0:
                    palette.append([0, 0, 0])
                else:
                    palette.append(np.median(group, axis=0))

            palette = np.array(palette)
            reconstructed = palette[labels]

            mse = np.mean((flat_pixels - reconstructed) ** 2)
            errors.append((k, mse))

        recommended_k = max_k

        for i in range(1, len(errors)):
            previous_error = errors[i - 1][1]
            current_error = errors[i][1]

            improvement = (previous_error - current_error) / previous_error

            if improvement < improvement_threshold:
                recommended_k = errors[i - 1][0]
                break

        score_rows = []

        first_error = errors[0][1]

        for k, mse in errors:
            relative_quality = 1 - (mse / first_error)

            score_rows.append((
                k,
                round(mse, 2),
                round(relative_quality, 4)
            ))

        return recommended_k, score_rows


    def grover_select_palette_color(pixel_feature, palette_features):
        distances = np.linalg.norm(
            palette_features - pixel_feature,
            axis=1
        )

        target_index = int(np.argmin(distances))

        qc = build_grover_circuit(
            num_items=len(palette_features),
            target_index=target_index
        )

        state = Statevector.from_instruction(qc)
        probs = np.abs(state.data) ** 2

        selected_index = int(np.argmax(probs[:len(palette_features)]))

        return selected_index


    def assign_labels_with_grover(pixel_features, palette_features):
        labels = []

        for pixel_feature in pixel_features:
            selected = grover_select_palette_color(
                pixel_feature,
                palette_features
            )
            labels.append(selected)

        return np.array(labels)

    def pixelize_fixed_palette(image, pixel_width, pixel_height, palette):
        image = image.convert("RGB")
        small = image.resize((pixel_width, pixel_height), Image.Resampling.BILINEAR)

        pixels = np.array(small)
        flat_pixels = pixels.reshape(-1, 3)

        pixel_features = make_quantum_features(flat_pixels)
        palette_features = make_quantum_features(np.array(palette))

        labels = []

        for pixel_feature in pixel_features:
            selected = grover_select_palette_color(
                pixel_feature,
                palette_features
            )
            labels.append(selected)

        labels = np.array(labels)

        palette = np.array(palette)

        palette = np.round(palette)

        palette = palette.astype(np.uint8)
        palette = np.array(palette).astype(np.uint8)
        pixel_art = palette[labels].reshape(pixel_height, pixel_width, 3)
        pattern = labels.reshape(pixel_height, pixel_width)

        return pixel_art, palette, pattern




    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        st.markdown("---")

        show_process = st.toggle(
            "Show Quantum Process",
            help="Display the complete quantum processing pipeline used by this application."
        )

        st.write(f"Original image size: {image.width} × {image.height}")

        pixel_width = st.slider(
            "Pixel width",
            8,
            128,
            64,
            help="Higher values preserve more image detail but require more beads or stitches."
        )

        recommended_height = round(pixel_width * image.height / image.width)
        recommended_height = max(8, min(128, recommended_height))

        st.info(
            f"Recommended height for original ratio: {recommended_height}"
        )

        pixel_height = st.slider(
            "Pixel height",
            8,
            128,
            recommended_height,
            help="Controls the vertical resolution of the craft pattern."
        )

        st.subheader("Original Image")
        st.image(image)

        final_pixel_art = None
        final_pattern = None

        if mode == "Option 1: Pick colors from image":
            st.subheader("Pick colors from image")

            display_width = 500
            ratio = display_width / image.width
            display_height = int(image.height * ratio)

            display_image = image.resize(
                (display_width, display_height),
                Image.Resampling.BILINEAR
            )

            coords = streamlit_image_coordinates(display_image, key="picker")

            if coords is not None:
                x_display = coords["x"]
                y_display = coords["y"]

                x_original = int(x_display / ratio)
                y_original = int(y_display / ratio)

                x_original = min(max(x_original, 0), image.width - 1)
                y_original = min(max(y_original, 0), image.height - 1)

                picked_color = image.getpixel((x_original, y_original))

                st.write("Current picked color:", rgb_to_hex(picked_color))
                st.color_picker(
                    "Preview picked color",
                    rgb_to_hex(picked_color),
                    disabled=True
                )

                if st.button("Add this color"):
                    st.session_state.picked_colors.append(picked_color)

            if st.button("Clear picked colors"):
                st.session_state.picked_colors = []

            st.subheader("Picked Palette")

            edited_palette = []

            num_columns = 10

            for row_start in range(0, len(st.session_state.picked_colors), num_columns):
                cols = st.columns(num_columns)

                for col_idx in range(num_columns):
                    i = row_start + col_idx

                    if i >= len(st.session_state.picked_colors):
                        with cols[col_idx]:
                            st.empty()
                        continue

                    color = st.session_state.picked_colors[i]

                    with cols[col_idx]:
                        selected = st.color_picker(
                            f"Color {i + 1}",
                            rgb_to_hex(color),
                            key=f"picked_color_{i}"
                        )

                        edited_palette.append(hex_to_rgb(selected))

                        if st.button("Remove", key=f"remove_color_{i}"):
                            st.session_state.picked_colors.pop(i)
                            st.rerun()

            if len(edited_palette) < 2:
                st.warning("Pick at least 2 colors.")
            else:
                with st.spinner("Running Qiskit quantum feature mapping..."):
                    pixel_art, palette, pattern = pixelize_fixed_palette(
                        image,
                        pixel_width,
                        pixel_height,
                        edited_palette
                    )

                final_pixel_art = pixel_art
                final_pattern = pattern

        else:
            if st.button("Recommend number of colors"):
                with st.spinner("Finding recommended number of colors..."):
                    recommended_k, color_scores = recommend_num_colors(
                        image,
                        pixel_width,
                        pixel_height,
                        min_k=2,
                        max_k=32,
                        improvement_threshold=0.08
                    )

                st.session_state.recommended_k = recommended_k
                st.session_state.color_scores = color_scores

            if "recommended_k" in st.session_state:
                st.success(
                    f"Recommended number of colors: {st.session_state.recommended_k}"
                )

                score_df = pd.DataFrame(
                    st.session_state.color_scores,
                    columns=["Number of colors", "MSE error", "Relative quality"]
                )

                st.dataframe(score_df)
            default_k = st.session_state.get("recommended_k", 8)

            num_colors = st.slider(
                "Number of colors",
                2,
                32,
                default_k,
                help="More colors produce a result closer to the original image but increase the complexity of the craft pattern."
            )
            with st.spinner("Running quantum clustering and Grover palette search..."):
                pixel_art, palette, pattern = pixelize_auto_palette(
                    image,
                    pixel_width,
                    pixel_height,
                    num_colors
                )

            st.subheader("Auto-generated Palette")

            edited_palette = []

            num_columns = 10

            for row_start in range(0, len(palette), num_columns):
                cols = st.columns(num_columns)

                for col_idx in range(num_columns):
                    i = row_start + col_idx

                    if i >= len(palette):
                        with cols[col_idx]:
                            st.empty()
                        continue

                    color = palette[i]

                    with cols[col_idx]:
                        selected = st.color_picker(
                            f"Color {i + 1}",
                            rgb_to_hex(color),
                            key=f"auto_color_{i}"
                        )

                        edited_palette.append(hex_to_rgb(selected))

            edited_palette = np.array(edited_palette, dtype=np.uint8)
            final_pixel_art = edited_palette[pattern]
            final_pattern = pattern

        if final_pixel_art is not None:
            st.subheader("Craft Preview")

            col_a, col_b, col_c = st.columns(3)

            with col_a:
                cell_size = st.slider(
                    "Preview cell size",
                    8,
                    32,
                    18,
                    help="Changes only the display size of each cell without affecting the actual pattern resolution."
                )

            with col_b:
                show_numbers = st.toggle(
                    "Show color numbers",
                    value=False,
                    help="Display palette numbers inside each cell for easier crafting."
                )

            with col_c:
                compare_with_original = st.toggle(
                    "Compare with original",
                    value=True,
                    help="Show the original image and the generated craft pattern side by side."
                )

            craft_image = render_craft_image(
                final_pixel_art,
                display_style,
                pattern=final_pattern,
                show_numbers=show_numbers,
                cell_size=cell_size
            )

            if compare_with_original:
                left, right = st.columns(2)

                with left:
                    st.markdown("### Original")
                    st.image(image, use_container_width=True)

                with right:
                    st.markdown("### Craft Pattern")
                    st.image(craft_image, use_container_width=True)


            else:
                st.image(craft_image, use_container_width=True)

            if show_process:

                st.markdown("---")
                st.header("Quantum Processing Pipeline")

                st.subheader("Step 1. Uploaded Image")

                st.image(image, width=300)

                st.subheader("Step 2. Sample Pixel")

                sample_pixel = image.getpixel((0, 0))

                st.write(f"RGB : {sample_pixel}")

                st.color_picker(
                    "Sample Pixel",
                    rgb_to_hex(sample_pixel),
                    disabled=True,
                    key="sample_process"
                )

                st.subheader("Step 3. Quantum Feature Mapping")

                r, g, b = np.array(sample_pixel) / 255

                qc = QuantumCircuit(3)

                qc.ry(np.pi * r, 0)
                qc.ry(np.pi * g, 1)
                qc.ry(np.pi * b, 2)

                qc.cx(0, 1)
                qc.cx(1, 2)

                qc.rz(np.pi * r * g, 0)
                qc.rz(np.pi * g * b, 1)
                qc.rz(np.pi * b * r, 2)

                st.pyplot(qc.draw(output="mpl"))

                state = Statevector.from_instruction(qc)

                probs = np.abs(state.data) ** 2

                feature_df = pd.DataFrame({
                    "State": [format(i, "03b") for i in range(8)],
                    "Probability": probs
                })

                st.dataframe(feature_df)

                st.subheader("Step 4. Generated Palette")

                cols = st.columns(len(edited_palette))

                for i, color in enumerate(edited_palette):
                    with cols[i]:
                        st.color_picker(
                            "",
                            rgb_to_hex(color),
                            disabled=True,
                            key=f"process_palette{i}"
                        )

                        st.caption(f"Color {i + 1}")

                st.subheader("Step 5. Grover Search")

                sample_feature = make_quantum_features(
                    np.array([sample_pixel])
                )[0]

                palette_features = make_quantum_features(
                    edited_palette
                )

                distances = np.linalg.norm(
                    palette_features - sample_feature,
                    axis=1
                )

                target = np.argmin(distances)

                grover = build_grover_circuit(
                    len(edited_palette),
                    target
                )

                st.pyplot(
                    grover.draw(output="mpl")
                )

                state = Statevector.from_instruction(grover)

                prob = np.abs(state.data) ** 2

                st.bar_chart(prob[:len(edited_palette)])

                st.subheader("Step 6. Final Craft Pattern")
                st.image(craft_image)

            st.subheader("Craft Pattern Number Grid")
            pattern_df = pd.DataFrame(final_pattern + 1)
            st.dataframe(pattern_df)



            st.subheader("Bead / Stitch Count")

            usage_df = make_color_usage_table(
                final_pattern,
                final_pixel_art.reshape(-1, 3)[
                    np.unique(final_pattern, return_index=True)[1]
                ]
            )


            # 위 방식은 색 순서가 꼬일 수 있어서 더 안전하게 다시 계산
            palette_for_usage = []

            for color_id in range(int(final_pattern.max()) + 1):
                pixels_for_color = final_pixel_art[final_pattern == color_id]

                if len(pixels_for_color) > 0:
                    palette_for_usage.append(pixels_for_color[0])
                else:
                    palette_for_usage.append([0, 0, 0])

            palette_for_usage = np.array(palette_for_usage, dtype=np.uint8)

            usage_df = make_color_usage_table(
                final_pattern,
                palette_for_usage
            )

            st.dataframe(usage_df)

            total_count = int(usage_df["Bead / Stitch Count"].sum())

            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>Total beads / stitches needed: {total_count}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

            pattern_title = st.text_input(
                "Pattern title for gallery",
                value="My Craft Pattern"
            )

            if st.button("Save to Gallery"):
                gallery_items = load_gallery()

                gallery_items.append({
                    "title": pattern_title,
                    "image_base64": image_to_base64(craft_image)
                })

                save_gallery(gallery_items)

                st.success("Saved to Gallery.")


            craft_image.save("craft_preview.png")
            Image.fromarray(final_pixel_art).save("pixel_art_raw.png")
            pattern_df.to_csv("pattern_grid.csv", index=False)

            usage_df.to_csv("bead_stitch_count.csv", index=False)

            with open("craft_preview.png", "rb") as f:
                st.download_button(
                    "Download Craft Preview PNG",
                    f,
                    file_name="craft_preview.png"
                )

            with open("pixel_art_raw.png", "rb") as f:
                st.download_button(
                    "Download Raw Pixel Art PNG",
                    f,
                    file_name="pixel_art_raw.png"
                )

            with open("pattern_grid.csv", "rb") as f:
                st.download_button(
                    "Download Pattern CSV",
                    f,
                    file_name="pattern_grid.csv"
                )

            with open("bead_stitch_count.csv", "rb") as f:
                st.download_button(
                    "Download Bead / Stitch Count CSV",
                    f,
                    file_name="bead_stitch_count.csv"
                )


elif page == "gallery":

    st.subheader("Gallery")

    gallery_items = load_gallery()

    if len(gallery_items) == 0:
        st.info("No saved patterns yet.")
    else:
        cols = st.columns(3)

        for i, item in enumerate(gallery_items):
            with cols[i % 3]:
                image = base64_to_image(item["image_base64"])
                st.image(
                    image,
                    caption=item["title"],
                    use_container_width=True
                )


elif page == "about":
    st.subheader("Info")

    st.write(
        """
        Quantum Craft Studio converts photos into craft patterns for pixel art,
        bead art, and cross-stitch. The app uses Qiskit-based quantum feature
        mapping to transform RGB colors into quantum features, then applies
        clustering and Grover search to generate craft patterns.
        """
    )

    st.markdown("---")

    st.header("Quantum Algorithm Visualization")

    st.subheader("1. Overall Pipeline")

    st.markdown(
        """
        ```text
        Uploaded Image
              ↓
        RGB Pixel Values
              ↓
        Quantum Feature Map
              ↓
        Quantum Feature Vector
              ↓
        KMeans Palette Generation
              ↓
        Grover Palette Search
              ↓
        Final Craft Pattern
        ```
        """
    )

    st.subheader("2. Quantum Feature Map Circuit")

    sample_rgb = st.color_picker(
        "Choose a sample RGB color",
        "#7A50C8",
        key="info_sample_color"
    )



    sample_color = hex_to_rgb(sample_rgb)

    # Normalize RGB
    r, g, b = np.array(sample_color) / 255.0

    st.markdown("### Rotation Angle Calculation")

    st.latex(
        rf"""
        R = {sample_color[0]}, \quad
        G = {sample_color[1]}, \quad
        B = {sample_color[2]}
        """
    )

    st.latex(
        rf"""
        \begin{{aligned}}
        r &= \frac{{{sample_color[0]}}}{{255}} = {r:.3f} \\
        g &= \frac{{{sample_color[1]}}}{{255}} = {g:.3f} \\
        b &= \frac{{{sample_color[2]}}}{{255}} = {b:.3f}
        \end{{aligned}}
        """
    )

    st.latex(
        rf"""
        \begin{{aligned}}
        \theta_R &= \pi r = {np.pi * r:.3f}\ \text{{rad}} \\\\
        \theta_G &= \pi g = {np.pi * g:.3f}\ \text{{rad}} \\\\
        \theta_B &= \pi b = {np.pi * b:.3f}\ \text{{rad}}
        \end{{aligned}}
        """
    )

    st.latex(
        rf"""
        \begin{{aligned}}
        \phi_{{RG}} &= \pi rg = {np.pi * r * g:.3f}\ \text{{rad}} \\\\
        \phi_{{GB}} &= \pi gb = {np.pi * g * b:.3f}\ \text{{rad}} \\\\
        \phi_{{BR}} &= \pi br = {np.pi * b * r:.3f}\ \text{{rad}}
        \end{{aligned}}
        """
    )

    st.info(
        """
    **How are the rotation angles determined?**

    1. Convert the RGB values (0–255) into normalized values (0–1).
    2. Encode each normalized value into an RY rotation:
       - Qubit 0 : π × R
       - Qubit 1 : π × G
       - Qubit 2 : π × B
    3. Encode interactions between color channels using RZ rotations:
       - π × R × G
       - π × G × B
       - π × B × R
    """
    )

    r, g, b = np.array(sample_color) / 255.0

    feature_qc = QuantumCircuit(3)

    feature_qc.ry(np.pi * r, 0)
    feature_qc.ry(np.pi * g, 1)
    feature_qc.ry(np.pi * b, 2)

    feature_qc.cx(0, 1)
    feature_qc.cx(1, 2)

    feature_qc.rz(np.pi * r * g, 0)
    feature_qc.rz(np.pi * g * b, 1)
    feature_qc.rz(np.pi * b * r, 2)

    st.pyplot(feature_qc.draw(output="mpl"))

    feature_state = Statevector.from_instruction(feature_qc)
    feature_probs = np.abs(feature_state.data) ** 2

    feature_df = pd.DataFrame({
        "Quantum State": [format(i, "03b") for i in range(8)],
        "Probability": feature_probs
    })

    st.write("Quantum feature probabilities:")
    st.dataframe(feature_df)

    st.subheader("3. Grover Search Circuit")

    num_palette_colors = st.slider(
        "Number of palette candidates",
        2,
        8,
        8,
        key="info_grover_num_colors"
    )

    example_palette = np.array([
        [122, 80, 200],
        [255, 180, 0],
        [60, 200, 100],
        [50, 120, 255],
        [255, 60, 120],
        [40, 40, 40],
        [230, 230, 230],
        [150, 90, 40]
    ])
    st.subheader("Palette Candidates")

    cols = st.columns(4)

    for i in range(num_palette_colors):
        color = example_palette[i]
        hex_color = rgb_to_hex(color)

        with cols[i % 4]:
            st.color_picker(
                f"Color {i + 1}",
                value=hex_color,
                disabled=True,
                key=f"palette_demo_{i}"
            )

            st.caption(hex_color)
            st.caption(f"RGB {tuple(color)}")

    sample_feature = make_quantum_features(
        np.array([sample_color])
    )[0]

    palette_features = make_quantum_features(
        example_palette[:num_palette_colors]
    )

    distances = np.linalg.norm(
        palette_features - sample_feature,
        axis=1
    )

    target_index = int(np.argmin(distances))

    st.info(
        f"The closest palette color to the sample pixel is Color {target_index + 1}. "
        "This color is used as the Grover oracle target."
    )

    grover_qc = build_grover_circuit(
        num_items=num_palette_colors,
        target_index=target_index
    )

    st.pyplot(grover_qc.draw(output="mpl"))

    grover_state = Statevector.from_instruction(grover_qc)
    grover_probs = np.abs(grover_state.data) ** 2

    valid_probs = grover_probs[:num_palette_colors]





    grover_df = pd.DataFrame({
        "Palette Color": [f"Color {i + 1}" for i in range(num_palette_colors)],
        "Probability after Grover": valid_probs
    })

    st.write("Grover output probabilities:")
    st.dataframe(grover_df)

    st.bar_chart(
        grover_df.set_index("Palette Color")
    )

    selected_color = int(np.argmax(valid_probs)) + 1

    winner = example_palette[selected_color - 1]

    winner_hex = rgb_to_hex(winner)

    st.subheader("Selected Palette Color")

    winner = example_palette[selected_color - 1]
    winner_hex = rgb_to_hex(winner)

    st.color_picker(
        "Grover Selected Color",
        value=winner_hex,
        disabled=True,
        key="winner_color"
    )

    st.success(
        f"Grover selected Color {selected_color}"
    )

    st.success(
        f"Grover search selects Color {selected_color} with the highest probability."
    )

    st.subheader("4. How Grover Is Used in This App")

    st.write(
        """
        In this application, each palette color is treated as a search candidate.
        The closest palette color to the current pixel is marked as the target state
        by the Grover oracle. Then, the diffuser amplifies the probability of the
        target state. Finally, the palette color with the highest probability is
        selected as the color for that pixel.
        """
    )

