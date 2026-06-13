import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from sklearn.cluster import KMeans
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from streamlit_image_coordinates import streamlit_image_coordinates


st.title("Qiskit Quantum Craft Pixelizer")

uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])

display_style = st.selectbox(
    "Craft display style",
    ["Square Pixel", "Round Bead", "Cross Stitch X"]
)

mode = st.radio(
    "Palette mode",
    [
        "Option 1: Pick colors from image",
        "Option 2: Auto-generate colors"
    ]
)

if "picked_colors" not in st.session_state:
    st.session_state.picked_colors = []


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


def rgb_to_hex(color):
    return "#{:02x}{:02x}{:02x}".format(
        int(color[0]), int(color[1]), int(color[2])
    )


def hex_to_rgb(hex_color):
    return tuple(int(hex_color[i:i + 2], 16) for i in (1, 3, 5))


def render_craft_image(pixel_art, style, cell_size=12):
    h, w, _ = pixel_art.shape

    canvas = Image.new(
        "RGB",
        (w * cell_size, h * cell_size),
        "white"
    )

    draw = ImageDraw.Draw(canvas)

    for y in range(h):
        for x in range(w):
            color = tuple(pixel_art[y, x])
            x0 = x * cell_size
            y0 = y * cell_size
            x1 = x0 + cell_size
            y1 = y0 + cell_size

            if style == "Square Pixel":
                draw.rectangle(
                    [x0, y0, x1, y1],
                    fill=color
                )

            elif style == "Round Bead":
                margin = max(1, cell_size // 8)
                draw.ellipse(
                    [x0 + margin, y0 + margin, x1 - margin, y1 - margin],
                    fill=color,
                    outline="gray"
                )

            elif style == "Cross Stitch X":
                draw.rectangle(
                    [x0, y0, x1, y1],
                    fill="white"
                )

                line_width = max(1, cell_size // 4)

                draw.line(
                    [x0 + 2, y0 + 2, x1 - 2, y1 - 2],
                    fill=color,
                    width=line_width
                )

                draw.line(
                    [x0 + 2, y1 - 2, x1 - 2, y0 + 2],
                    fill=color,
                    width=line_width
                )

                draw.rectangle(
                    [x0, y0, x1, y1],
                    outline="lightgray"
                )

    return canvas


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

    labels = kmeans.fit_predict(quantum_features)

    palette = []

    for i in range(num_colors):
        group = flat_pixels[labels == i]

        if len(group) == 0:
            palette.append([0, 0, 0])
        else:
            palette.append(np.median(group, axis=0))

    palette = np.array(palette).astype(np.uint8)

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

def pixelize_fixed_palette(image, pixel_width, pixel_height, palette):
    image = image.convert("RGB")
    small = image.resize((pixel_width, pixel_height), Image.Resampling.BILINEAR)

    pixels = np.array(small)
    flat_pixels = pixels.reshape(-1, 3)

    pixel_features = make_quantum_features(flat_pixels)
    palette_features = make_quantum_features(np.array(palette))

    distances = np.linalg.norm(
        pixel_features[:, None, :] - palette_features[None, :, :],
        axis=2
    )

    labels = np.argmin(distances, axis=1)

    palette = np.array(palette)

    palette = np.round(palette)

    palette = palette.astype(np.uint8)
    palette = np.array(palette).astype(np.uint8)
    pixel_art = palette[labels].reshape(pixel_height, pixel_width, 3)
    pattern = labels.reshape(pixel_height, pixel_width)

    return pixel_art, palette, pattern


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.write(f"Original image size: {image.width} × {image.height}")

    pixel_width = st.slider("Pixel width", 8, 128, 64)

    recommended_height = round(pixel_width * image.height / image.width)
    recommended_height = max(8, min(128, recommended_height))

    st.info(
        f"Recommended height for original ratio: {recommended_height}"
    )

    pixel_height = st.slider("Pixel height", 8, 128, recommended_height)

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

        for i, color in enumerate(st.session_state.picked_colors):
            col1, col2 = st.columns([4, 1])

            with col1:
                selected = st.color_picker(
                    f"Color {i + 1}",
                    rgb_to_hex(color),
                    key=f"picked_color_{i}"
                )
                edited_palette.append(hex_to_rgb(selected))

            with col2:
                st.write("")
                st.write("")
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
            default_k
        )
        with st.spinner("Running Qiskit quantum color clustering..."):
            pixel_art, palette, pattern = pixelize_auto_palette(
                image,
                pixel_width,
                pixel_height,
                num_colors
            )

        st.subheader("Auto-generated Palette")

        edited_palette = []

        for i, color in enumerate(palette):
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

        cell_size = st.slider("Preview cell size", 4, 24, 10)

        craft_image = render_craft_image(
            final_pixel_art,
            display_style,
            cell_size
        )

        st.image(craft_image)

        st.subheader("Craft Pattern Number Grid")
        pattern_df = pd.DataFrame(final_pattern + 1)
        st.dataframe(pattern_df)

        craft_image.save("craft_preview.png")
        Image.fromarray(final_pixel_art).save("pixel_art_raw.png")
        pattern_df.to_csv("pattern_grid.csv", index=False)

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