import gradio as gr

from .pipeline import DefensePipeline
from .stargan import deepfake_attack
from .utils import blend_face_effect, calculate_metrics, crop_face_region, ensure_square_image, to_pil


def build_gradio_app(pipeline: DefensePipeline):
    min_face_size = 60

    def interactive_defense(image, attribute_choice, epsilon_val, face_margin_val, blend_feather_val):
        full_pil = image.convert("RGB")
        full_pil, was_cropped_to_square = ensure_square_image(full_pil)
        full_tensor = pipeline.transform(full_pil).unsqueeze(0).to(pipeline.device)
        _, face_bbox, used_face = crop_face_region(full_pil, margin=face_margin_val, min_size=min_face_size)
        attention_map = pipeline.attention_module.get_attention_map(full_tensor)

        pipeline.defense_framework.epsilon = epsilon_val
        vaccinated_full_tensor, _ = pipeline.defense_framework.vaccinate_image(full_tensor, attention_map)
        vaccinated_full_pil = to_pil(vaccinated_full_tensor).resize(full_pil.size)

        target_attr = pipeline.attributes[attribute_choice]
        clean_fake_full = deepfake_attack(pipeline.stargan_generator, full_tensor, target_attr)
        vac_fake_full = deepfake_attack(pipeline.stargan_generator, vaccinated_full_tensor, target_attr)

        clean_fake_full_pil = to_pil(clean_fake_full).resize(full_pil.size)
        vac_fake_full_pil = to_pil(vac_fake_full).resize(full_pil.size)

        out_clean = blend_face_effect(full_pil, clean_fake_full_pil, face_bbox, feather_ratio=blend_feather_val)
        out_vac_f = blend_face_effect(vaccinated_full_pil, vac_fake_full_pil, face_bbox, feather_ratio=blend_feather_val)

        def_metric = calculate_metrics(clean_fake_full, vac_fake_full)
        vis_metric = calculate_metrics(full_tensor, vaccinated_full_tensor)
        success = "SUCCESS" if def_metric["L2"] > 0.05 else "FAILED"
        bbox_text = f"face_bbox={face_bbox}" if used_face else "face_bbox=full_image_fallback"
        square_text = "square_center_crop_applied" if was_cropped_to_square else "already_square"

        metrics_text = (
            f"Defense: {success}\n"
            f"L2 Distance: {def_metric['L2']:.4f}\n\n"
            f"Quality:\n"
            f"PSNR: {vis_metric['PSNR']:.2f} dB\n"
            f"SSIM: {vis_metric['SSIM']:.4f}\n\n"
            f"Display ROI: {bbox_text}\n"
            f"Input: {square_text}\n"
            f"Face Margin (display): {face_margin_val:.2f}\n"
            f"Blend Feather (display): {blend_feather_val:.2f}"
        )

        return full_pil, vaccinated_full_pil, out_clean, out_vac_f, metrics_text

    with gr.Blocks(theme=gr.themes.Soft(), title="Deepfake Defense") as demo:
        gr.Markdown("# Deepfake Defense: Texture-Aware Protection")
        gr.Markdown("Vaccination runs on the full image. Face detection is used only to localize the displayed deepfake effect.")

        with gr.Row():
            with gr.Column(scale=1):
                inp_image = gr.Image(type="pil", label="Upload Image")
                inp_attr = gr.Dropdown(list(pipeline.attributes.keys()), value="Blonde Hair", label="Attack Type")
                inp_epsilon = gr.Slider(0.01, 0.1, value=0.05, step=0.01, label="Epsilon")
                inp_face_margin = gr.Slider(0.10, 0.60, value=0.30, step=0.01, label="Face Margin")
                inp_blend_feather = gr.Slider(0.00, 0.25, value=0.10, step=0.01, label="Blend Feather")
                run_btn = gr.Button("Run Defense", variant="primary")

            with gr.Column(scale=2):
                with gr.Row():
                    out_orig = gr.Image(label="Original (Full)")
                    out_vacc = gr.Image(label="Vaccinated (Full)")
                with gr.Row():
                    out_clean = gr.Image(label="Deepfake Effect on Original (Face Region)")
                    out_vac_f = gr.Image(label="Deepfake Effect on Vaccinated (Face Region)")
                out_metrics = gr.Textbox(label="Metrics", lines=10)

        run_btn.click(
            fn=interactive_defense,
            inputs=[inp_image, inp_attr, inp_epsilon, inp_face_margin, inp_blend_feather],
            outputs=[out_orig, out_vacc, out_clean, out_vac_f, out_metrics],
        )

    return demo
