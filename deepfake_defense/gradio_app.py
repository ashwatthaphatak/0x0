import gradio as gr

from .pipeline import DefensePipeline
from .stargan import deepfake_attack
from .utils import calculate_metrics, to_pil


def build_gradio_app(pipeline: DefensePipeline):
    def interactive_defense(image, attribute_choice, epsilon_val):
        img_tensor = pipeline.transform(image).unsqueeze(0).to(pipeline.device)
        attention_map = pipeline.attention_module.get_attention_map(img_tensor)

        pipeline.defense_framework.epsilon = epsilon_val
        vaccinated, _ = pipeline.defense_framework.vaccinate_image(img_tensor, attention_map)

        target_attr = pipeline.attributes[attribute_choice]
        clean_fake = deepfake_attack(pipeline.stargan_generator, img_tensor, target_attr)
        vac_fake = deepfake_attack(pipeline.stargan_generator, vaccinated, target_attr)

        def_metric = calculate_metrics(clean_fake, vac_fake)
        vis_metric = calculate_metrics(img_tensor, vaccinated)
        success = "SUCCESS" if def_metric["L2"] > 0.05 else "FAILED"

        metrics_text = (
            f"Defense: {success}\n"
            f"L2 Distance: {def_metric['L2']:.4f}\n\n"
            f"Quality:\n"
            f"PSNR: {vis_metric['PSNR']:.2f} dB\n"
            f"SSIM: {vis_metric['SSIM']:.4f}"
        )

        return to_pil(img_tensor), to_pil(vaccinated), to_pil(clean_fake), to_pil(vac_fake), metrics_text

    with gr.Blocks(theme=gr.themes.Soft(), title="Deepfake Defense") as demo:
        gr.Markdown("# Deepfake Defense: Texture-Aware Protection")

        with gr.Row():
            with gr.Column(scale=1):
                inp_image = gr.Image(type="pil", label="Upload Image")
                inp_attr = gr.Dropdown(list(pipeline.attributes.keys()), value="Blonde Hair", label="Attack Type")
                inp_epsilon = gr.Slider(0.01, 0.1, value=0.05, step=0.01, label="Epsilon")
                run_btn = gr.Button("Run Defense", variant="primary")

            with gr.Column(scale=2):
                with gr.Row():
                    out_orig = gr.Image(label="Original")
                    out_vacc = gr.Image(label="Vaccinated")
                with gr.Row():
                    out_clean = gr.Image(label="Clean -> Deepfake")
                    out_vac_f = gr.Image(label="Vaccinated -> Deepfake")
                out_metrics = gr.Textbox(label="Metrics", lines=10)

        run_btn.click(
            fn=interactive_defense,
            inputs=[inp_image, inp_attr, inp_epsilon],
            outputs=[out_orig, out_vacc, out_clean, out_vac_f, out_metrics],
        )

    return demo
