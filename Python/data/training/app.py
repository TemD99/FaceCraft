import gradio as gr
import subprocess
from css import css

# python /app/Python/data/SG2/train.py --outdir=/data/output --data=/app/Data/TESTDATASET.zip --img-resolution=256 --cfg=lite --batch=56
# python train script                    output folder               data zip                        resolution       flag       batch flag

gr.set_static_paths(['.'])
title = 'FaceCraft StyleGan-T Training'

LAST_PROMPT = None

def build_command(
    dir_output_textbox,
    dir_data_textbox,
    img_resolution_numbox,
    batch_numbox,
    batch_gpu_numbox,
    config_radio,
    cbase_numbox,
    cmax_numbox,
    res_blocks_numbox,
    resume_textbox,
    resume_numbox,
    train_mode_radio,
    clip_weight_numbox,
    init_blur_numbox,
    blur_duration_numbox,
    suffix_textbox,
    metrics_textbox,
    duration_numbox,
    progress_numbox,
    snap_numbox,
    seed_numbox,
    mix_precision_checkbox,
    benchmarking_checkbox,
    dataloader_numbox,
    dry_run_checkbox,
    dir_output_toggle,
    dir_data_toggle,
    img_resolution_toggle,
    batch_toggle,
    batch_gpu_toggle,
    config_toggle,
    cbase_toggle,
    cmax_toggle,
    res_blocks_toggle,
    resume_toggle,
    resume_num_toggle,
    train_mode_toggle,
    clip_weight_toggle,
    init_blur_toggle,
    blur_duration_toggle,
    suffix_toggle,
    metrics_toggle,
    duration_toggle,
    progress_toggle,
    snap_toggle,
    seed_toggle,
    mix_precision_toggle,
    benchmarking_toggle,
    dataloader_toggle,
    dry_run_toggle):

    command = "python /app/Python/data/SG2/train.py"

    if(dir_output_toggle):
        command += f" --outdir={dir_output_textbox}"
    
    if(dir_data_toggle):
        command += f" --data={dir_data_textbox}"

    if(img_resolution_toggle):
        command += f" --img-resolution={img_resolution_numbox}"

    if(batch_toggle):
        command += f" --batch={batch_numbox}"

    if(batch_gpu_toggle):
        command += f" --batch-gpu={batch_gpu_numbox}"

    if(config_toggle):
        command += f" --cfg={config_radio}"
    
    if(cbase_toggle):
        command += f" --cbase={cbase_numbox}"

    if(cmax_toggle):
        command += f" --cmax={cmax_numbox}"

    if(res_blocks_toggle):
        command += f" --res-blocks={res_blocks_numbox}"

    if(resume_toggle):
        command += f" --resume={resume_textbox}"

    if(resume_num_toggle):
        command += f" --resume-kimg={resume_numbox}"

    if(train_mode_toggle):
        command += f" --train-mode={train_mode_radio}"

    if(clip_weight_toggle):
        command += f" --clip-weight={clip_weight_numbox}"

    if(init_blur_toggle):
        command += f" --blur-init={init_blur_numbox}"

    if(blur_duration_toggle):
        command += f" --blur-fade-kimg={blur_duration_numbox}"

    if(suffix_toggle):
        command += f" --suffix={suffix_textbox}"

    if(metrics_toggle):
        command += f" --metrics=[{metrics_textbox}]"

    if(duration_toggle):
        command += f" --kimg={duration_numbox}"

    if(progress_toggle):
        command += f" --tick={progress_numbox}"

    if(snap_toggle):
        command += f" --snap={snap_numbox}"

    if(seed_toggle):
        command += f" --seed={seed_numbox}"

    if(mix_precision_toggle):
        command += f" --fp32={mix_precision_checkbox}"

    if(benchmarking_toggle):
        command += f" --nobench={benchmarking_checkbox}"

    if(dataloader_toggle):
        command += f" --workers={dataloader_numbox}"

    if(dry_run_toggle):
        command += f" --dry-run={dry_run_checkbox}"

    return command


with gr.Blocks(title=title,css=css) as demo:
    gr.HTML("""
                <header>
                    <img class="logo" src="file/Assets/logo.png" />
                    <h1>FaceCraft</h1>
                </header>
            """)

    with gr.Row():
        debug_label = gr.Textbox(value="Waiting to start training...",label="Debug Output",show_label=True,lines=7,interactive=False)

    with gr.Accordion("Inputs"):
        # Required.
        gr.Label("Required.")
        with gr.Row():
            dir_output_textbox = gr.Textbox(label="output folder destination", info= "--outdir", lines=1, value="/data/output", visible=True, show_label=True, interactive=True)
            dir_data_textbox = gr.Textbox(label="dataset zip folder destination", info="--data", lines=1, value="/app/Data/TESTDATASET.zip", visible=True, show_label=True, interactive=True)
            img_resolution_numbox = gr.Number(label="Resolution for webdatasets", info = "--img-resolution", minimum=8, value=256, scale = 1, interactive=True)
            batch_numbox = gr.Number(label="Total batch size", info = "--batch", minimum=1, value=56,scale = 1, interactive=True)
            batch_gpu_numbox = gr.Number(label="Limit batch size per GPU", info = "--batch-gpu", minimum=1, value=1, scale = 1, interactive=True)
        # G architecture args
        gr.Label("G Architecture Args")
        with gr.Row():
            config_radio = gr.Radio(choices=["custom","lite","full"], label="Base config.", info="--cfg", value="lite", interactive=True)
            cbase_numbox = gr.Number(label="Capacity multiplier", info ="--cbase", minimum=1, value=32768, interactive=True)
            cmax_numbox = gr.Number(label="Max. feature maps", info ="--cmax", minimum=1, value=512, interactive=True)
            res_blocks_numbox = gr.Number(label="Number of residual blocks", info ="--res-blocks", minimum=1, value=2, interactive=True)
        # Resuming.
        gr.Label("Resuming.")
        with gr.Row():
            resume_textbox = gr.Textbox(label="Resume from given network pickle", info= "--resume", lines=1, placeholder="network pickle file (maybe?)", visible=True, show_label=True, interactive=True)
            resume_numbox = gr.Number(label="Resume from given kimg", info ="--resume-kimg", minimum=0, value=0, interactive=True)
        # Training params.
        gr.Label("Training params.")
        with gr.Row():
            train_mode_radio = gr.Radio(choices=["all","text_encoder","freeze64"], label="Which layers to train", info="--train-mode", value="all", interactive=True)
            clip_weight_numbox = gr.Number(label="Loss weight for clip loss", info ="--clip-weight", value=0, interactive=True)
            init_blur_numbox = gr.Number(label="Init blur width", info ="--blur-init", minimum=0, value=32, interactive=True)
            blur_duration_numbox = gr.Number(label="Discriminator blur duration", info ="--blur-fade-kimg", minimum=0, value=1000, interactive=True)
        # Misc settings.
        gr.Label("Misc settings.")
        with gr.Row():
            suffix_textbox = gr.Textbox(label="Suffix of result dirname", info= "--suffix", lines=1, placeholder="suffix", visible=True, show_label=True, interactive=True)
            metrics_textbox = gr.Textbox(label="Quality metrics", info= "--metrics", lines=1, placeholder="Seperate by comma list: One,Two,Three", visible=True, show_label=True, interactive=True)
            duration_numbox = gr.Number(label="Total training duration", info ="--kimg", minimum=1, value=25000, interactive=True)
            progress_numbox = gr.Number(label="How often to print progress", info ="--tick", minimum=1, value=4, interactive=True)
            snap_numbox = gr.Number(label="How often to save snapshots", info ="--snap", minimum=0, value=5, interactive=True)
            seed_numbox = gr.Number(label="Random seed", info ="--seed", minimum=0, value=0, interactive=True)
            mix_precision_checkbox = gr.Checkbox(label="Disable mixed-precision", info="--fp32", value=False, interactive=True)
            benchmarking_checkbox = gr.Checkbox(label="Disable cuDNN benchmarking", info="--nobench", value=False, interactive=True)
            dataloader_numbox = gr.Number(label="DataLoader worker processes", info ="--workers", minimum=1, value=3, interactive=True)
            dry_run_checkbox = gr.Checkbox(label="Print training options and exit", info="--dry-run", value=True, interactive=True)

    with gr.Accordion("Toggle On/Off"):
            dir_output_toggle = gr.Checkbox(info="output folder destination", label= "--outdir", value=True, visible=True, show_label=True, interactive=False,scale = 0)
            dir_data_toggle = gr.Checkbox(info="dataset zip folder destination", label="--data", value=True, visible=True, show_label=True, interactive=False,scale = 0)
            img_resolution_toggle = gr.Checkbox(info="Resolution for webdatasets", label = "--img-resolution", value=True, scale = 0, interactive=False)
            batch_toggle = gr.Checkbox(info="Total batch size", label = "--batch", value=True,scale = 0, interactive=False)
            batch_gpu_toggle = gr.Checkbox(info="Limit batch size per GPU", label = "--batch-gpu", value=False, scale = 0, interactive=True)
            config_toggle = gr.Checkbox(info="Base config.", label="--cfg", value=True, interactive=True,scale = 0)
            cbase_toggle = gr.Checkbox(info="Capacity multiplier", label ="--cbase", value=False, interactive=True,scale = 0)
            cmax_toggle = gr.Checkbox(info="Max. feature maps", label ="--cmax", value=False, interactive=True,scale = 0)
            res_blocks_toggle = gr.Checkbox(info="Number of residual blocks", label ="--res-blocks", value=False, interactive=True,scale = 0)
            resume_toggle = gr.Checkbox(info="Resume from given network pickle", label= "--resume", value=False, visible=True, show_label=True, interactive=True,scale = 0)
            resume_num_toggle = gr.Checkbox(info="Resume from given kimg", label ="--resume-kimg", value=False, interactive=True,scale = 0)
            train_mode_toggle = gr.Checkbox(info="Which layers to train", label="--train-mode", value=False, interactive=True,scale = 0)
            clip_weight_toggle = gr.Checkbox(info="Loss weight for clip loss", label ="--clip-weight", value=False, interactive=True,scale = 0)
            init_blur_toggle = gr.Checkbox(info="Init blur width", label ="--blur-init", value=False, interactive=True,scale = 0)
            blur_duration_toggle = gr.Checkbox(info="Discriminator blur duration", label ="--blur-fade-kimg", value=False, interactive=True,scale = 0)
            suffix_toggle = gr.Checkbox(info="Suffix of result dirname", label= "--suffix", value=False, visible=True, show_label=True, interactive=True,scale = 0)
            metrics_toggle = gr.Checkbox(info="Quality metrics", label= "--metrics", value=False, visible=True, show_label=True, interactive=True,scale = 0)
            duration_toggle = gr.Checkbox(info="Total training duration", label ="--kimg", value=False, interactive=True,scale = 0)
            progress_toggle = gr.Checkbox(info="How often to print progress", label ="--tick", value=False, interactive=True,scale = 0)
            snap_toggle = gr.Checkbox(info="How often to save snapshots", label ="--snap", value=False, interactive=True,scale = 0)
            seed_toggle = gr.Checkbox(info="Random seed", label ="--seed", value=False, interactive=True,scale = 0)
            mix_precision_toggle = gr.Checkbox(info="Disable mixed-precision", label="--fp32", value=False, interactive=True,scale = 0)
            benchmarking_toggle = gr.Checkbox(info="Disable cuDNN benchmarking", label="--nobench", value=False, interactive=True,scale = 0)
            dataloader_toggle = gr.Checkbox(info="DataLoader worker processes", label ="--workers", value=False, interactive=True,scale = 0)
            dry_run_toggle = gr.Checkbox(info="Print training options and exit", label="--dry-run", value=False, interactive=True,scale = 0)

    with gr.Row():
        btn_generate = gr.Button("Train", scale=1, min_width=200)
        btn_generate.click(fn=build_command, inputs=[
            dir_output_textbox,
            dir_data_textbox,
            img_resolution_numbox,
            batch_numbox,
            batch_gpu_numbox,
            config_radio,
            cbase_numbox,
            cmax_numbox,
            res_blocks_numbox,
            resume_textbox,
            resume_numbox,
            train_mode_radio,
            clip_weight_numbox,
            init_blur_numbox,
            blur_duration_numbox,
            suffix_textbox,
            metrics_textbox,
            duration_numbox,
            progress_numbox,
            snap_numbox,
            seed_numbox,
            mix_precision_checkbox,
            benchmarking_checkbox,
            dataloader_numbox,
            dry_run_checkbox,
            dir_output_toggle,
            dir_data_toggle,
            img_resolution_toggle,
            batch_toggle,
            batch_gpu_toggle,
            config_toggle,
            cbase_toggle,
            cmax_toggle,
            res_blocks_toggle,
            resume_toggle,
            resume_num_toggle,
            train_mode_toggle,
            clip_weight_toggle,
            init_blur_toggle,
            blur_duration_toggle,
            suffix_toggle,
            metrics_toggle,
            duration_toggle,
            progress_toggle,
            snap_toggle,
            seed_toggle,
            mix_precision_toggle,
            benchmarking_toggle,
            dataloader_toggle,
            dry_run_toggle],
            outputs=[debug_label])
        
    gr.HTML("""
                <footer class="custom-footer">
                    <p>Developed By</p>
                    <p>Will Hoover | Temitayo Shorunke | Ethan Stanks</p>
                </footer>
            """)

demo.launch()