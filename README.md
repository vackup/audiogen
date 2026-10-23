# AceStep 1.5 XL Turbo — Google Colab

This repository provides a Google Colab notebook for running the **AceStep 1.5 XL Turbo 4B** music-generation model through Wan2GP and a Gradio interface. It is optimized for a single Tesla T4-class GPU using INT8 quantization and CPU/RAM offloading.

## Requirements

- Google Colab with a CUDA GPU runtime, preferably Tesla T4 or better
- Internet access enabled
- Approximately 7 GB of model downloads
- Enough available Colab disk space and RAM

The notebook creates all runtime files under `/content`; Colab runtimes are temporary.

## Quick Start

1. Open `acestep-1-5-xl-turbo-4b-colab-t4-gpu.ipynb` in Google Colab.
2. Select **Runtime → Change runtime type → T4 GPU**.
3. Restart the runtime before the first run if packages from an earlier attempt are loaded.
4. Run the cells in order:
   - Environment setup
   - Dependency installation
   - Model download
   - Gradio app generation
   - Application launch
5. Open the `https://*.gradio.live` URL printed by the launch cell.

The first generation may take longer while Triton compiles its kernels. Use a short duration for the initial smoke test.

## Architecture and Execution Flow

The notebook is the deployment orchestrator. It prepares the Colab runtime, downloads the model assets, generates the application script, and starts Gradio in the same Python process as the model.

```text
Browser
  │ public gradio.live URL
  ▼
Gradio UI → generate_music() → _generate_single()
                                  │
                                  ▼
                         ace_model.generate()
                                  │
                    AceStep + Wan2GP + mmgp
                                  │
                                  ▼
                         CUDA GPU / T4
```

The notebook performs five stages:

1. **Environment setup** — detects the CUDA GPU and configures memory-related environment variables.
2. **Dependency installation** — checks out the pinned Wan2GP commit, installs its requirements, installs FFmpeg, and pins `torchcodec==0.11.1` for the current Torch 2.11 Colab runtime.
3. **Model download** — stores AceStep checkpoints and encoder files under `/content/Wan2GP/models`.
4. **Application generation** — writes `/content/run_acestep_xl.py`, which loads the model with `family_handler.load_model()` and configures mmgp offloading.
5. **Application launch** — calls `demo.launch(share=True, server_name="0.0.0.0")` and prints the public Gradio URL.

The model is loaded once at application startup and remains in memory. Gradio requests call `ace_model.generate()` directly; there is no separate model server or HTTP service between Gradio and AceStep. In Cover Mode, the uploaded reference audio is decoded through `torchaudio`, TorchCodec, and FFmpeg before being passed to the model. Generated tensors are written to temporary WAV/MP3 files and returned to Gradio for playback.

## Reproducibility

Wan2GP is pinned to commit `38ed31cdfb040d76d56993ccbc9bf73a06fc6aaf`, the snapshot used for the original working setup. The notebook installs and verifies these key packages:

```text
mmgp==3.8.0
diffusers==0.36.0
transformers==4.54.0
numpy==2.1.2
gradio==5.29.0
torchcodec==0.11.1
```

FFmpeg is installed through the Colab system package manager because Cover Mode requires its shared libraries for reference-audio decoding.

Do not add an unversioned `pip install mmgp gradio` command after the requirements installation; it can silently upgrade the working stack.

## Notebook Runtime Layout

- `/content/Wan2GP` — pinned Wan2GP checkout
- `/content/Wan2GP/models` — downloaded checkpoints
- `/content/run_acestep_xl.py` — generated Gradio application

## Troubleshooting

- **No GPU detected:** select a CUDA accelerator and restart the runtime.
- **Dependency drift detected:** restart the runtime and rerun the notebook from the beginning.
- **Triton `bitcast` compilation error:** verify that the pinned commit and package versions are installed; do not update Wan2GP to `main` during troubleshooting.
- **No public URL:** confirm that the launch cell is still running and inspect the output for the first Python traceback.

## Development Validation

For notebook-only changes, validate the file as nbformat JSON and parse every Python code cell. Full model loading and audio generation require a Colab CUDA runtime.

