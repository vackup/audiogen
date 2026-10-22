# Repository Guidelines

## Project Structure

This repository contains a single executable Google Colab notebook:

- `acestep-1-5-xl-turbo-4b-colab-t4-gpu.ipynb` — installs the pinned Wan2GP environment, downloads AceStep model files, writes the Gradio app, and launches it.
- The notebook creates runtime-only files under `/content` in Colab, including `/content/Wan2GP` and `/content/run_acestep_xl.py`. These generated files are not repository source.
- There are currently no separate application modules, test directories, or checked-in model assets.

## Build, Test, and Development Commands

The project is notebook-driven; there is no local build system. In Colab:

1. Select a CUDA GPU runtime.
2. Run the notebook cells in order, from environment setup through launch.
3. Open the `gradio.live` URL printed by the final cell.

For local structural validation, parse the notebook as JSON and compile its Python cells. Dependency installation and inference require a Colab CUDA runtime.

The notebook pins Wan2GP to the known-good 15/09/2026 snapshot and verifies the expected versions of `mmgp`, `diffusers`, `transformers`, `numpy`, and `gradio`. Preserve those checks when changing the setup.

## Coding Style and Naming

Use Python with four-space indentation, `snake_case` for variables and functions, and `UPPER_SNAKE_CASE` for configuration constants such as `COLAB_ROOT` and `WAN2GP_COMMIT`. Keep notebook cells focused on one setup stage. Prefer explicit paths derived from `COLAB_ROOT`; keep model downloads idempotent and flush progress messages for long-running operations.

## Testing Guidelines

Before committing notebook changes, validate that the file is valid nbformat JSON and that all Python code cells parse. In Colab, verify GPU detection, dependency-version validation, model loading, the public Gradio URL, and one short audio generation. Record meaningful failures rather than treating a completed cell as successful inference.

## Commit and Pull Request Guidelines

Use concise, imperative commit subjects, for example: `Pin Colab dependencies to known-good Wan2GP snapshot`. Keep commits focused on the notebook. Pull requests should describe the runtime/dependency change, include the validation performed, and include the resulting Colab or Gradio output when behavior or UI changes.
