from src.demo.app_utils import gradio_init


if __name__ == "__main__":
    demo = gradio_init(explanation=False)
    demo.launch(share=True, debug=True)
