import gradio as gr
import numpy as np

from transformers import pipeline
from custom_chat_interface import CustomChatInterface

from llama_cpp import Llama
from llama_cpp.llama_chat_format import MoondreamChatHandler

"""
For more information on `huggingface_hub` Inference API support, please check the docs: https://huggingface.co/docs/huggingface_hub/v0.22.2/en/guides/inference
"""


class MyModel:
    def __init__(self):
        self.client = None
        self.current_model = ""

    def respond(
        self,
        message,
        history: list[tuple[str, str]],
        model,
        system_message,
        max_tokens,
        temperature,
        top_p,
    ):
        if model != self.current_model or self.current_model is None:
            model_id, filename = model.split(",")
            client = Llama.from_pretrained(
                repo_id=model_id.strip(),
                filename=f"*{filename.strip()}*.gguf",
                n_ctx=2048,  # n_ctx should be increased to accommodate the image embedding
            )

            self.client = client
            self.current_model = model

        messages = [{"role": "system", "content": system_message}]

        for val in history:
            if val[0]:
                messages.append({"role": "user", "content": val[0]})
            if val[1]:
                messages.append({"role": "assistant", "content": val[1]})

        messages.append({"role": "user", "content": message})

        response = ""
        for message in self.client.create_chat_completion(
            messages,
            temperature=temperature,
            top_p=top_p,
            stream=True,
            max_tokens=max_tokens,
        ):
            delta = message["choices"][0]["delta"]
            if "content" in delta:
                response += delta["content"]
                yield response


transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")


def transcribe(audio):
    sr, y = audio

    # Convert to mono if stereo
    if y.ndim > 1:
        y = y.mean(axis=1)

    y = y.astype(np.float32)
    y /= np.max(np.abs(y))

    text = transcriber({"sampling_rate": sr, "raw": y})["text"]
    return text


"""
For information on how to customize the ChatInterface, peruse the gradio docs: https://www.gradio.app/docs/chatinterface
"""
my_model = MyModel()
model_choices = [
    "lab2-as/lora_model_gguf, Q4",
    "lab2-as/lora_model_no_quant_gguf, Q4",
    "lab2-as/lora_model_math_optimized_gguf, Q4",
]
demo = CustomChatInterface(
    my_model.respond,
    transcriber=transcribe,
    additional_inputs=[
        gr.Dropdown(
            choices=model_choices,
            value=model_choices[0],
            label="Select Model",
        ),
        gr.Textbox(
            value="You are a chatbot with a proficiency in math. You are to answer the mathematical equations correctly and efficiently. You are to reason and explain your solutions thoroughly.",
            label="System message",
        ),
        gr.Slider(
            minimum=1,
            maximum=2048,
            value=512,
            step=1,
            label="Max new tokens",
        ),
        gr.Slider(
            minimum=0.1,
            maximum=4.0,
            value=0.4,
            step=0.1,
            label="Temperature",
        ),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.15,
            step=0.05,
            label="Top-p (Nucleus sampling)",
        ),
    ],
)


if __name__ == "__main__":
    demo.launch()
