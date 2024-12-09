# Fine Tuning LLM

## UI and Fine Tuned Llama 3.2 1B
We fine tuned the Llama 3.2 1B model using unsloth and lora adapters. 

As a user interface we used huggingface spaces with the gradio framework and llama-cpp python framework for model inference.
The user interface consists of a chatbox along with audio input for your prompt.

The models are uploaded in the GGUF format which is compatible with the llama-cpp framework.

###
The user interface is available [here](link_to_uiXXXXXXXXXXXXXXXx).

## Improve model performance

### Model-centric approach

To improve performace you could use Full Fine-Tuning which retrains all parameters. This however uses a lot more computational resources.

When using LoRA, using quantization reduces performace and memory usage so to improve performance we could not use 4bit quantization.
Another technique we could use to possible improve performace is use rank-stabilized LoRA.

We could further experiment with different learning rates, learning rate schedulers and batch size, we could also adjust the rank (r) and scaling factor (alpha) for the injected LoRA layers. 

One could also use a model with more parameters if there's enough computational resource available. 
As seen in recent years, more parameters usually improves model performance at the cost of computational resources required.

Lastly, pre-prompt engineering is important. 
A well designed pre-promt can improve model performance by allowing the model to gain more context of the user query.

### Data-centric approach

Our models uses a small subpart of a larger dataset so using the full 1.75M rows [The tome](https://huggingface.co/datasets/arcee-ai/The-Tome) should improve performace.

If we want more specialized knowledge we could use other specialized datasets.

### Testing
We tested our model on a few math questions to see how well it performed, the following questions were answered wrongly.

Question one: `Ken can do 20 sit-ups without stopping. Nathan can do twice as many, and Bob can do half the number of Ken and Nathan's combined sit-ups. How many more sit-ups can Bob do compared to Ken?` (answer=10) 

Question two: `(50+60) * (50 * 3/60)=` (answer = 275)

Question three: `7*x+10*y=17339 ; x+y=2000` (answer x=887, y=1113)

In order to solve this we fine tuned the model using a [math](https://huggingface.co/datasets/Macropodus/MWP-Instruct) dataset which contained about 250'000 math equations and solutions. 
This improved the accuracy of the model, allowing it to answer all of our questions correctly at least 1 in 5 times. 

