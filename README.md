# Fine Tuning LLM

## Improve model performance

### Model-centric approach

To improve performace you could use Full Fine-Tuning which retrains all parameters. This however uses a lot more computational resources.

When using LoRA, using quantization reduces performace and memory usage so to improve performance we could not use 4bit quantization.
Another technique we could use to possible improve performace is use rank-stabilized LoRA.

We could further experiment with different learning rates, learning rate schedulers and batch size, we could also adjust the rank (r) and scaling factor (alpha) for the injected LoRA layers. 

Lastly, pre-prompt engineering is important. 
A well designed pre-promt can significantly improve model performance by allowing the model to gain more context of the user query.

### Data-centric approach

Our models uses a small subpart of a larger dataset so using the full 1.75M rows [The tome](https://huggingface.co/datasets/arcee-ai/The-Tome) should improve performace.

Also if we want more specialized knowledge we could use other specialized datasets.

