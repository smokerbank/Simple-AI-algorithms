from transformers import GPT2LMHeadModel, GPT2Tokenizer

class SimpleLLM:
    def __init__(self, model_name='gpt2'):
        # Load pre-trained model and tokenizer from Hugging Face
        self.model_name = model_name
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    def generate_text(self, prompt, max_length=50):
        # Tokenize input prompt and generate text
        inputs = self.tokenizer.encode(prompt, return_tensors='pt')
        outputs = self.model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2)
        
        # Decode the generated output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text

if __name__ == "__main__":
    # Create an instance of the SimpleLLM class
    llm = SimpleLLM()

    # Input prompt for text generation
    prompt = "I love AI because"
    
    # Generate and print the text
    generated_text = llm.generate_text(prompt)
    print("Generated Text: ", generated_text)