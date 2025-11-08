from transformers import pipeline

def generate_card(prompt):
    """
    Uses a pre-trained language model to generate text based on a prompt.
    """
    # 1. Load the text generation pipeline
    # The 'text-generation' pipeline handles tokenization, model loading, and output decoding.
    generator = pipeline("text-generation", model="distilgpt2")

    # 2. Configure generation parameters
    # The 'prompt' is the text the model starts with.
    # The 'max_length' controls how long the generated card will be.
    # The 'num_return_sequences' controls how many options the model generates.
    
    # We ask for 3 options and cap the length at about 20-30 words.
    results = generator(
        prompt,
        max_length=40,  # Keep it short like a real card
        num_return_sequences=3,
        # Stop generation at the first newline to keep it clean
        pad_token_id=generator.tokenizer.eos_token_id 
    )

    # 3. Print the results
    print(f"\n--- Generated Card Options for Prompt: '{prompt}' ---\n")
    for i, result in enumerate(results):
        # We strip the original prompt for a cleaner display
        generated_text = result['generated_text'].replace(prompt, '').strip()
        print(f"Card {i+1}: {generated_text}")
        
    print("\n------------------------------------------------\n")

# --- Run the Generator ---

# Use a prompt that mimics the style of a Monopoly card.
chance_prompt = "Chance: Pay poor tax of "
community_chest_prompt = "Community Chest: Bank error in your favor. Collect "

generate_card(chance_prompt)
generate_card(community_chest_prompt)

# Example of a themed prompt for "Hackopoly"
hackathon_prompt = "Hackopoly Chance: Your code compiled successfully. Collect prize money of "
generate_card(hackathon_prompt)