from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

# Cargar el tokenizador y el modelo
tokenizer = GPT2Tokenizer.from_pretrained('gpt2-xl')
model =   GPT2LMHeadModel.from_pretrained('gpt2-xl')


output_dir = './gpt2_xl_model'

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

