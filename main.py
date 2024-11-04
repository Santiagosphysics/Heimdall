import os 
import pygame 
import speech_recognition as sr 
import torch
from gtts import gTTS


from transformers import GPT2Tokenizer, GPT2LMHeadModel


def speak(text):

    voice_1 = "en-US-JessaNeural"
    voice_2 = "en-US.AriaNeural"
    voice_3 = "en-US-GuyNeural"

    voice_4 = "es-CL-CatalinaNeural"
    voice_5 = "es-MX-DaliaNeural"
    voice_6 = "es-VE-SebastianNeural"
    voice_7 = "es-US-AlonsoNeural"
    voice_8 = "es-ES-ManuelEsCUNeural"
    # command = f'edge-tts --voice "{voice_1}" --text "{text}" --write-media "output.mp3"'
    # os.system(command)
    
    tts = gTTS(text=text, lang='en')
    tts.save("output.mp3")
    pygame.init()
    pygame.mixer.init()

    try:
        pygame.mixer.music.load("output.mp3")
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(100)

    except Exception as e:
        print(e)
    
    finally:
        pygame.mixer.music.stop()
        pygame.mixer.quit()
    


def take_command():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("listening...")
        r.pause_threshold = 3
        audio = r.listen(source)
    
    try:
        print("Recognizing...")
        # query = r.recognize_google(audio, language='es-col')
        query = r.recognize_google(audio, language='en-US')


    except Exception as e:
        print(e)
        return ""
    return query


def generate_text(prompt):

    output_dir = './gpt2_model'
    tokenizer = GPT2Tokenizer.from_pretrained(output_dir)
    model = GPT2LMHeadModel.from_pretrained(output_dir).to('cuda')
    gpu_name = torch.cuda.get_device_name(0)
    print(f'GPU en uso {gpu_name}')

    print('User:', prompt, '\n')
    inputs = tokenizer.encode(prompt, return_tensors='pt').to('cuda')
    attention_mask = torch.ones(inputs.shape, dtype=torch.long).to('cuda')  # Crear la máscara de atención
    
    output = model.generate(
        inputs,
        attention_mask=attention_mask,  # Pasar la máscara de atención
        max_length=50,
        num_return_sequences=1,
        do_sample=True,            # Activa el muestreo
        temperature=0.05,           # Controla la aleatoriedad
        top_k=10,                  # Limita a los 50 tokens más probables
        top_p=0.6,
        repetition_penalty=1.9,               
        pad_token_id=tokenizer.eos_token_id  # Token de fin de respuesta
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)


while True:
    query = take_command().lower()
    if query:
        if 'finalizar' in query or 'fin' in query or 'finish' in query:
            speak('Que tengas un buen día')
            break
        else:
            result = generate_text(query)
            print(result)
            speak(text = result)
