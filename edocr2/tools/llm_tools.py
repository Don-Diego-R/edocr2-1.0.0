from dotenv import load_dotenv
import cv2, os, ast, io, base64
from PIL import Image

def call_VL(model, processor, device, messages):
    
    from qwen_vl_utils import process_vision_info
    
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    cleaned_output = output_text[0].strip('```python\n```')
    # Convert the cleaned string into a dictionary
    return ast.literal_eval(cleaned_output)

def load_VL(model_name = "Qwen/Qwen2-VL-7B-Instruct"):
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype="auto", device_map="auto")
    
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

def llm_dim(llm, img, device, scale = 1):
    resized_img = cv2.resize(img, (int(img.shape[1]*scale), int(img.shape[0]*scale)), interpolation=cv2.INTER_AREA)
    img = Image.fromarray(resized_img)

    messages = [
        {"role": "system",
            "content": [{"type": "text", "text": '''You are a specialized OCR system capable of reading mechanical drawings. You read:
                        Measurements, usually scattered and oriented text in the image and with arrows in the surroundings. If tolerances are present, read them as "nominal" "upper" "lower". e.g: "10 +0.1 0"
                        Angles, usually oriented text with arrows in the surroundings
                        Do not include surface finishes'''},],
        },
        {"role": "user",
            "content": [{"type": "image","image": img,},
                        {"type": "text", "text": "Based on the image, return ONLY A PYTHON LIST OF STRINGS extracting dimensions"},],
        }]
    output_text = call_VL(model=llm[0], processor=llm[1], device = device, messages=messages)
    print(output_text)
    return output_text

def llm_table(tables, llm, img, device, query):
    for b in tables[0]:
        tab_img = img[b.y : b.y + b.h, b.x : b.x + b.w][:]
    tab_img = Image.fromarray(tab_img)

    query_string = ', '.join(query)
    messages = [
        {"role": "user",
            "content": [{"type": "image","image": tab_img,},
                        {"type": "text", "text": f"Based on the image, return only a python dictionary extracting this information: {query_string}"},],
        }]
    
    llm_dict = call_VL(model=llm[0], processor=llm[1], device = device, messages = messages)
    return llm_dict

def gpt4_dim(img):

    def convert_img(img):
        pil_img=Image.fromarray(img)
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG')
        byte_im = buf.getvalue()
        return base64.b64encode(byte_im).decode('utf-8')

    from openai import OpenAI
    
    load_dotenv()
    API_KEY = os.getenv("OPENAI_API_KEY")
    client = OpenAI(api_key=API_KEY)
    img_ = convert_img(img)

    messages = [
        {"role": "system",
            "content": [{"type": "text", "text": '''You are a specialized OCR system capable of reading mechanical drawings. You read:
                        Measurements, usually scattered and oriented text in the image and with arrows in the surroundings. If tolerances are present, read them as "nominal" "upper" "lower". e.g: "10 +0.1 0"
                        Angles, usually oriented text with arrows in the surroundings
                        Feature Control Frames, usually in boxes, return either the symbol or its description, then the rest of the text'''},],
        },
        {"role": "user",
            "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_}", "detail": "high"}},
                        {"type": "text", "text": "Based on the image, return ONLY A PYTHON LIST OF STRINGS extracting dimensions"},],
        }]
    
    response = client.chat.completions.create(model="gpt-4o", messages=messages, max_tokens=3000)
    assistant_response=response.choices[0].message.content
    cleaned_output = assistant_response.strip('```python\n```')
    # Convert the cleaned string into a dictionary
    return ast.literal_eval(cleaned_output)
