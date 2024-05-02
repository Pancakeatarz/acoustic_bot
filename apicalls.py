import os
import time
from io import BytesIO

import requests
import torch
import torchvision.models as models
from PIL import Image
from pygelbooru import Gelbooru
from replit import db
from torchvision import transforms
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


##db
##
# Function to add URL to the database
def add_url(url):
  db[url] = time.time()  # Get current timestamp
    # Store URL and timestamp in the database

  # Function to clean up old URLs
def clean_up_old_urls(threshold):
  current_time = time.time()
  for url, timestamp in db.items():
    if current_time - timestamp > threshold:
      del db[url]  # Remove URL if older than threshold
    # Query the database for URLs and their timestamps
    # Iterate over URLs and remove those older than the threshold


##nekko
###
# Define neko commands dictionary
commandsNeko = {
    "$neko": "https://nekos.best/api/v2/neko",
    "$baka": "https://nekos.best/api/v2/baka",
    "$hug": "https://nekos.best/api/v2/hug",
    "$kiss": "https://nekos.best/api/v2/kiss",
    "$pat": "https://nekos.best/api/v2/pat",
    "$slap": "https://nekos.best/api/v2/slap",
    "$smug": "https://nekos.best/api/v2/smug",
    "$tickle": "https://nekos.best/api/v2/tickle",
    "$feed": "https://nekos.best/api/v2/feed",
    "$cuddle": "https://nekos.best/api/v2/cuddle",
    "$smile": "https://nekos.best/api/v2/smile",
    "$waifu": "https://nekos.best/api/v2/waifu",
    "$bite": "https://nekos.best/api/v2/bite",
    "$blush": "https://nekos.best/api/v2/blush",
    "$bored": "https://nekos.best/api/v2/bored",
    "$dance": "https://nekos.best/api/v2/dance",
    "$cry": "https://nekos.best/api/v2/cry",
    "$happy": "https://nekos.best/api/v2/happy",
    "$highfive": "https://nekos.best/api/v2/highfive",
    "$laugh": "https://nekos.best/api/v2/laugh",
    "$punch": "https://nekos.best/api/v2/punch",
    "$shrug": "https://nekos.best/api/v2/shrug",
    "$stare": "https://nekos.best/api/v2/stare",
}
# Function to get neko from API
def get_nekos_neko(command):
    resp = requests.get(commandsNeko[command])
    data = resp.json()
    return data["results"][0]["url"]


##gelbooru
##
api_key = os.environ['API_KEY'] # api key from .env
username = os.environ['USERNAME'] # username from .env

gelbooru = Gelbooru(api_key, username)# gelborru api call with pygelbooru


# Function to get random gelbooru from API
async def get_gelbooru_image(tags): 
  tags_list = tags.split()
  result = await gelbooru.random_post(tags=tags_list, exclude_tags=['nude','loli','shota'])
  return result


##makima
## 
# Messages list for Makima ( not really api calls )
mkmMessageList = [
    ":pray::pray::pray:",
    "we love makima ! :pray:",
    "woof woof ^^",
    ":heart_eyes::heart_eyes::heart_eyes:",
    "smash",
    "https://tenor.com/bC7cl.gif",
    "https://tenor.com/qdV56jU2BJt.gif",
    "shut up nigger chan ^^",
    "yay",
    "based",
    "https://tenor.com/b1I86.gif",
    "https://tenor.com/b1ddd.gif",
]  

##Translate
##
# Function to perform image recognition and generate text descriptions
def recognize_image(image_url):
    # Load the pretrained image recognition model
    model = models.resnet50(pretrained=True)
    model.eval()

    # Load the pretrained NLP model
    nlp_model = GPT2LMHeadModel.from_pretrained('gpt2')  # Load your pre-trained NLP model

    # Define image preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Download and preprocess the image
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))

    # Perform image inference
    with torch.no_grad():
        # Convert Image to PyTorch tensor before adding batch dimension
        image_tensor = preprocess(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        image_features = model(image_tensor)
      # Generate text descriptions using the NLP model and the image features
        input_ids = tokenizer.encode("The image shows:", return_tensors="pt")
        text_descriptions = nlp_model.generate(input_ids=input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2)


    # Return the generated text descriptions
    return text_descriptions