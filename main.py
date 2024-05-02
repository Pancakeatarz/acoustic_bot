import asyncio
import os
import random

import discord
import requests
import json
from pygelbooru import Gelbooru
import asyncio
from apicalls import *

# Get token from .env file
token = os.environ['TOKEN']

# Define intents
intents = discord.Intents.default()
intents.message_content = True

# Initialize client
client = discord.Client(intents=intents)
  
# Bot login message
@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

# Verify if message is from the bot
@client.event
async def on_message(message):
  if message.author == client.user:
    return# ignore if from the bot

# Check if the message is a command
  if message.content.startswith("$"):
    command = message.content
# Check if the command is $fart
    if command == "$fart":
      if message.author.voice is None:
        await message.channel.send(f"{message.author.mention} you are not in a voice channel")
      else:
        voice_channel = message.author.voice.channel
        voice_client = await voice_channel.connect()  # connect 
        url = "fart_sound.mp3" 
        source = discord.FFmpegPCMAudio(url)
        voice_client.play(source) #fart 
        await message.channel.send("Farting...")
        await asyncio.sleep(3)
        await voice_client.disconnect() # disconect

# check if its a neko command
    elif command in commandsNeko:
      neko = get_nekos_neko(command)
      await message.channel.send(neko)

# Check if it's a Gelbooru command
    elif message.content.startswith("$gelbooru"):
      tags = message.content[10:]
      dnbr = await get_gelbooru_image(tags)
      await message.channel.send(dnbr)

# check if its a translate command
    elif message.content.startswith("$recognize"):
      image_url = message.attachments[0].url  # Assuming the image is sent as an attachment
      result = recognize_image(image_url)
      await message.channel.send(f"Image recognition result for {image_url}: {result}")
  
# Makima messages
  if "makima" in message.content.lower():
    await message.channel.send(random.choice(mkmMessageList))

# Periodically clean up old URLs stored in db (e.g., once a day) 
clean_up_old_urls(threshold=24 * 60 * 60)  # 24 hours

# Run the client
client.run(token)
##no code after this because it wouldn't work
