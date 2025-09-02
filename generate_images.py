import pandas as pd
import openai
import requests
from PIL import Image
from io import BytesIO
from openai import OpenAI
import openai
import os
import re
import emoji

def clean_text(post):
    # Convert emojis to text descriptions
    post = emoji.demojize(post, delimiters=(" ", " "))  # "😢" → " crying face emoji "
    
    # Allow useful punctuation and symbols
    allowed_chars = r'[^a-zA-Z0-9\s.,!?…()"\'\[\]:;_—\-=$%]'  
    
    # Remove only unwanted symbols
    post = re.sub(allowed_chars, '', post)  
    
    # Normalize whitespace
    post = re.sub(r'\s+', ' ', post).strip()
    
    return post  # Keep case sensitivity

# Define the function to generate an image based on stress indication
def generate_image(post, response, answer, index, output_folder, output_folder_negative):
    client = OpenAI(api_key='.....')
    
    if answer == "yes":
        user_text = f"""
            Based on the following text: "{post}", generate a **consistent, structured** image that visually represents a state of stress or anxiety. 
            Ensure the image includes:
            - A tense or overwhelming environment (e.g., dim lighting, clutter, urban stress).
            - Facial expressions that indicate worry, exhaustion, or distress (if humans are depicted).
            - A darker, cooler color palette to evoke a stressed mood.
            """

        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=user_text,
                size="1024x1024",
                quality="standard",
                n=1,
            )
        except openai.BadRequestError as e:
            print(f"Skipping index {index} due to content policy violation.")
            with open("rejected_prompts_prompt_new_img.log", "a") as f:
                f.write(f"Index {index}: {post}\n")
            return  # Skip this iteration instead of stopping inference

        # Get the image URL
        image_url = response.data[0].url

        # Fetch the image from the URL
        image_response = requests.get(image_url)

        # Open the image
        image = Image.open(BytesIO(image_response.content))

        # # Display the image
        # image.show()

        # Save the image with the corresponding index
        image_path = os.path.join(output_folder, f"image_{index}.png")
        image.save(image_path)
        print(f"Image saved at {image_path}")

        #### do the same for the negative
        user_text = f"""
            Based on the following text: "{post}", generate a **consistent, structured** image that visually represents a calm, relaxed, and stress-free state. 
            Ensure the image includes:
            - A bright, cheerful environment (e.g., nature, sunshine, smiling individuals).
            - No elements that depict stress, tension, or anxiety.
            - A harmonious color palette, using soft and warm tones.
            """

        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=user_text,
                size="1024x1024",
                quality="standard",
                n=1,
            )
        except openai.BadRequestError as e:
            print(f"Skipping index {index} due to content policy violation.")
            with open("rejected_prompts_prompt_new_img_negative.log", "a") as f:
                f.write(f"Index {index}: {post}\n")
            return  # Skip this iteration instead of stopping inference

        # Get the image URL
        image_url = response.data[0].url

        # Fetch the image from the URL
        image_response = requests.get(image_url)

        # Open the image
        image = Image.open(BytesIO(image_response.content))

        # # Display the image
        # image.show()

        # Save the image with the corresponding index
        image_path = os.path.join(output_folder_negative, f"image_{index}.png")
        image.save(image_path)
        print(f"Image saved at {image_path}")


    elif answer == "no":
        user_text = f"""
            Based on the following text: "{post}", generate a **consistent, structured** image that visually represents a calm, relaxed, and stress-free state. 
            Ensure the image includes:
            - A bright, cheerful environment (e.g., nature, sunshine, smiling individuals).
            - No elements that depict stress, tension, or anxiety.
            - A harmonious color palette, using soft and warm tones.
            """

        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=user_text,
                size="1024x1024",
                quality="standard",
                n=1,
            )
        except openai.BadRequestError as e:
            print(f"Skipping index {index} due to content policy violation.")
            with open("rejected_prompts_prompt_new_img.log", "a") as f:
                f.write(f"Index {index}: {post}\n")
            return  # Skip this iteration instead of stopping inference

        # Get the image URL
        image_url = response.data[0].url

        # Fetch the image from the URL
        image_response = requests.get(image_url)

        # Open the image
        image = Image.open(BytesIO(image_response.content))

        # # Display the image
        # image.show()

        # Save the image with the corresponding index
        image_path = os.path.join(output_folder, f"image_{index}.png")
        image.save(image_path)
        print(f"Image saved at {image_path}")

        #### do the same for the negative
        user_text = f"""
            Based on the following text: "{post}", generate a **consistent, structured** image that visually represents a state of stress or anxiety. 
            Ensure the image includes:
            - A tense or overwhelming environment (e.g., dim lighting, clutter, urban stress).
            - Facial expressions that indicate worry, exhaustion, or distress (if humans are depicted).
            - A darker, cooler color palette to evoke a stressed mood.
            """

        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=user_text,
                size="1024x1024",
                quality="standard",
                n=1,
            )
        except openai.BadRequestError as e:
            print(f"Skipping index {index} due to content policy violation.")
            with open("rejected_prompts_prompt_new_img_negative.log", "a") as f:
                f.write(f"Index {index}: {post}\n")
            return  # Skip this iteration instead of stopping inference

        # Get the image URL
        image_url = response.data[0].url

        # Fetch the image from the URL
        image_response = requests.get(image_url)

        # Open the image
        image = Image.open(BytesIO(image_response.content))

        # # Display the image
        # image.show()

        # Save the image with the corresponding index
        image_path = os.path.join(output_folder_negative, f"image_{index}.png")
        image.save(image_path)
        print(f"Image saved at {image_path}")
    else:
        print('The answer has NO valid entry!')
        
# Example usage
if __name__ == "__main__":

    # Define the file path
    file_path = './dreaddit-train-2.csv'

    # Define the output folder for images
    output_folder = './dreadit_images_prompt_new/train/images'

    # Define the output folder for negative images
    output_folder_negative = './dreadit_images_prompt_new/train/negative_images'

    # Read the CSV file with pandas
    data = pd.read_csv(file_path)

    # Iterate through each row of the DataFrame
    for index, row in data.iterrows():
        if index == 102:
        # if index == 125:
            # Extract the 'post' and 'response'
            post = row['post']
            response = row['response']
            
            # Check if the response contains "yes" or "no"
            if "yes" in response.lower():
                answer = "yes"
            elif "no" in response.lower():
                answer = "no"
            else:
                answer = "unknown"

            # # Sample input
            # post = "I'm feeling very overwhelmed with work and personal issues."
            # response = "yes. Reasoning: The poster's language reflects significant stress."
            
            # Print the extracted post and answer
            print(f"The index is: {index}")
            # print(f"Post: {post}")
            cleaned_post = clean_text(post)
            # print(f"Post: {cleaned_post}")
            # print(cleaned_post == post)
            # print(f"Answer: {answer}")
            # print(f"Response: {response}")
            print("-" * 50)

            generate_image(cleaned_post, response, answer, index, output_folder, output_folder_negative)

            # input()
