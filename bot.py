import os
from deepface import DeepFace
import numpy as np
import multiprocessing as mp
import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes,MessageHandler, filters, Updater,CallbackContext
import pandas as pd 
from fpdf import FPDF
import cv2 as cv
from urllib.request import urlopen, HTTPError

data=pd.read_csv('new.csv')


# Load environment variables from the .env file
load_dotenv()
TOKEN = os.getenv('TOKEN')

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(update.message.text)

async def Help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Help')
async def website(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text('Website')

async def download_image(update, context):
    chat_id = update.effective_chat.id
    message = update.message
    if message.photo:
        file_id = message.photo[-1].file_id
        file_info = await context.bot.get_file(file_id)
        file_path = file_info.file_path
        response = requests.get(file_path, stream=True)
        with open('downloads/photototest.jpg', 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
 #pool

async def face_detection(update:Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    message = update.message
    if message.photo:
        file_id = message.photo[-1].file_id
        file_info = await context.bot.get_file(file_id)
        file_path = file_info.file_path
        response = requests.get(file_path, stream=True)
        img_path="downloads/photototest.jpg"
        with open('downloads/photototest.jpg', 'wb') as f:
            for chunk in response.iter_content(1024):
                f.write(chunk)
    best_match=[]
    index_value=[]
    for i in range(len(data['Student Photo'])):
        try:
            img=urlopen(data['Student Photo'][i]).read()
            test_img_path="target{}.jpg".format(data['Roll Numbers'][i])
            open(test_img_path, 'wb').write(img)
            result=DeepFace.verify(img_path,test_img_path)
            output=result['verified']
            distance_value=result['distance']
            if output==True:
                best_match.append(distance_value)
                index_value.append(i)
            else:
                continue
            os.remove(test_img_path)
        except:
            continue

        
    arr_best_match=np.array(best_match)
    sorted_arr=np.sort(arr_best_match)[:10]
    list_best_match=sorted_arr.tolist()
    for i in range(len(list_best_match)):
        x=best_match.index(list_best_match[i])
        data_index=index_value[x]

        await update.message.reply_text(data['Names'][data_index])
        print(data_index)
    return data['Names'][data_index]




        

      













# #Pre trained Models
# import torch
# import torchvision.transforms as transforms
# from torchvision.models import vgg16
# from PIL import Image

# # Load a pre-trained VGG model
# model = vgg16(pretrained=True).eval()

# # Define the transformation
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# def get_features(image_path, model, transform):
#     image = Image.open(image_path)
#     image = transform(image).unsqueeze(0)
#     with torch.no_grad():
#         features = model(image)
#     return features

# # Load and transform the images
# features1 = get_features('photototest.jpg', model, transform)
# # features2 = get_features('dp.jpg', model, transform)
# data_feature=[]
# for i in range(len(data['Student Photo'])):
#     try:
#         img=urlopen(data['Student Photo'][i]).read()
#         img_path="target{}.jpg".format(data['Roll Numbers'][i])
#         open(img_path, 'wb').write(img)
#         features = get_features(img_path, model, transform)
#         os.remove(img_path)
#         cosine_similarity = torch.nn.functional.cosine_similarity(features1, features)
#         if cosine_similarity.item()>0.75:
#             print(data['Names'][i]+f"Cosine Similarity: {cosine_similarity.item()}")
#     except:
#         continue

# Compute the cosine similarity
# try:
#     for features2 in range(data_feature):
#         cosine_similarity = torch.nn.functional.cosine_similarity(features1, features2)
#         if cosine_similarity.item()>0.78:
#             print(data['Names'][features2]+f"Cosine Similarity: {cosine_similarity.item()}")
# except:
#     print("error")
# # cosine_similarity = torch.nn.functional.cosine_similarity(features1, features2)
# print(f"Cosine Similarity: {cosine_similarity.item()}")


#To handle wing
async def handle_wing(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    names = ""
    input = update.message.text
    input = input[:len(input)-1] + '-' + input[-1]
    pdf=FPDF()
    for i in range(len(data['Roll Numbers'])):
        if(input == data['Address'][i][:len(input)]):
            names = names + data['Names'][i] +"("+data['Address'][i].split(',')[0]+")"+ "\n"
            print(data['Names'][i])
            pdf.add_page()
            img=urlopen(data['Student Photo'][i]).read()
            img_path="downloads/target{}.jpg".format(data['Roll Numbers'][i])
            open(img_path, 'wb').write(img)
            pdf.image(img_path, x= 50, y=50, w=120)
            os.remove(img_path)
    pdf.output("Photos{}.pdf".format(update.message.chat_id))
    await context.bot.send_document(chat_id=update.message.chat_id,document="Photos{}.pdf".format(update.message.chat_id),caption=names)
    # await update.message.reply_text(names)


#To handle roll No.
async def handle_roll_no(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        roll_no=update.message.text.lstrip('/')
        print(roll_no)
        index_value=0
        if roll_no.isdigit() and len(roll_no)==6:
                for i in range(len(data['Roll Numbers'])):
                    if str(data['Roll Numbers'][i])==roll_no:
                        sliced = data['Address'][i].split(',')
                        chat_id=update.message.chat_id
                        if i==1045:
                            img_path="my_dp.jpg"
                            await context.bot.send_photo(chat_id=chat_id, photo=img_path, caption=f"""Student Name:-  {data['Names'][i]}\nStudent Roll:-  {data['Roll Numbers'][i]}\nStudent Room No.:-  {sliced[0]}\nStudent Hall :- {sliced[1]}""")
                        else:
                            img=urlopen(data['Student Photo'][i]).read()
                            img_path="target{}.jpg".format(data['Roll Numbers'][i])
                            open(img_path, 'wb').write(img)
                            await context.bot.send_photo(chat_id=chat_id, photo=img_path, caption=f"""Student Name:-  {data['Names'][i]}\nStudent Roll:-  {data['Roll Numbers'][i]}\nStudent Room No.:-  {sliced[0]}\nStudent Hall :- {sliced[1]}""")
                            os.remove(img_path)
                    
                        await context.bot.send_message(chat_id=chat_id, text="Hello :) ")
                        return
                    else:
                        continue
        

                        await update.message.reply_text("No such student found")
def main():
    if not TOKEN:
        print("Error: BOT_TOKEN is not set.")
        return
#E-1 E1-1 C-1
    # Create the Application
    application = Application.builder().token(TOKEN).build()

    # Add command handlers
    application.add_handler(CommandHandler("Start", start))
    application.add_handler(CommandHandler("Help", Help))
    application.add_handler(CommandHandler("website", website))
    application.add_handler(MessageHandler(filters.Regex(r'^\d{6}$'), handle_roll_no))
    application.add_handler(MessageHandler(filters.Regex(r'([A-Za-z][1-6])'), handle_wing))
    application.add_handler(MessageHandler(filters.Regex(r'([A-Za-z][1-6][1-6])'), handle_wing))
    application.add_handler(MessageHandler(filters.PHOTO, face_detection))


    # Start the Bot
    application.run_polling()


if __name__ == '__main__':
    main()
