# Standard library imports
import os
import time
import sys
import json
import glob
import copy
import shutil

# Third-party imports
import numpy as np
import cv2
import matplotlib.pyplot as plt
import face_recognition
from PIL import Image as pImage

# Django imports
from django.shortcuts import render, redirect
from django.conf import settings

# Torch imports
import torch
import torchvision
from torchvision import transforms, models
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
from torch import nn

# Local imports
from .forms import VideoUploadForm

# Constants
INDEX_TEMPLATE = 'index.html'
PREDICT_TEMPLATE = 'predict.html'
ABOUT_TEMPLATE = "about.html"
IM_SIZE = 112
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'gif', 'webm', 'avi', '3gp', 'wmv', 'flv', 'mkv'}

# Transforms
sm = nn.Softmax()
inv_normalize = transforms.Normalize(
    mean=-1 * np.divide(MEAN, STD),
    std=np.divide([1, 1, 1], STD)
)
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# Model definition
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

# Dataset class
class ValidationDataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        for frame in self.frame_extract(video_path):
            faces = face_recognition.face_locations(frame)
            try:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            except IndexError:
                pass
            frames.append(self.transform(frame))
            if len(frames) == self.count:
                break
        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)
    
    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path) 
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image

# Utility functions
def im_convert(tensor, video_file_name):
    image = tensor.to("cpu").clone().detach()
    image = image.squeeze()
    image = inv_normalize(image)
    image = image.numpy()
    image = image.transpose(1, 2, 0)
    image = image.clip(0, 1)
    return image

def predict(model, img, path='./', video_file_name=""):
    fmap, logits = model(img.to(DEVICE))
    img = im_convert(img[:, -1, :, :, :], video_file_name)
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    return [int(prediction.item()), confidence]

def plot_heat_map(i, model, img, path='./', video_file_name=''):
    fmap, logits = model(img.to(DEVICE))
    weight_softmax = model.linear1.weight.detach().cpu().numpy()
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    idx = np.argmax(logits.detach().cpu().numpy())
    bz, nc, h, w = fmap.shape
    out = np.dot(fmap[i].detach().cpu().numpy().reshape((nc, h*w)).T, weight_softmax[idx, :].T)
    predict = out.reshape(h, w)
    predict = predict - np.min(predict)
    predict_img = predict / np.max(predict)
    predict_img = np.uint8(255 * predict_img)
    out = cv2.resize(predict_img, (IM_SIZE, IM_SIZE))
    heatmap = cv2.applyColorMap(out, cv2.COLORMAP_JET)
    img = im_convert(img[:, -1, :, :, :], video_file_name)
    result = heatmap * 0.5 + img * 0.8 * 255
    heatmap_name = f"{video_file_name}_heatmap_{i}.png"
    image_name = os.path.join(settings.PROJECT_DIR, 'uploaded_images', heatmap_name)
    cv2.imwrite(image_name, result)
    return image_name

def get_accurate_model(sequence_length):
    model_name = []
    sequence_model = []
    list_models = glob.glob(os.path.join(settings.PROJECT_DIR, "models", "*.pt"))

    for model_path in list_models:
        model_name.append(os.path.basename(model_path))

    for model_filename in model_name:
        try:
            seq = model_filename.split("_")[3]
            if int(seq) == sequence_length:
                sequence_model.append(model_filename)
        except IndexError:
            pass

    if len(sequence_model) > 1:
        accuracy = [float(filename.split("_")[1]) for filename in sequence_model]
        max_index = accuracy.index(max(accuracy))
        final_model = os.path.join(settings.PROJECT_DIR, "models", sequence_model[max_index])
    elif len(sequence_model) == 1:
        final_model = os.path.join(settings.PROJECT_DIR, "models", sequence_model[0])
    else:
        print("No model found for the specified sequence length.")
        final_model = None

    return final_model

def allowed_video_file(filename):
    return filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS

# View functions
def index(request):
    if request.method == 'GET':
        video_upload_form = VideoUploadForm()
        for key in ['file_name', 'preprocessed_images', 'faces_cropped_images']:
            if key in request.session:
                del request.session[key]
        return render(request, INDEX_TEMPLATE, {"form": video_upload_form})
    else:
        video_upload_form = VideoUploadForm(request.POST, request.FILES)
        if video_upload_form.is_valid():
            video_file = video_upload_form.cleaned_data['upload_video_file']
            video_file_ext = video_file.name.split('.')[-1]
            sequence_length = video_upload_form.cleaned_data['sequence_length']
            video_content_type = video_file.content_type.split('/')[0]
            
            if video_content_type in settings.CONTENT_TYPES and video_file.size > int(settings.MAX_UPLOAD_SIZE):
                video_upload_form.add_error("upload_video_file", "Maximum file size 100 MB")
                return render(request, INDEX_TEMPLATE, {"form": video_upload_form})

            if sequence_length <= 0:
                video_upload_form.add_error("sequence_length", "Sequence Length must be greater than 0")
                return render(request, INDEX_TEMPLATE, {"form": video_upload_form})
            
            if not allowed_video_file(video_file.name):
                video_upload_form.add_error("upload_video_file", "Only video files are allowed")
                return render(request, INDEX_TEMPLATE, {"form": video_upload_form})
            
            saved_video_file = f'uploaded_file_{int(time.time())}.{video_file_ext}'
            upload_path = os.path.join(settings.PROJECT_DIR, 'uploaded_videos', 'app', 'uploaded_videos') if not settings.DEBUG else os.path.join(settings.PROJECT_DIR, 'uploaded_videos')
            
            with open(os.path.join(upload_path, saved_video_file), 'wb') as vFile:
                shutil.copyfileobj(video_file, vFile)
            
            request.session['file_name'] = os.path.join(upload_path, saved_video_file)
            request.session['sequence_length'] = sequence_length
            return redirect('ml_app:predict')
        else:
            return render(request, INDEX_TEMPLATE, {"form": video_upload_form})

def predict_page(request):
    if request.method == "GET":
        if 'file_name' not in request.session:
            return redirect("ml_app:home")
        
        video_file = request.session['file_name']
        sequence_length = request.session['sequence_length']
        path_to_videos = [video_file]
        video_file_name = os.path.basename(video_file)
        video_file_name_only = os.path.splitext(video_file_name)[0]
        
        production_video_name = os.path.join('/home/app/staticfiles/', video_file_name.split('/')[3]) if not settings.DEBUG else video_file_name

        video_dataset = ValidationDataset(path_to_videos, sequence_length=sequence_length, transform=train_transforms)

        model = Model(2).to(DEVICE)
        model_name = get_accurate_model(sequence_length)
        if not model_name:
            return render(request, 'predict_template_name.html', {"no_model": True})
        
        path_to_model = os.path.join(settings.PROJECT_DIR, model_name)
        model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
        model.eval()

        start_time = time.time()
        print("<=== | Started Videos Splitting | ===>")
        preprocessed_images = []
        faces_cropped_images = []
        
        cap = cv2.VideoCapture(video_file)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
            else:
                break
        cap.release()

        print(f"Number of frames: {len(frames)}")
        padding = 40
        faces_found = 0
        for i in range(sequence_length):
            if i >= len(frames):
                break
            frame = frames[i]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            image_name = f"{video_file_name_only}_preprocessed_{i+1}.png"
            image_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name)
            img_rgb = pImage.fromarray(rgb_frame, 'RGB')
            img_rgb.save(image_path)
            preprocessed_images.append(image_name)

            face_locations = face_recognition.face_locations(rgb_frame)
            if len(face_locations) == 0:
                continue

            top, right, bottom, left = face_locations[0]
            frame_face = frame[top - padding:bottom + padding, left - padding:right + padding]
            rgb_face = cv2.cvtColor(frame_face, cv2.COLOR_BGR2RGB)
            img_face_rgb = pImage.fromarray(rgb_face, 'RGB')
            image_name = f"{video_file_name_only}_cropped_faces_{i+1}.png"
            image_path = os.path.join(settings.PROJECT_DIR, 'uploaded_images', image_name)
            img_face_rgb.save(image_path)
            faces_found += 1
            faces_cropped_images.append(image_name)

        print("<=== | Videos Splitting and Face Cropping Done | ===>")
        print(f"--- {time.time() - start_time} seconds ---")

        if faces_found == 0:
            return render(request, 'predict_template_name.html', {"no_faces": True})

        try:
            heatmap_images = []
            output = ""
            confidence = 0.0

            for i in range(len(path_to_videos)):
                print("<=== | Started Prediction | ===>")
                prediction = predict(model, video_dataset[i], './', video_file_name_only)
                confidence = round(prediction[1], 1)
                output = "REAL" if prediction[0] == 1 else "FAKE"
                print(f"Prediction: {prediction[0]} == {output} Confidence: {confidence}")
                print("<=== | Prediction Done | ===>")
                print(f"--- {time.time() - start_time} seconds ---")

            context = {
                'preprocessed_images': preprocessed_images,
                'faces_cropped_images': faces_cropped_images,
                'heatmap_images': heatmap_images,
                'original_video': production_video_name,
                'models_location': os.path.join(settings.PROJECT_DIR, 'models'),
                'output': output,
                'confidence': confidence
            }

            return render(request, PREDICT_TEMPLATE, context)

        except Exception as e:
            print(f"Exception occurred during prediction: {e}")
            return render(request, 'cuda_full.html')

def about(request):
    return render(request, ABOUT_TEMPLATE)

def handler404(request, exception):
    return render(request, '404.html', status=404)

def cuda_full(request):
    return render(request, 'cuda_full.html')