from flask import Flask, render_template, request, send_file
import os
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import imagehash
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics.pairwise import cosine_similarity
from werkzeug.utils import secure_filename
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load pretrained model (ResNet18)
# model = models.resnet18(pretrained=True)
model = models.resnet18(weights=None)

model.eval()
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def mse(img1, img2):
    return np.mean((img1.astype("float") - img2.astype("float")) ** 2)

def get_hashes(img):
    return {
        "aHash": str(imagehash.average_hash(img)),
        "dHash": str(imagehash.dhash(img)),
        "pHash": str(imagehash.phash(img)),
    }

def get_deep_features(img):
    img = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = feature_extractor(img)
    return features.view(-1).numpy()

def compare_images(file1, file2):
    result = {
        "Image 1": file1,
        "Image 2": file2
    }

    path1 = os.path.join(UPLOAD_FOLDER, file1)
    path2 = os.path.join(UPLOAD_FOLDER, file2)

    img1 = cv2.imread(path1)
    img2 = cv2.imread(path2)

    if img1 is None or img2 is None:
        result["Error"] = "One or both images are not readable"
        return result

    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    result["Resolution Match"] = (h1 == h2 and w1 == w2)
    result["Resolution 1"] = f"{w1}x{h1}"
    result["Resolution 2"] = f"{w2}x{h2}"

    img2_resized = cv2.resize(img2, (w1, h1))

    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2_resized, cv2.COLOR_BGR2GRAY)

    result["SSIM"] = round(ssim(gray1, gray2), 4)
    result["MSE"] = round(mse(gray1, gray2), 2)

    pil1 = Image.open(path1).convert('RGB')
    pil2 = Image.open(path2).convert('RGB')

    hashes1 = get_hashes(pil1)
    hashes2 = get_hashes(pil2)
    for key in hashes1:
        result[f"{key} Similar"] = hashes1[key] == hashes2[key]

    feat1 = get_deep_features(pil1)
    feat2 = get_deep_features(pil2)
    deep_sim = cosine_similarity([feat1], [feat2])[0][0]
    result["Deep Similarity"] = round(float(deep_sim), 4)

    zoom_ratio = round((w2 * h2) / (w1 * h1), 2)
    result["Zoom Ratio"] = zoom_ratio
    if zoom_ratio > 1.1:
        result["Zoom Status"] = "Zoomed In"
    elif zoom_ratio < 0.9:
        result["Zoom Status"] = "Zoomed Out"
    else:
        result["Zoom Status"] = "No Zoom"

    if result["SSIM"] > 0.90 and result["Deep Similarity"] > 0.90:
        result["Verdict"] = "Similar"
    elif result["Zoom Status"] != "No Zoom" and result["Deep Similarity"] > 0.85:
        result["Verdict"] = "Zoomed Variant"
    else:
        result["Verdict"] = "Different"

    return result

@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        files = request.files.getlist('files')
        filenames = []
        for file in files:
            if file:
                filename = secure_filename(file.filename)
                save_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(save_path)
                filenames.append(filename)

        comparisons = []
        for i in range(len(filenames)):
            for j in range(i+1, len(filenames)):
                result = compare_images(filenames[i], filenames[j])
                comparisons.append(result)

        df = pd.DataFrame(comparisons)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_path = os.path.join(RESULT_FOLDER, f'comparison_result_{timestamp}.xlsx')
        df.to_excel(result_path, index=False)
        return send_file(result_path, as_attachment=True)
    return render_template('index.html')

# 🔥 THIS SHOULD BE AT THE BOTTOM AND ONLY ONCE
if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
