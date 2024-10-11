import streamlit as st
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import efficientnet_b0
import time
import matplotlib.pyplot as plt
import numpy as np
import requests
from io import BytesIO

# Параметры модели
num_classes = 10
class_names = {
    0: "AnnualCrop", 1: "Forest", 2: "HerbaceousVegetation", 3: "Highway",
    4: "Industrial", 5: "Pasture", 6: "PermanentCrop", 7: "Residential", 
    8: "River", 9: "SeaLake"
}

# Описание проекта с новым оформлением
st.markdown("""
    <div style='background-color: #1a1a1a; padding: 15px; border-radius: 10px;'>
        <h1 style='color: #00cc66; text-align: center;'>🌍 Классификация спутниковых изображений</h1>
        <h3 style='color: #e6f7ff; text-align: center;'>Приложение для классификации спутниковых изображений</h3>
        <p style='text-align: center; color: #e6f7ff; font-size: 16px;'>
            Данное приложение использует две модели: <strong>ResNet-50</strong> и <strong>EfficientNet-B0</strong> 
            для классификации спутниковых изображений по 10 различным категориям, таким как <em>Forest</em>, <em>River</em> и т.д.
            Загружайте одно или несколько изображений, и вы получите предсказание с вероятностями для каждой модели, а также время выполнения каждой модели. 
            Вы также можете ввести ссылку на изображение и получить предсказание. Результаты представлены в виде графиков.
        </p>
    </div>
""", unsafe_allow_html=True)

# Загрузка моделей
def load_model(model_name):
    if model_name == "ResNet-50":
        model = models.resnet50()
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        # Явно указываем устройство 'map_location'
        model.load_state_dict(torch.load('models/resnet50_eurosat.pth', map_location=torch.device('cpu')))
    elif model_name == "EfficientNet-B0":
        model = efficientnet_b0(weights=None)
        model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
        # Явно указываем устройство 'map_location'
        model.load_state_dict(torch.load('models/efficientnet_b0_eurosat.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

# Загрузка моделей
resnet_model = load_model("ResNet-50")
efficientnet_model = load_model("EfficientNet-B0")

# Трансформация изображений
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Возможность загрузки изображения по ссылке
st.markdown("## Загрузка изображения по ссылке")
image_url = st.text_input("Введите URL изображения:")

if image_url:
    try:
        response = requests.get(image_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
        st.image(image, caption=f"Загруженное изображение по ссылке", use_column_width=True)
    except Exception as e:
        st.error(f"Ошибка загрузки изображения по ссылке: {e}")
        image = None

# Возможность загрузки нескольких изображений
uploaded_files = st.file_uploader("Загрузите одно или несколько изображений (форматы: jpg, jpeg, png):", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Список для хранения изображений
images = []

if uploaded_files:
    for uploaded_file in uploaded_files:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption=f"Загруженное изображение: {uploaded_file.name}", use_column_width=True)
        images.append((image, uploaded_file.name))

# Добавляем изображение, загруженное по ссылке, если есть
if image_url and image is not None:
    images.append((image, "Image from URL"))

# Если изображения загружены, запускаем предсказание
if images:
    st.markdown("---")  # Разделитель

    # Проходим по всем изображениям
    for img, img_name in images:
        st.subheader(f"📷 Изображение: {img_name}")
        image_tensor = transform(img).unsqueeze(0)

        # Разделение на две колонки для моделей
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔍 Предсказания модели ResNet-50")
            start_time = time.time()
            with torch.no_grad():
                output_resnet = resnet_model(image_tensor)
            end_time = time.time()
            resnet_time = end_time - start_time
            top5_resnet_prob, top5_resnet_classes = torch.topk(torch.softmax(output_resnet, 1), 5)
            st.info(f"⏱ Время выполнения: {resnet_time:.4f} секунд")

            for i in range(5):
                class_idx = top5_resnet_classes[0][i].item()
                class_name = class_names[class_idx]
                probability = top5_resnet_prob[0][i].item() * 100
                st.write(f"**Класс {class_idx}: {class_name}** - Вероятность: {probability:.2f}%")

        with col2:
            st.subheader("🔍 Предсказания модели EfficientNet-B0")
            start_time = time.time()
            with torch.no_grad():
                output_efficientnet = efficientnet_model(image_tensor)
            end_time = time.time()
            efficientnet_time = end_time - start_time
            top5_efficientnet_prob, top5_efficientnet_classes = torch.topk(torch.softmax(output_efficientnet, 1), 5)
            st.info(f"⏱ Время выполнения: {efficientnet_time:.4f} секунд")

            for i in range(5):
                class_idx = top5_efficientnet_classes[0][i].item()
                class_name = class_names[class_idx]
                probability = top5_efficientnet_prob[0][i].item() * 100
                st.write(f"**Класс {class_idx}: {class_name}** - Вероятность: {probability:.2f}%")

    # Визуализация различий между моделями с помощью matplotlib
    st.markdown("---")
    st.subheader("📊 Сравнение моделей")

    # Данные для графиков
    resnet_values = [top5_resnet_prob[0][i].item() * 100 for i in range(5)]
    efficientnet_values = [top5_efficientnet_prob[0][i].item() * 100 for i in range(5)]
    labels = [f"Класс {top5_resnet_classes[0][i].item()}:\n{class_names[top5_resnet_classes[0][i].item()]}" for i in range(5)]

    # Построение двух столбчатых диаграмм рядом
    x = np.arange(len(labels))  # Метки классов

    fig, ax = plt.subplots(figsize=(10, 5))
    width = 0.35  # Ширина столбцов

    # Построение столбцов для каждой модели
    bars1 = ax.bar(x - width/2, resnet_values, width, label='ResNet-50', color='#1f77b4')
    bars2 = ax.bar(x + width/2, efficientnet_values, width, label='EfficientNet-B0', color='#ff7f0e')

    # Подписи и параметры осей
    ax.set_xlabel('Классы')
    ax.set_ylabel('Вероятность (%)')
    ax.set_title('Сравнение предсказаний моделей ResNet-50 и EfficientNet-B0')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylim(0, 100)  # Ограничение по оси Y от 0 до 100%

    # Добавление легенды
    ax.legend()

    # Показ графика в Streamlit
    st.pyplot(fig)