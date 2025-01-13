import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def load_dataset(dataset_path='faces'):
    images = []
    labels = []

    # Загружаем изображения и метки
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                # Загружаем и изменяем размер изображения
                image = load_img(image_path, target_size=(64, 64))
                image = img_to_array(image)
                images.append(image)
                labels.append(person_name)

    # Преобразуем списки в массивы numpy
    X = np.array(images)
    y = np.array(labels)

    # Нормализуем значения пикселей
    X = X.astype('float32') / 255.0

    # Кодируем метки
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = to_categorical(y)

    return X, y, le.classes_


def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model


def train_model():
    print("Загрузка датасета...")
    X, y, class_names = load_dataset()

    # Сохраняем имена классов
    np.save('class_names.npy', class_names)

    # Разделяем данные на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Создание модели...")
    model = create_model((64, 64, 3), len(class_names))

    # Компилируем модель
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("Начало обучения...")
    # Обучаем модель
    history = model.fit(X_train, y_train,
                        epochs=50,
                        batch_size=32,
                        validation_data=(X_test, y_test))

    # Оцениваем модель
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"\nТочность на тестовых данных: {test_acc * 100:.2f}%")

    # Сохраняем модель
    model.save('1.h5')
    print("Модель сохранена как '1.h5'")


if __name__ == '__main__':
    train_model()