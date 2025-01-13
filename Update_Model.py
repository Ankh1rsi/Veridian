import os
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import tensorflow as tf


def load_dataset(dataset_path='faces'):
    images = []
    labels = []

    print("Загрузка изображений...")
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if os.path.isdir(person_dir):
            for image_name in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_name)
                # Загружаем изображение размером 224x224
                image = load_img(image_path, target_size=(224, 224))
                image = img_to_array(image)
                images.append(image)
                labels.append(person_name)

    if not images:
        raise ValueError("Не найдено изображений в указанной папке")

    X = np.array(images)
    y = np.array(labels)

    # Предобработка для ResNet50
    X = X.astype('float32')
    X = preprocess_input(X)

    # Кодируем метки
    le = LabelEncoder()
    y = le.fit_transform(y)
    y = to_categorical(y)

    return X, y, le.classes_


def create_model(num_classes):
    # Загружаем предобученную ResNet50 без верхних слоев
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=(224, 224, 3)
    )

    # Замораживаем веса базовой модели
    for layer in base_model.layers:
        layer.trainable = False

    # Добавляем новые слои для классификации с регуляризацией
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.7)(x)
    x = Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.7)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    return model, base_model


def train_model():
    try:
        print("Загрузка датасета...")
        X, y, class_names = load_dataset()

        # Сохраняем имена классов
        np.save('class_names.npy', class_names)
        print(f"Найдено классов: {len(class_names)}")
        print("Имена классов:", class_names)

        # Разделяем данные
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("\nСоздание модели...")
        model, base_model = create_model(len(class_names))

        # Компилируем модель
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Добавляем аугментацию данных
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomBrightness(0.2),
            tf.keras.layers.RandomContrast(0.2),
        ])

        # Используем аугментацию при обучении
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.map(
            lambda x, y: (data_augmentation(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE
        )

        print("\nНачальное обучение...")
        # Обучаем модель
        history = model.fit(
            train_dataset.batch(32),
            validation_data=(X_test, y_test),
            epochs=20,
            callbacks=[
                ModelCheckpoint(
                    'face_recognition_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max',
                    verbose=1
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=0.00001
                )
            ]
        )

        print("\nТонкая настройка...")
        # Размораживаем последние слои ResNet
        for layer in base_model.layers[-30:]:
            layer.trainable = True

        # Перекомпилируем модель с меньшей скоростью обучения
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.00001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Продолжаем обучение с размороженными слоями
        history = model.fit(
            train_dataset.batch(16),
            validation_data=(X_test, y_test),
            epochs=20,
            callbacks=[
                ModelCheckpoint(
                    'face_recognition_model.h5',
                    monitor='val_accuracy',
                    save_best_only=True,
                    mode='max',
                    verbose=1
                ),
                EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=0.000001
                )
            ]
        )

        # Оцениваем модель
        test_loss, test_acc = model.evaluate(X_test, y_test)
        print(f"\nТочность на тестовых данных: {test_acc * 100:.2f}%")

        print("\nМодель сохранена как 'face_recognition_model.h5'")
        print("Имена классов сохранены в 'class_names.npy'")

    except Exception as e:
        print(f"\nОшибка при обучении модели: {e}")
        raise


if __name__ == '__main__':
    print("Обучение модели распознавания лиц")
    print("================================")
    train_model()