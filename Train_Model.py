import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt


def build_model_from_scratch(input_shape=(224, 224, 3), num_classes=2):
    model = Sequential([
        # Первый сверточный блок
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Второй сверточный блок
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Третий сверточный блок
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Четвертый сверточный блок
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.25),

        # Полносвязные слои
        Flatten(),
        Dense(512, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(256, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    return model


def train_model(data_dir, model_save_dir='models', input_shape=(224, 224, 3), batch_size=32, epochs=1000):
    # Создаем директорию для сохранения моделей
    os.makedirs(model_save_dir, exist_ok=True)

    # Подсчитываем количество классов
    num_classes = len([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    print(f"Найдено классов: {num_classes}")

    # Создаем генераторы данных с аугментацией
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )

    # Создаем генераторы для обучения и валидации
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=input_shape[:2],
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # Создаем модель
    model = build_model_from_scratch(input_shape, num_classes)

    # Компилируем модель
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Выводим структуру модели
    model.summary()

    # Настраиваем callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(model_save_dir, 'Model_1.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Обучаем модель
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        epochs=epochs,
        callbacks=callbacks
    )

    # Сохраняем финальную модель
    model.save(os.path.join(model_save_dir, 'final_model0.h5'))

    # Сохраняем историю обучения
    np.save(os.path.join(model_save_dir, 'training_history.npy'), history.history)

    # Строим графики
    plot_training_history(history, model_save_dir)

    return model, history


def plot_training_history(history, save_dir):
    plt.figure(figsize=(12, 4))

    # График точности
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    # График функции потерь
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()
def clean_dataset(data_dir, min_images=10):
    """Очищает датасет от папок с малым количеством изображений"""
    removed = 0
    for person_folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, person_folder)
        if os.path.isdir(folder_path):
            images = [f for f in os.listdir(folder_path)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if len(images) < min_images:
                print(f"Удаляем {person_folder}: всего {len(images)} изображений")
                import shutil
                shutil.rmtree(folder_path)
                removed += 1
    print(f"Удалено {removed} папок с менее чем {min_images} изображениями")

def main():
    # Параметры обучения
    DATA_DIR = 'lfw'
    INPUT_SHAPE = (224, 224, 3)
    BATCH_SIZE = 32
    EPOCHS = 1000
    clean_dataset(DATA_DIR, min_images=10)
    # Проверяем наличие датасета
    if not os.path.exists(DATA_DIR):
        raise ValueError(f"Директория датасета не найдена: {DATA_DIR}")

    try:
        # Обучаем модель
        model, history = train_model(
            DATA_DIR,
            input_shape=INPUT_SHAPE,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS
        )

        # Выводим финальные метрики
        final_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        print(f"\nФинальная точность на обучающей выборке: {final_acc:.4f}")
        print(f"Финальная точность на валидационной выборке: {final_val_acc:.4f}")

    except Exception as e:
        print(f"Ошибка при обучении: {str(e)}")
        raise


if __name__ == '__main__':
    main()