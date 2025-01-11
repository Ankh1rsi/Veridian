import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import numpy as np
from keras.models import load_model, Model
from keras.layers import Dense, Dropout, BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt


def update_model_for_new_classes(base_model_path, data_dir, save_dir='models'):
    """
    Обновляет модель для работы с новым количеством классов
    """
    # Создаем директорию для сохранения если её нет
    os.makedirs(save_dir, exist_ok=True)

    # Загружаем базовую модель
    old_model = load_model(base_model_path)

    # Подсчитываем новое количество классов
    num_classes = len([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    print(f"Найдено классов в новом датасете: {num_classes}")

    # Создаем новую модель с тем же основанием, но новым выходным слоем
    x = old_model.layers[-4].output  # Берем выход слоя перед классификатором

    # Добавляем уникальные имена для новых слоев
    x = Dense(512, activation='relu', name='new_dense_1')(x)
    x = BatchNormalization(name='new_batch_norm_1')(x)
    x = Dropout(0.3, name='new_dropout_1')(x)
    predictions = Dense(num_classes, activation='softmax', name='new_predictions')(x)

    # Создаем новую модель
    model = Model(inputs=old_model.input, outputs=predictions)

    # Замораживаем все слои кроме новых
    for layer in model.layers[:-4]:
        layer.trainable = False

    # Выводим структуру модели для проверки
    print("\nСтруктура обновленной модели:")
    model.summary()

    # Компилируем модель
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    # Генератор данных с аугментацией
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        brightness_range=[0.8, 1.2],
        horizontal_flip=True,
        validation_split=0.2
    )

    # Определяем размер батча
    batch_size = 32

    # Создаем генераторы с shuffle=True
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    validation_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=True
    )

    # Вычисляем правильное количество шагов
    total_train = train_generator.samples
    total_val = validation_generator.samples

    steps_per_epoch = total_train // batch_size
    if total_train % batch_size != 0:
        steps_per_epoch += 1

    validation_steps = total_val // batch_size
    if total_val % batch_size != 0:
        validation_steps += 1

    print(f"Всего обучающих изображений: {total_train}")
    print(f"Всего валидационных изображений: {total_val}")
    print(f"Шагов за эпоху: {steps_per_epoch}")
    print(f"Шагов валидации: {validation_steps}")

    # Сохраняем маппинг классов
    class_mapping = {v: k for k, v in train_generator.class_indices.items()}
    np.save(os.path.join(save_dir, 'class_mapping.npy'), class_mapping)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            os.path.join(save_dir, 'updated_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Обучаем модель
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        epochs=50,
        callbacks=callbacks,
        verbose=1
    )

    # Сохраняем финальную модель
    model.save(os.path.join(save_dir, 'final_updated_model.h5'))

    # Строим и сохраняем графики обучения
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

    return model, history, class_mapping


if __name__ == '__main__':
    BASE_MODEL_PATH = 'Veridian.h5'
    NEW_DATA_DIR = 'faces'  # Путь к новому датасету

    try:
        # Обновляем модель
        model, history, class_mapping = update_model_for_new_classes(
            BASE_MODEL_PATH,
            NEW_DATA_DIR
        )

        print("\nОбновление модели завершено!")
        print(f"Количество классов: {len(class_mapping)}")
        print("\nПримеры классов:")
        for i, (class_id, class_name) in enumerate(class_mapping.items()):
            if i < 10:  # Показываем первые 10 классов
                print(f"{class_id}: {class_name}")
            else:
                print("...")
                break

        # Выводим финальные метрики
        final_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        print(f"\nФинальная точность на обучающей выборке: {final_acc:.4f}")
        print(f"Финальная точность на валидационной выборке: {final_val_acc:.4f}")

    except Exception as e:
        print(f"Ошибка при обновлении модели: {str(e)}")