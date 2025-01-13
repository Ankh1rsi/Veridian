import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import dlib

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array


def recognize_live():
    # Загружаем обученную модель и имена классов
    try:
        model = load_model('1.h5')
        class_names = np.load('class_names.npy')
        print("Модель успешно загружена")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return

    # Загружаем каскад Хаара для детекции лиц
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Включаем камеру
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть камеру")
        return

    print("Нажмите 'q' для выхода")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка при получении кадра")
            break

        # Конвертируем кадр в градации серого для детекции лиц
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Обнаруживаем лица
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Обрабатываем каждое найденное лицо
        for (x, y, w, h) in faces:
            # Вырезаем область с лицом
            face_roi = frame[y:y + h, x:x + w]

            try:
                # Подготавливаем изображение для модели
                face_roi = cv2.resize(face_roi, (64, 64))
                face_roi = img_to_array(face_roi)
                face_roi = face_roi.astype('float32') / 255.0
                face_roi = np.expand_dims(face_roi, axis=0)

                # Получаем предсказание от модели
                predictions = model.predict(face_roi, verbose=0)
                person_id = np.argmax(predictions)
                confidence = predictions[0][person_id] * 100

                # Получаем имя распознанного человека
                name = class_names[person_id]

                # Отображаем рамку и имя
                color = (0, 255, 0) if confidence > 70 else (
                0, 255, 255)  # Зеленый если уверенность >70%, желтый если меньше
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                # Добавляем текст с именем и уверенностью
                label = f"{name} ({confidence:.1f}%)"
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            except Exception as e:
                print(f"Ошибка при обработке лица: {e}")
                continue

        # Показываем результат
        cv2.imshow('Face Recognition', frame)

        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Освобождаем ресурсы
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    recognize_live()