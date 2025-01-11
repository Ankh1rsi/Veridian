import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import dlib

class RealtimeRecognizer:
    def __init__(self, model_path, label_map_path=None, confidence_threshold=0.7):
        # Загружаем обученную модель
        print("Загрузка модели...")
        self.model = load_model(model_path)
        self.confidence_threshold = confidence_threshold

        # Загружаем маппинг классов если есть
        if label_map_path and os.path.exists(label_map_path):
            self.label_map = np.load(label_map_path, allow_pickle=True).item()
            print(f"Загружено {len(self.label_map)} классов")
        else:
            print("Маппинг классов не найден, будут использоваться номера классов")
            self.label_map = None

        # Инициализируем детектор лиц dlib
        print("Инициализация детектора лиц...")
        self.face_detector = dlib.get_frontal_face_detector()

    def preprocess_face(self, face_img):
        """Предобработка изображения лица"""
        # Изменяем размер до 224x224 (как при обучении)
        face_img = cv2.resize(face_img, (224, 224))
        # Нормализуем пиксели
        face_img = face_img.astype('float32') / 255.0
        # Добавляем размерность батча
        face_img = np.expand_dims(face_img, axis=0)
        return face_img

    def predict_face(self, face_img):
        """Распознавание лица"""
        predictions = self.model.predict(face_img, verbose=0)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]

        # Если уверенность ниже порога, считаем неизвестным
        if confidence < self.confidence_threshold:
            return "Unknown", confidence

        # Получаем метку класса
        if self.label_map:
            label = self.label_map.get(class_idx, f"Unknown_{class_idx}")
        else:
            label = f"Person_{class_idx}"

        return label, confidence

    def run_recognition(self):
        """Запуск распознавания в реальном времени"""
        # Инициализируем камеру
        print("Запуск камеры...")
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise ValueError("Не удалось открыть камеру!")

        print("Нажмите 'q' для выхода")

        while True:
            # Читаем кадр
            ret, frame = cap.read()
            if not ret:
                print("Ошибка при получении кадра")
                break

            # Конвертируем в RGB для dlib
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Детектируем лица
            faces = self.face_detector(rgb_frame)

            # Обрабатываем каждое найденное лицо
            for face in faces:
                # Получаем координаты лица
                x = face.left()
                y = face.top()
                w = face.right() - x
                h = face.bottom() - y

                # Проверяем границы
                if x >= 0 and y >= 0 and w > 0 and h > 0:
                    # Вырезаем лицо
                    face_img = frame[y:y + h, x:x + w]

                    # Предобрабатываем лицо
                    processed_face = self.preprocess_face(face_img)

                    # Получаем предсказание
                    label, confidence = self.predict_face(processed_face)

                    # Выбираем цвет рамки в зависимости от уверенности
                    if confidence > self.confidence_threshold:
                        color = (0, 255, 0)  # Зеленый для известных лиц
                    else:
                        color = (0, 0, 255)  # Красный для неизвестных

                    # Рисуем рамку
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                    # Добавляем текст с именем и уверенностью
                    text = f"{label}: {confidence:.2f}"
                    cv2.putText(frame, text, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Показываем результат
            cv2.imshow('Face Recognition', frame)

            # Выход по нажатию 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Освобождаем ресурсы
        cap.release()
        cv2.destroyAllWindows()


def main():
    # Пути к файлам
    MODEL_PATH = 'Veridian.h5'
    LABEL_MAP_PATH = 'models/class_mapping.npy'  # Путь к маппингу классов, если есть

    try:
        # Создаем и запускаем распознаватель
        recognizer = RealtimeRecognizer(
            model_path=MODEL_PATH,
            label_map_path=LABEL_MAP_PATH,
            confidence_threshold=0.7  # Порог уверенности
        )

        # Запускаем распознавание
        recognizer.run_recognition()

    except Exception as e:
        print(f"Ошибка: {str(e)}")


if __name__ == '__main__':
    main()