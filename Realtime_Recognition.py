import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input


def recognize_live():
    try:
        model = load_model('face_recognition_model.h5')
        class_names = np.load('class_names.npy')
        print("Модель успешно загружена")
    except Exception as e:
        print(f"Ошибка при загрузке модели: {e}")
        return

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть камеру")
        return

    print("Нажмите 'q' для выхода")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Находим лица на кадре
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            try:
                # Вырезаем и обрабатываем лицо
                face_roi = frame[y:y + h, x:x + w]
                face_roi = cv2.resize(face_roi, (224, 224))  # ResNet требует 224x224
                face_roi = img_to_array(face_roi)
                face_roi = np.expand_dims(face_roi, axis=0)
                face_roi = preprocess_input(face_roi)

                # Получаем предсказание
                predictions = model.predict(face_roi, verbose=0)
                person_id = np.argmax(predictions)
                confidence = predictions[0][person_id] * 100

                # Получаем имя человека
                name = class_names[person_id]

                # Отображаем результат
                color = (0, 255, 0) if confidence > 90 else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                label = f"{name} ({confidence:.1f}%)"
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            except Exception as e:
                print(f"Ошибка при обработке лица: {e}")
                continue

        cv2.imshow('Face Recognition', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    recognize_live()