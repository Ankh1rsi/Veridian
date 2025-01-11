import cv2
import dlib
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import time
from datetime import datetime


class FaceDatasetCapture:
    def __init__(self, output_dir='faces', capture_duration=10):
        self.output_dir = output_dir
        self.capture_duration = capture_duration
        self.face_detector = dlib.get_frontal_face_detector()

    def create_person_directory(self, name):
        """Создает директорию для нового человека"""
        person_dir = os.path.join(self.output_dir, name)
        os.makedirs(person_dir, exist_ok=True)
        return person_dir

    def capture_faces(self, name):
        """Захват изображений лица с камеры"""
        person_dir = self.create_person_directory(name)

        # Инициализируем камеру
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Не удалось открыть камеру!")

        print(f"\nНачинаем запись для {name}")
        print("Расположите лицо перед камерой")
        print("Медленно поворачивайте голову для лучшего результата")

        # Задержка перед началом записи
        for i in range(3, 0, -1):
            print(f"Начало через {i}...")
            time.sleep(1)

        start_time = time.time()
        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Проверяем время записи
            elapsed_time = time.time() - start_time
            if elapsed_time > self.capture_duration:
                break

            # Конвертируем в RGB для детектора лиц
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.face_detector(rgb_frame)

            frame_count += 1
            # Сохраняем каждый 3-й кадр для разнообразия
            if frame_count % 3 == 0:
                for face in faces:
                    x = face.left()
                    y = face.top()
                    w = face.right() - x
                    h = face.bottom() - y

                    if x >= 0 and y >= 0 and w > 0 and h > 0:
                        # Вырезаем и сохраняем лицо
                        face_img = frame[y:y + h, x:x + w]
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = os.path.join(person_dir, f"{name}_{timestamp}.jpg")
                        cv2.imwrite(filename, face_img)
                        saved_count += 1

            # Отображаем прогресс
            remaining_time = int(self.capture_duration - elapsed_time)
            for face in faces:
                x = face.left()
                y = face.top()
                w = face.right() - x
                h = face.bottom() - y
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Добавляем информацию на кадр
            cv2.putText(frame, f"Осталось: {remaining_time} сек", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Сохранено: {saved_count}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Capturing Face Dataset', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        print(f"\nЗапись завершена!")
        print(f"Сохранено {saved_count} изображений в {person_dir}")
        return saved_count


def main():
    # Создаем основную директорию если её нет
    if not os.path.exists('faces'):
        os.makedirs('faces')

    while True:
        # Запрашиваем имя
        name = input("\nВведите имя человека (или 'q' для выхода): ").strip()
        if name.lower() == 'q':
            break

        if not name:
            print("Имя не может быть пустым!")
            continue

        try:
            # Создаем экземпляр класса и начинаем запись
            capturer = FaceDatasetCapture(capture_duration=10)
            num_saved = capturer.capture_faces(name)

            if num_saved == 0:
                print("Предупреждение: не удалось сохранить ни одного изображения!")
            elif num_saved < 10:
                print("Предупреждение: сохранено мало изображений. Рекомендуется повторить запись.")

            # Спрашиваем, хочет ли пользователь продолжить
            choice = input("\nХотите записать еще одного человека? (y/n): ").strip().lower()
            if choice != 'y':
                break

        except Exception as e:
            print(f"Ошибка при записи: {str(e)}")
            choice = input("\nХотите попробовать снова? (y/n): ").strip().lower()
            if choice != 'y':
                break


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nЗапись прервана пользователем")
    except Exception as e:
        print(f"\nКритическая ошибка: {str(e)}")
    finally:
        cv2.destroyAllWindows()import cv2
import dlib
import os
import time
from datetime import datetime

class FaceDatasetCapture:
    def __init__(self, output_dir='faces', capture_duration=10):
        self.output_dir = output_dir
        self.capture_duration = capture_duration
        self.face_detector = dlib.get_frontal_face_detector()

    def create_person_directory(self, name):
        """Создает директорию для нового человека"""
        person_dir = os.path.join(self.output_dir, name)
        os.makedirs(person_dir, exist_ok=True)
        return person_dir

    def capture_faces(self, name):
        """Захват изображений лица с камеры"""
        person_dir = self.create_person_directory(name)

        # Инициализируем камеру
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("Не удалось открыть камеру!")

        print(f"\nНачинаем запись для {name}")
        print("Расположите лицо перед камерой")
        print("Медленно поворачивайте голову для лучшего результата")

        # Задержка перед началом записи
        for i in range(3, 0, -1):
            print(f"Начало через {i}...")
            time.sleep(1)

        start_time = time.time()
        frame_count = 0
        saved_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Проверяем время записи
            elapsed_time = time.time() - start_time
            if elapsed_time > self.capture_duration:
                break

            # Конвертируем в RGB для детектора лиц
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = self.face_detector(rgb_frame)

            frame_count += 1
            # Сохраняем каждый 3-й кадр для разнообразия
            if frame_count % 3 == 0:
                for face in faces:
                    x = face.left()
                    y = face.top()
                    w = face.right() - x
                    h = face.bottom() - y

                    if x >= 0 and y >= 0 and w > 0 and h > 0:
                        # Вырезаем и сохраняем лицо
                        face_img = frame[y:y+h, x:x+w]
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = os.path.join(person_dir, f"{name}_{timestamp}.jpg")
                        cv2.imwrite(filename, face_img)
                        saved_count += 1

            # Отображаем прогресс
            remaining_time = int(self.capture_duration - elapsed_time)
            for face in faces:
                x = face.left()
                y = face.top()
                w = face.right() - x
                h = face.bottom() - y
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Добавляем информацию на кадр
            cv2.putText(frame, f"Осталось: {remaining_time} сек", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Сохранено: {saved_count}", (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Capturing Face Dataset', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        print(f"\nЗапись завершена!")
        print(f"Сохранено {saved_count} изображений в {person_dir}")
        return saved_count

def main():
    # Создаем основную директорию если её нет
    if not os.path.exists('faces'):
        os.makedirs('faces')

    while True:
        # Запрашиваем имя
        name = input("\nВведите имя человека (или 'q' для выхода): ").strip()
        if name.lower() == 'q':
            break

        if not name:
            print("Имя не может быть пустым!")
            continue

        try:
            # Создаем экземпляр класса и начинаем запись
            capturer = FaceDatasetCapture(capture_duration=10)
            num_saved = capturer.capture_faces(name)

            if num_saved == 0:
                print("Предупреждение: не удалось сохранить ни одного изображения!")
            elif num_saved < 10:
                print("Предупреждение: сохранено мало изображений. Рекомендуется повторить запись.")

            # Спрашиваем, хочет ли пользователь продолжить
            choice = input("\nХотите записать еще одного человека? (y/n): ").strip().lower()
            if choice != 'y':
                break

        except Exception as e:
            print(f"Ошибка при записи: {str(e)}")
            choice = input("\nХотите попробовать снова? (y/n): ").strip().lower()
            if choice != 'y':
                break

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nЗапись прервана пользователем")
    except Exception as e:
        print(f"\nКритическая ошибка: {str(e)}")
    finally:
        cv2.destroyAllWindows()