import cv2
import os
import time


def capture_faces(name, duration=30, fps=30):
    # Создаем директорию для датасета, если её нет
    dataset_dir = 'faces'
    person_dir = os.path.join(dataset_dir, name)

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
    if not os.path.exists(person_dir):
        os.makedirs(person_dir)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть камеру")
        return

    print(f"Начинаем запись видео для {name} на {duration} секунд...")
    start_time = time.time()
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Ошибка при получении кадра")
            break

        # Показываем видео в реальном времени
        cv2.imshow('Recording', frame)

        # Сохраняем кадры в подпапку
        if frame_count % int(cap.get(cv2.CAP_PROP_FPS) / fps) == 0:
            filename = os.path.join(person_dir, f"{name}_{frame_count}.jpg")
            cv2.imwrite(filename, frame)

        frame_count += 1

        # Останавливаем запись через заданное время
        if time.time() - start_time > duration:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Запись завершена. Сохранено {frame_count} кадров в {person_dir}")


if __name__ == '__main__':
    name = input("Введите имя человека для создания датасета: ")
    capture_faces(name)