import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import urllib.request
import tarfile
from tqdm import tqdm

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_lfw_dataset():
    # URL датасета LFW
    url = "http://vis-www.cs.umass.edu/lfw/lfw.tgz"
    output_dir = "lfw-dataset"
    archive_path = "lfw.tgz"

    # Создаем директорию если её нет
    os.makedirs(output_dir, exist_ok=True)

    # Скачиваем датасет
    print("Скачивание датасета LFW...")
    try:
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc="Downloading") as t:
            urllib.request.urlretrieve(url, archive_path,
                                       reporthook=t.update_to)
    except Exception as e:
        print(f"Ошибка при скачивании: {e}")
        return False

    # Распаковываем архив
    print("\nРаспаковка архива...")
    try:
        with tarfile.open(archive_path, "r:gz") as tar:
            total = len(tar.getmembers())
            for member in tqdm(tar.getmembers(), total=total, desc="Extracting"):
                tar.extract(member, output_dir)
    except Exception as e:
        print(f"Ошибка при распаковке: {e}")
        return False

    # Удаляем архив
    try:
        os.remove(archive_path)
    except Exception as e:
        print(f"Предупреждение: не удалось удалить архив: {e}")

    # Выводим статистику
    try:
        num_persons = len([d for d in os.listdir(f"{output_dir}/lfw")
                           if os.path.isdir(os.path.join(f"{output_dir}/lfw", d))])
        total_images = sum(len(os.listdir(os.path.join(f"{output_dir}/lfw", d)))
                           for d in os.listdir(f"{output_dir}/lfw")
                           if os.path.isdir(os.path.join(f"{output_dir}/lfw", d)))

        print("\nДатасет успешно загружен!")
        print(f"Количество персон: {num_persons}")
        print(f"Всего изображений: {total_images}")
        print(f"Датасет находится в директории: {output_dir}/lfw")

    except Exception as e:
        print(f"Предупреждение: не удалось подсчитать статистику: {e}")

    return True


if __name__ == "__main__":
    print("Начало загрузки датасета LFW...")
    if download_lfw_dataset():
        print("Загрузка завершена успешно!")
    else:
        print("Произошла ошибка при загрузке датасета")