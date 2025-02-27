# Тестовое задание для кандидата Devops Engineer

Для выполнения тестового задания рекомендуем пользоваться одним из дистрибутивов **Linux GNU**.
Также для выполнения задания понадобится компьютер с GPU компании Nvidia.

Дерево проекта:

## 1. Сконвертируйте модель

Наша команда разработки часто работает с tensorrt(TRT), как на cpp так и на python. В данном задание мы проверяем кандидатов на работу с настройкой окружения python3 и работой с командной строкой.

В данном шаге задания необходимо выполнить запустить инференс модели в среде, которую вы создадите!:D **скрипты уже подготовлены**

1. Перевести `model.onnx` в `model.engine` с использованием **уже созданного скрипта** - `onnx2trt.py`

Для этого рекомендуется создать среду и установить зависимости `requirements.txt`, это может быть как `conda env`, так и `python3 venv`, рекомендованная версия `Python: 3.12`.

* создайте среду установите зависимости
* для конвертации модели перейдите в папку `model` и запустите скрипт согласно файлу `model/readme.md`.  

В последних графах после конвертации должно появится, примерно такое сообщение:

```
Layer(Reformat): /model.22/Mul_2_output_0 copy, Tactic: 0x0000000000000000, /model.22/Mul_2_output_0 (Half[1,4,5040]) -> output0 (Float[1,4,5040])
[02/26/2025-13:41:18] [TRT] [I] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 2 MiB, GPU 157 MiB
[02/26/2025-13:41:18] [TRT] [V] Adding 1 engine(s) to plan file.
[02/26/2025-13:41:18] [TRT] [V] Adding 1 engine weights(s) to plan file.

```
После конвертации модели в папке `model` должен появиться файл `model.engine`. 


## 2. Написать скрипт на python3

В данном шаге необходимо **дописать/модифицировать скрипт `main.py`** на python3 для запуска inference модели и его отображение через окно Qt (метод `imshow`). В данном случае мы хотели бы, чтобы кандидаты продемострировали свое владение python3 с приминением **принципов ООП**. В качестве baseline приводим код ниже:

```
if __name__ == "__main__":
    PATH_MODEL = "./model/model.engine"
    IMG_PATH = "./test_img/1.png"
    model = Model(PATH_MODEL, (640, 384))
    img  = cv2.imread(IMG_PATH)
    img = cv2.resize(img, (640, 384))
    boxes = model(img)
    frame = draw_bboxes(img, boxes)
    cv2.imshow("yolo", img)
    cv2.waitKey(0)

```
Как можно заметить в коде присутствует класс `TrtYOLO`, который отвечает за получение и обработку боксов на основе входного изображения. Данный класс необходимо ипортировать в `main.py` из `modules/inference.py`, также необходимо импортировать библиотеку `opencv`.

## 3. Создание Dockerfile

Не обходимо написать `dockerfile` для запуска инференса модели (то что вы писали на прошлом шаге) внутри контейнера при **запуске**, должно появляться окно - метод `cv2.imshow`. Склонируйте файлы, установите зависимости, используйте образы `nvidia` с `dockerhub`. Предложите варианты защиты контейнера.
В данном случае мы проверяем кандидатов на владение `docker`.

## 4. Создание docker-compose

Необходимо написать `docker-compose` и `dockerfile` для клиент-серверного приложения приведенного в папке `server_client` и запустить его, соответственно два скрипта должны лежать в разных контейнерах. В данном случае мы проверяем кандидатов на владение `docker-compose`.
**Внимание** в приложение клиента после получения сообщения от сервера происходит визуализация изображения нужно сделать так, чтобы изображение появлялось на экране после запуска **из контейнера**.


## 5. Опубликуйте репозиторий и пришлите ссылку на него.














 
