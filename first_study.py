# Импорт необходимых библиотек ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import numpy as np
import scipy.special as sc
import math as m
import matplotlib as mpl
import matplotlib.pyplot as plt
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
mpl.use('GTK3Agg')      # Использование библиотеки GTK для вывода графиков
bias = 1.0              # Нейрон смещения, всегда равен 1
errors_array = []
# Определение класса нейронной сети ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class NN:
    # Функция инициализации нейронной сети ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):

        # Количество узлов на входном, скрытом и выходном слоях
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        # Коэффициент обучения
        self.lr = learningrate

        # Матрицы весовых коэффициентов связей
        # при инициализации заполняются случайными числами с нормальным распределением
        #   wih - между входным и скрытым слоями
        #   who - между скрытым и выходным слоями
        # numpy.random.normal(loc=0.0, scale=1.0, size=None) - метод, возвращающий случайные числа
        # по нормальному (Гаусову) распределению, параметры которого:
        #   loc - "центр" распределения, в нашем случае 0.0
        #   scale - стандартное отклонение (спред или "ширина"), в нашем случае величина
        # стандартного отклонения обратно пропорциональна квадратному корню из количества связей
        # на узел
        #   size - необязательный параметр, определяющий форму вывода функции, в нашем случае это
        # двумерная матрица с размерами (hnodes x inodes)
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), ((self.hnodes - 1), self.inodes))
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # Функции активации:
        self.hidden_act_func = lambda x: sc.expit(x)    # сигмоида для скрытого слоя
        self.output_act_func = lambda x: np.tanh(x)     # гиперболический тангенс для выходного слоя
        
        pass
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Функция тренировки нейронной сети ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def train(self, inputs_list, targets_list):
        # Добавление нейрона смещения к массиву входных данных
        inputs_list = np.append(inputs_list, bias)
        # Преобразование массива входных и целевых значений в двумерный массив
        inputs = np.array(inputs_list, ndmin = 2).T
        targets = np.array(targets_list, ndmin = 2).T

        # Расчет входящих сигналов для скрытого слоя
        hidden_inputs = np.dot(self.wih, inputs)
        # Добавление нейрона смещения к скрытому слою
        hidden_inputs = np.append(hidden_inputs, bias)
        # Преобразование массива входных данных скрытого слоя в двумерный массив
        hidden_inputs = np.array(hidden_inputs, ndmin = 2).T
        # Расчёт исходящих сигналов скрытого слоя
        hidden_outputs = self.hidden_act_func(hidden_inputs)

        # Расчет входящих сигналов для выходного слоя
        final_inputs = np.dot(self.who, hidden_outputs)
        # Расчёт исходящих сигналов выходного слоя
        final_outputs = self.output_act_func(final_inputs)

        # Ошибка - разница между целью и результатом сети
        output_errors = targets - final_outputs
        #print(final_outputs)
        # Формирование массива ошибок для графика
        errors_array.append(output_errors[0])

        # Вычисление ошибок скрытого слоя
        hidden_errors = np.dot(self.who.T, output_errors)

        # Обновление весов связей между выходным и скрытым слоями
        self.who += self.lr * np.dot((output_errors * (1 - pow(final_outputs, 2))), np.transpose(hidden_outputs))

        # Удаление последнего элемента массива (нейрона смещения)
        hidden_errors = np.delete(hidden_errors, (self.hnodes - 1), axis = 0)
        hidden_outputs = np.delete(hidden_outputs, (self.hnodes - 1), axis = 0)
        # Обновление весов связей между скрытым и входным слоями
        self.wih += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

        pass
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Функция опроса нейронной сети ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    def query(self, inputs_list):
        # Добавление нейрона смещения к массиву входных данных
        inputs_list = np.append(inputs_list, bias)
        # Преобразование массива входных значений в двумерный массив
        inputs = np.array(inputs_list, ndmin = 2).T

        # Расчет входящих сигналов для скрытого слоя
        hidden_inputs = np.dot(self.wih, inputs)
        # Добавление нейрона смещения к скрытому слою
        hidden_inputs = np.append(hidden_inputs, bias)
        # Расчёт исходящих сигналов скрытого слоя
        hidden_outputs = self.hidden_act_func(hidden_inputs)

        # Расчет входящих сигналов для выходного слоя
        final_inputs = np.dot(self.who, hidden_outputs)
        # Расчёт исходящих сигналов выходного слоя
        final_outputs = self.output_act_func(final_inputs)
        
        return final_outputs[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Характеристики экземпляра класса нейронной сети ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# количество свечей для одного набора
candles_quantity = 15

# количество входных узлов (параметров) - 6 параметров свечей * 5 свечей + 1 нейрон смещения
input_nodes = 6 * candles_quantity + 1

# количество скрытых узлов - равно количеству входнх узлов + нейрон смещения
hidden_nodes = input_nodes + 100

# количество выходных узлов
output_nodes = 1

# коэффициент обучения
learning_rate = 0.01

nn = NN(input_nodes, hidden_nodes, output_nodes, learning_rate)

# Подготовка входящих тренировочных и тестовых данных ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
dataset_file = "./dataset/RI_extended.csv"

# Открытие файла с данными для чтения
dataset_file = open(dataset_file, "r", encoding="utf-8")

# Чтение всех строк из файла и закрытие файла
all_list = dataset_file.readlines()
dataset_file.close()

# Общее количество строк (каждая строка соответствует одной дневной свече)
total_candles = len(all_list)
# Разделение данных на две группы - тренировочную (~80%) и проверочную (~20%)
total_trainset = int(total_candles * 0.8)
total_testset = total_candles - total_trainset

# Тренировка сети заданное количество раз (эпох) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
epochs = 500
for e in range(epochs):
    # Формирование массива тренировочных данных ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    i = 0
    while i <= total_trainset:          # Цикл перебора ~80% строк
        ohlc_dataset = []               # "Open-High-Low-Close" - массив для временного хранения 
        volume_dataset = []             # "Volume" - массив для временного хранения
        change_dataset = []             # "Change" - массив для временного хранения изменений цены
        inputs = []                     # Массив тренировочных данных
        targets = []                    # Массив "ответов" (целевых значений), здесь один элемент
        j = i
        while j <= i + (candles_quantity - 1):      # Цикл перебора n-свечек
            temp_row = all_list[j].split(";")
            ohlc_row = temp_row[2:6]
            k = 0
            while k <= 3:                           # Цикл перебора "Open-High-Low-Close"
                ohlc_dataset.append(float(ohlc_row[k]))
                k += 1
            volume_dataset.append(float(temp_row[6]))
            change_dataset.append(float(temp_row[7]))
            j += 1

        targets = int(all_list[j].split(";")[8])    # Целевое значение - "направление" следующей свечи
        targets = np.asfarray(targets)              # Преобразование списка в массив numpy

        # Итоговый список одного набора данных для тренировки
        inputs = ohlc_dataset + volume_dataset + change_dataset
        
        # Подготовка для нормальзиции входных данных: определение диаппазона между макс. и  мин.;
        # и приведение к диаппазону 0.01...1.0
        inputs_min = min(inputs)
        inputs_max = max(inputs)
        inputs_range = inputs_max + m.fabs(inputs_min)
        
        # Преобразование списка в массив numpy и нормализация
        inputs = np.asfarray(inputs)
        inputs = ((inputs + m.fabs(inputs_min)) / inputs_range) * 0.99 + 0.01

        # ВЫЗОВ ТРЕНИРОВОЧНОГО МЕТОДА
        nn.train(inputs, targets)

        i += 1
    pass

# Формирование массива тестовых данных ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
scorecard = []      # Массив оценок работы сети
i = total_trainset
while i < total_candles - candles_quantity: # Цикл перебора ~20% последних строк всего набора
    ohlc_dataset = []                       # "Open-High-Low-Close" - массив для временного хранения 
    volume_dataset = []                     # "Volume" - массив для временного хранения
    change_dataset = []                     # "Change" - массив для временного хранения изменений цены
    inputs = []                             # Массив тренировочных данных
    correct_direction = []                  # Массив "ответов" (целевых значений), здесь один элемент
    j = i
    while j <= i + (candles_quantity - 1):              # Цикл перебора n-свечек
        temp_row = all_list[j].split(";")
        ohlc_row = temp_row[2:6]
        k = 0
        while k <= 3:                                   # Цикл перебора "Open-High-Low-Close"
            ohlc_dataset.append(float(ohlc_row[k]))
            k += 1
        volume_dataset.append(float(temp_row[6]))
        change_dataset.append(float(temp_row[7]))
        j += 1

    correct_direction = int(all_list[j].split(";")[8])  # Целевое значение - "направление" 6-й свечи

    # Итоговый список одного набора данных для тренировки
    inputs = ohlc_dataset + volume_dataset + change_dataset

    # Подготовка для нормальзиции входных данных: определение диаппазона между макс. и  мин.;
    # и приведение к диаппазону 0.01...1.0
    inputs_min = min(inputs)
    inputs_max = max(inputs)
    inputs_range = inputs_max + m.fabs(inputs_min)
        
    # Преобразование списка в массив numpy и нормализация
    inputs = np.asfarray(inputs)
    inputs = ((inputs + m.fabs(inputs_min)) / inputs_range) * 0.99 + 0.01
    
    # Опрос сети на тестовых данных
    outputs = nn.query(inputs)

    # Подсчёт количества правильных ответов
    if correct_direction < 0 and outputs < 0.0:
        scorecard.append(1)
    elif correct_direction > 0 and outputs > 0.0:
        scorecard.append(1)
    else:
        scorecard.append(0)

    i += 1

# Вычисление точности сети
print("Точность сети:", 100 * (sum(scorecard) / len(scorecard)))

# Предсказание "на завтра" по последним n-свечкам ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
j = total_candles - candles_quantity
ohlc_dataset = []                       # "Open-High-Low-Close" - массив для временного хранения 
volume_dataset = []                     # "Volume" - массив для временного хранения
change_dataset = []                     # "Change" - массив для временного хранения изменений цены
inputs = []                             # Массив тренировочных данных
correct_direction = []                  # Массив "ответов" (целевых значений), здесь один элемент
while j <= i + (candles_quantity - 1):              # Цикл перебора n-свечек
    temp_row = all_list[j].split(";")
    ohlc_row = temp_row[2:6]
    k = 0
    while k <= 3:                                   # Цикл перебора "Open-High-Low-Close"
        ohlc_dataset.append(float(ohlc_row[k]))
        k += 1
    volume_dataset.append(float(temp_row[6]))
    change_dataset.append(float(temp_row[7]))
    j += 1

# Дата последней записи в файле данных
last_date = all_list[j - 1].split(";")[0]

# Итоговый список одного набора данных для тренировки
inputs = ohlc_dataset + volume_dataset + change_dataset

# Подготовка для нормальзиции входных данных: определение диаппазона между макс. и  мин.;
# и приведение к диаппазону 0.01...1.0
inputs_min = min(inputs)
inputs_max = max(inputs)
inputs_range = inputs_max + m.fabs(inputs_min)
       
# Преобразование списка в массив numpy и нормализация
inputs = np.asfarray(inputs)
inputs = ((inputs + m.fabs(inputs_min)) / inputs_range) * 0.99 + 0.01
    
# Опрос сети на тестовых данных
outputs = nn.query(inputs)

i += 1

print("Дата последней записи в файле данных:", last_date)

if outputs > 0:
    print("Предполагаемое движение цены 'завтра' - РОСТ. Результат сети:", outputs)
elif outputs < 0:
    print("Предполагаемое движение цены 'завтра' - СНИЖЕНИЕ. Результат сети:", outputs)
else:
    print("Результат сети:", outputs)

plt.plot(errors_array[2000000:])
plt.grid()
plt.show()
