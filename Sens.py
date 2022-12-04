import numpy as np
import scipy.stats as sps
import itertools as iter


def function_def(ind, cv):
    """
    Метод применяет заданную функцию с индикатором ind к столбцу cv
    :param ind: индикатор заданной функции
    :param cv: столбец параметра
    :return: значение заданной функции ind с аргументом cv
    """
    return {
        ind == 1: cv ** 0,
        ind == 2: cv ** 1,
        ind == 3: cv ** 2,
        ind == 4: cv ** 3,
        ind == 5: cv ** 4
    }[True]


def function_def_name(ind, param_name):
    """
    Метод вывода конечной формулы с исходными переменными для пользователя
    :param ind: индикатор заданной функции
    :param param_name: название параметра
    :return: вывод строки формулы с исходными переменными
    """
    return {
        ind == 1: '',
        ind == 2: "*" + param_name,
        ind == 3: "*" + param_name + "^2",
        ind == 4: "*" + param_name + "^3",
        ind == 5: "*" + param_name + "^4"
    }[True]


def function_def_name_x(ind, param_name):
    """
    Метод вывода конечной формулы с исходными переменными для конфигурационного файла
    :param ind: индикатор заданной функции
    :param param_name: название параметра
    :return: вывод строки формулы с переменными Xi
    """
    return {
        ind == 1: '',
        ind == 2: "*" + param_name,
        ind == 3: "*" + param_name + "**2",
        ind == 4: "*" + param_name + "**3",
        ind == 5: "*" + param_name + "**4"
    }[True]


def regress(x, y):
    """
    Метод построение линейной регрессии
    с помощью методов и классов библиотеки numpy
    :param x: матрица признаков
    :param y: вектор отликов
    :return: параметры линейно регрессии
    """
    n, m = x.shape
    b = ((np.linalg.inv((x.T).dot(x))).dot(x.T)).dot(y)
    yy = x.dot(b)
    e = y - yy
    s = e.T.dot(e)
    sigma2 = s / (n - m)
    return b, s, sigma2, yy


def student(x, y):
    """
    В методе реализован критерий Стьюдента для определения значимости отдельного фактора в регрессии
    с помощью методов и классов библиотеки scipy, numpy, regress
    :param x: матрица признаков
    :param y: вектор отликов
    :return: вектор с номерами незначимых факторов
    """
    # Определение параметров
    n, m = x.shape
    # Вектор оценок коэффициентов
    b, s, sigma2, yy = regress(x, y)
    # Проверка гипотезы значимости факторов по критерию Стьюдента
    v = np.linalg.inv(x.T.dot(x))
    stud = []
    for i in range(m):
        if v[i, i] >= 0:
            t_real = b[i] / np.sqrt(sigma2 * v[i, i])
        else:
            t_real = 0
        # t_real = b[i] / np.sqrt(sigma2 * v[i, i])
        p = 2 * (1 - sps.t(n - m).cdf(abs(t_real)))
        if p > 0.01:
            stud.append(i)
    return stud


def fisher(x_short, x_long, y, k):
    """
    В методе реализован критерий Фишера для определения значимости группы факторов в регрессии
    с помощью методов и классов библиотеки scipy, regress
    :param x_short: сокращенная матрица признаков
    :param x_long: полная матрица признаков
    :param y: вектор отликов
    :param k: количество исключенных факторов
    :return: bool, если True, то группа исключенных факторов незначима
    """
    # Определение параметров
    n, m = x_long.shape
    alpha = 0.05
    b_long, s_long, sigma2_long, yy_long = regress(x_long, y)
    b_short, s_short, sigma2_short, yy_short = regress(x_short, y)
    f = ((s_short - s_long) / k) / (s_long / (n - m))
    if f < sps.f.ppf(1 - alpha, k, n - m):
        return True
    else:
        return False


def recurrent_verification(x_long, y, stud, k):
    """
    Метод выявления наибольшей по размерности группы факторов для удаления,
    в случае нахождения происходит исключение этой группы
    с помощью методов и классов библиотеки scipy, itertools, fisher, recurrent_verification
    :param x_long: полная матрица признаков
    :param y: вектор отликов
    :param stud: вектор с номерами незначимых факторов по критерию Стьюдента
    :param k: количество исключенных признаков
    :return: сокращенная матрица признаков, количество исключенных факторов
    """
    if k != 0:
        com_set = iter.combinations(stud, k)
        for list_del in com_set:
            x_short = np.delete(x_long, list_del, 1)
            if fisher(x_short, x_long, y, k):
                return x_short, k, list_del
        return recurrent_verification(x_long, y, stud, k - 1)
    else:
        return x_long, k, []


def removing_insignificant_factors(x, y, mass_on, n_cv, n_functions):
    """
    В методе реализовано исключение наибольшей группы незначимых
    факторов на основе критерия Стьюдента и Фишера
    с помощью методов и классов библиотеки student, regress, recurrent_verification
    :param x: матрица признаков
    :param y: вектор отликов
    :param mass_on: массив включения заданной функции с данным аргументом в виде фактора в формуле регрессии
    :param n_cv: количество входных cv
    :param n_functions: количество заданных функций
    :return: сокращенная матрица признаков, значения функции регрессии,
    """
    # Вызов функции критерия Стьюдента
    stud = student(x, y)
    k = len(stud)
    while k != 0:
        # Получение списка исключений на основе критерия Стьюдента и Фишера
        x, k, list_del = recurrent_verification(x, y, stud, k)
        len_del = len(list_del)
        index = 0
        sum_ind = 0
        if k != 0:
            # Исключение незначимых параметров из массива включения
            try:
                for i in range(n_cv):
                    for j in range(n_functions):
                        if mass_on[i, j] != 0:
                            sum_ind += 1
                            if sum_ind == (list_del[index] + 1):
                                mass_on[i, j] = 0
                                index += 1
                        if index == len_del:
                            raise StopIteration
            except StopIteration:
                pass
        # В случае, если было исключение хотя бы одного параметра, то повторная проверка на наличие незначимых факторов
        if k != 0:
            stud = student(x, y)
            k = len(stud)
    b, s, sigma2, yy = regress(x, y)
    return x, yy, b, mass_on, s


def function_name_user(mass_on, n_cv, name_cv):
    """
    Метод формирует строку модели для вывода пользователю
    с помощью методов и классов библиотеки function_def_name
    :param b: вектор коэффициентов
    :param mass_on: массив включения
    :param n_cv: количество входных cv
    :param name_cv: строка названия CV
    :return:
    """
    name_finish = ''
    indic = 0
    nakop = 0
    for i in range(n_cv):
        for j in mass_on[i, :]:
            if j != 0:
                if indic == 0:
                    indic = 1
                else:
                    name_finish += '+'
                name_finish += 'k' + str(nakop + 1)
                nakop += 1
                name_finish += function_def_name(j, name_cv[i])

    return name_finish


def function_name_config(b, mass_on, n_cv, name_cv_x):
    """
    Метод формирует строку модели для конфигурационного файла
    с помощью методов и классов библиотеки function_def_name_x
    :param b: вектор коэффициентов
    :param mass_on: массив включения
    :param n_cv: количество входных cv
    :param name_cv_x: строка названия CV в формате Xi
    :return: строка модели для конфигурационного файла
    """
    name_finish = ''
    indic = 0
    nakop = 0
    for i in range(n_cv):
        for j in mass_on[i, :]:
            if j != 0:
                if indic == 0:
                    indic = 1
                else:
                    if b[nakop] > 0:
                        name_finish += '+'
                name_finish += str(b[nakop])
                nakop += 1
                name_finish += function_def_name_x(j, name_cv_x[i])

    return name_finish


def specified_model(model, cv, y, name_cv_x):
    """
    Метод по заданной вручную модели вычисляет коэффициенты линейной регрессии МНК
    с помощью методов и классов библиотеки numpy, function_def,  removing_insignificant_factors,
    function_name_user, function_name_config
    :param model: строка модели, введенная вручную
    :param cv: матрица признаков CV
    :param y: вектор отликов
    :param name_cv_x: строка наименований CV в формате Xi
    :return: строка для вывода пользователю, строка для конфигурационного файла, вектор для графика,
    коэффициенты регрессии, массив включения
    """
    n, n_cv = cv.shape

    # Встроенные переменные
    # Количество функций в модели: n_functions
    n_functions = 5

    model = model.replace('^', '**')
    model = model.replace(' ', '')
    funct = model.split("+")

    # Определяем массив включения
    mass_on = np.arange(n_cv * n_functions).reshape(n_cv, n_functions)

    # В начальный момент выключены все вариации F_Di(cvj)
    for i in range(n_cv):
        for j in range(n_functions):
            mass_on[i, j] = 0

    # Определяем какие включения были заданы пользователем
    ind = 0
    for slg in funct:
        if '1' == slg:
            mass_on[0, 0] = 1
        for i in reversed(range(n_cv)):
            if name_cv_x[i] in slg:
                ind = 0
                slg = slg.replace(name_cv_x[i], '')
                for j in range(2, n_functions):
                    if str(j) in slg:
                        mass_on[i, j] = j + 1
                        ind = 1
                if ind == 0:
                    mass_on[i, 1] = 2

    # Определяем массив X (да, можно было бы записать его и в цикл выше, но так легче понять суть)
    ind = 0
    x = []
    for i in range(n_cv):
        for j in mass_on[i, :]:
            if j != 0:
                if ind == 0:
                    x = np.vstack(function_def(j, cv[:, i])).T
                    ind = 1
                else:
                    x = np.vstack([x, function_def(j, cv[:, i])])
    x = x.T

    b, s, sigma2, yy = regress(x, y)

    # Для пользователя(вывод)
    name_finish_x_short = function_name_user(mass_on, n_cv, name_cv_x)
    name_finish_x = function_name_config(b, mass_on, n_cv, name_cv_x)
    return name_finish_x_short, name_finish_x, yy, b, mass_on


def proposed_model(cv, y, name_cv_x):
    """
    Метод по заданным параметрам СV составляет модель со значимыми факторами и вычисляет коэффициенты
    линейной регрессии МНК с помощью методов и классов библиотеки numpy, function_def,  removing_insignificant_factors,
    function_name_user, function_name_config
    :param cv: матрица признаков CV
    :param y: вектор отликов
    :param name_cv_x: строка наименований CV в формате Xi
    :return: строка для вывода пользователю, строка для конфигурационного файла, вектор для графика,
    коэффициенты регрессии, массив включения
    """
    # Изменяемые переменные:
    # Количество наблюдений: n
    # Количество входных параметров: n_cv
    n, n_cv = cv.shape
    # Встроенные переменные
    # Количество функций в модели: n_functions
    n_functions = 5

    # Определяем массив включения
    mass_on = np.arange(n_cv * n_functions).reshape(n_cv, n_functions)

    # В начальный момент включены все вариации F_Di(cvj)
    for i in range(n_cv):
        for j in range(n_functions):
            mass_on[i, j] = j + 1
            if i != 0:
                mass_on[i, 0] = 0

    # Определяем массив X (да, можно было бы записать его и в цикл выше, но так легче понять суть)
    ind = 0
    x = []
    for i in range(n_cv):
        for j in mass_on[i, :]:
            if j != 0:
                if ind == 0:
                    x = function_def(j, cv[:, i])
                    ind = 1
                else:
                    x = np.vstack([x, function_def(j, cv[:, i])])
    x = x.T

    x, yy, b, mass_on, s = removing_insignificant_factors(x, y, mass_on, n_cv, n_functions)
    name_finish_x_short = function_name_user(mass_on, n_cv, name_cv_x)
    name_finish_x = function_name_config(b, mass_on, n_cv, name_cv_x)

    return name_finish_x_short, name_finish_x, yy, b, mass_on


def adjustment(b, mass_on, cv, name_cv_x):
    """
    Метод обновления модели после ручной корректировки коэффициента
    :param b: вектор коэффициентов
    :param mass_on: массив включения
    :param cv: матрица признаков CV
    :param name_cv_x: строка наименований CV в формате Xi
    :return: строка модели для конфигурационного файла
    """
    n, n_cv = cv.shape

    # Определяем массив X (да, можно было бы записать его и в цикл выше, но так легче понять суть)
    ind = 0
    x = []
    for i in range(n_cv):
        for j in mass_on[i, :]:
            if j != 0:
                if ind == 0:
                    x = np.vstack(function_def(j, cv[:, i])).T
                    ind = 1
                else:
                    x = np.vstack([x, function_def(j, cv[:, i])])
    x = x.T
    YY = x.dot(b)
    name_finish_x = function_name_config(b, mass_on, n_cv, name_cv_x)
    name_finish_x_short = function_name_user(mass_on, n_cv, name_cv_x)
    return name_finish_x_short, name_finish_x, YY


def sens_predict(name_finish_x, name_cv_x, cv):
    """
    Метод вычисления значения виртуального анализатора
    :param name_finish_x: конечная формула
    :param name_cv_x: строка наименований CV в формате Xi
    :param cv: матрица признаков CV
    :return: строка модели для конфигурационного файла
    """
    # Делаем замены Xi на значения cvi
    n_cv = len(cv)

    for j in range(n_cv):
        name_finish_x = name_finish_x.replace(name_cv_x[j], str(cv[j]))

    return eval(name_finish_x)
