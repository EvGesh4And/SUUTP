import numpy as np


def implement_transfer_function(control_mass, time):
    # character_indicator: характер поведения передаточной функции
    # k: коэффициент усиления
    # alpha: параметр передаточной функции
    # beta: параметр передаточной функции
    # Размер time
    n = time.size
    character_indicator = control_mass[1]
    tz = control_mass[2]
    k = control_mass[3]
    alpha = control_mass[4]
    beta = control_mass[5]
    cv_plot1 = np.array([])
    for i in range(n):
        if time[0] < tz:
            cv_plot1 = np.append(cv_plot1, 0)
            time = np.delete(time, 0)

    if len(time) != 0:
        # Время начинается с нулевой отметки
        time = time - min(time)
        cv_plot2 = []
        if character_indicator == 'growth':
            if beta == 0:
                cv_plot2 = k * (1 - np.exp(-alpha * time))
            else:
                cv_plot2 = k * (1 - np.exp(-alpha * time) * (alpha * np.sin(beta * time) / beta + np.cos(beta * time)))
        if character_indicator == 'decrease':
            cv_plot2 = k * np.exp(-alpha * time) - k
        cv_plot = np.hstack((cv_plot1, cv_plot2))
    else:
        cv_plot = cv_plot1

    return cv_plot


def implement_transfer_function_x(character_indicator, tz,  k, alpha, beta, time):
    # character_indicator: характер поведения передаточной функции
    # k: коэффициент усиления
    # alpha: параметр передаточной функции
    # beta: параметр передаточной функции
    # Размер time
    if time < tz:
        cv_plot = 0
    else:
        time = time - tz
        # Время начинается с нулевой отметки
        if character_indicator == 'growth':
            if beta == 0:
                cv_plot = k * (1 - np.exp(-alpha * time))
            else:
                cv_plot = k * (1 - np.exp(-alpha * time) * (alpha * np.sin(beta * time) / beta + np.cos(beta * time)))
        if character_indicator == 'decrease':
            cv_plot = k * np.exp(-alpha * time) - k
    return cv_plot


def deter_param_transfer_function(time, cv):
    print(cv)
    print(time)
    """
    Метод определяет параметры колебательного звена 2-го порядка(частный случай - апериодическое звено 1-го порядка)
    с помощью методов и классов библиотеки numpy
    :param time: вектор моментов времени замера показателя CV
    :param cv: вектор значений реакции CV на единичный импульс
    :return: характерное поведение передаточной функции, параметры передаточной функции и строка для конфиг. файла
    :rtype: list, str
    """
    # Рассматривается только прирост/падение значения
    cv = cv - min(cv)
    # Время начинается с нулевой отметки
    time = time - min(time)
    # Количество измерений
    n = cv.size
    # Начальное значение контролируемого параметра
    init_value = cv[0]
    # Значение стабилизации контролируемого параметра
    stable_value = (cv[n-1]+cv[n-2]+cv[n-3]+cv[n-4])/4
    # Максимальное значение
    max_cv = max(cv)
    # Определение времени задержки tz
    # определим как момент времени, когда значение cv отличается
    # от stable_value на 5 %
    tz = 0

    for i in range(n):
        if np.abs(cv[0] - cv[1]) < np.abs(stable_value - init_value) * 0.05:
            tz = time[0]
            time = np.delete(time, 0)
            cv = np.delete(cv, 0)
        else:
            i = n

    # Количество измерений
    n = cv.size
    # Время начинается с нулевой отметки
    time = time - min(time)
    # Определение характера изменения
    if init_value < stable_value:
        character_indicator = 'growth'
    else:
        character_indicator = 'decrease'

    # Параметры передаточной функции
    alpha = 0
    beta = 0
    # Передаточная функция 1-го и 2-го порядка (роста)
    transfer_name = ''
    if character_indicator == 'growth':
        k = stable_value
        if abs(stable_value-max_cv) < 0.05*stable_value:
            transfer_name = "type1"
            beta = 0
            # Определение параметра alpha
            value_search = (1-np.exp(-1))*stable_value
            try:
                for i in range(n):
                    if cv[i] > value_search:
                        if time[i] != 0:
                            alpha = 1/time[i]
                        raise StopIteration
            except StopIteration:
                pass
        else:
            transfer_name = "type2"
            time_ext = 1
            try:
                for i in range(n):
                    if cv[i] == max_cv:
                        time_ext = time[i]
                        raise StopIteration
            except StopIteration:
                pass
            beta = np.pi/time_ext
            alpha = -(beta/np.pi)*np.log((max_cv-stable_value)/stable_value)

    # Передаточная функция 1-го порядка (убывания)
    if character_indicator == 'decrease':
        transfer_name = "type3"
        k = init_value
        beta = 0
        value_search = k*np.exp(-1)
        # Определение параметра alpha
        try:
            for i in range(n):
                if cv[i] < value_search:
                    if time[i] != 0:
                        alpha = 1/time[i]
                    raise StopIteration
        except StopIteration:
            pass

    # Определение момента стабилизации
    t_min = min(time)
    t_max = max(time)
    nn = 100
    t = np.linspace(t_min, t_max, nn)
    tt = 0
    if character_indicator == 'growth':
        if beta == 0:
            tt = - np.log(0.01 / k) / alpha
        else:
            tt = t[n - 1]
            try:
                for i in range(nn - 1, 0, -1):
                    if abs(k - implement_transfer_function_x(character_indicator, tz, k, alpha, beta, tt)) / k >= 0.01:
                        tt = t[i]
                        raise StopIteration
            except StopIteration:
                pass
    else:
        tt = - np.log(0.01)/alpha
    # Строка для конфигурационного файла
    str_csv = transfer_name + '_' + character_indicator + '_' + str(tz) + '_' + str(k) + '_' + str(alpha)\
              + '_' + str(beta) + '_' + str(tt)
    # Возвращаем массив
    #transfer_name:
    #"type1" = "Передаточная функция 1-го порядка (роста)"
    #"type2" = "Передаточная функция 2-го порядка"
    #"type3" = "Передаточная функция 1-го порядка (убывания)"
    return [transfer_name, character_indicator, tz, k, alpha, beta, tt], str_csv