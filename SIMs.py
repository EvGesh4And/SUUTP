import numpy as np
import csv
import math
import random


def reading_configuration_file(config_file_MV = "Control_MV"):
    """
    Метод считывает из конфигурационных файлов Sens и Control данные, которые были
    сформированы в соответствующих модулях
    :return: параметры, которые характеризуют виртуальные анализаторы, а также переходные функции
    """

    # Считывание данных о переходных функциях для MV из конфигурационного файла модуля Control
    mv_Control = []
    cv_Control = []
    dependency = []
    transfer_name = []
    character_indicator = []
    tz = []
    k = []
    alpha = []
    beta = []
    T = []

    with open(config_file_MV, encoding='utf-8') as Control_MV_file:
        file_reader_Control = csv.reader(Control_MV_file, delimiter=",")
        sh = 0
        n_mv = 0
        for row in file_reader_Control:
            dependency_per = []
            transfer_name_per = []
            character_indicator_per = []
            tz_per = []
            k_per = []
            alpha_per = []
            beta_per = []
            T_per = []

            if sh == 0:
                row.pop(0)
                mv_Control = row
                n_mv = len(mv_Control)
                sh += 1
            else:
                cv_Control.append(row[0])
                for i in range(n_mv):
                    str_per = row[i+1].split("_")
                    if str_per[0] != 'independent':
                        dependency_per.append(1)
                        transfer_name_per.append(str_per[0])
                        character_indicator_per.append(str_per[1])
                        tz_per.append(float(str_per[2]))
                        k_per.append(float(str_per[3]))
                        alpha_per.append(float(str_per[4]))
                        beta_per.append(float(str_per[5]))
                        T_per.append(float(str_per[6]))
                    else:
                        dependency_per.append(0)
                        transfer_name_per.append('type0')
                        character_indicator_per.append('no')
                        tz_per.append(0.)
                        k_per.append(0.)
                        alpha_per.append(0.)
                        beta_per.append(0.)
                        T_per.append(0.)

                dependency.append(dependency_per)
                transfer_name.append(transfer_name_per)
                character_indicator.append(character_indicator_per)
                tz.append(tz_per)
                k.append(k_per)
                alpha.append(alpha_per)
                beta.append(beta_per)
                T.append(T_per)
    Control_MV_file.close()

    return mv_Control, cv_Control, dependency, transfer_name, character_indicator, tz, k, alpha, beta, T


def implement_transfer_function_x(character_indicator, tz,  k, alpha, beta, time):
    # character_indicator: характер поведения передаточной функции
    # k: коэффициент усиления
    # alpha: параметр передаточной функции
    # beta: параметр передаточной функции
    # Размер time
    cv_plot = 0
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


def determination_n_s(delta_T, dependency, character_indicator, tz, k, alpha, beta, T, mv_Control, cv_Control):
    n_mv = len(mv_Control)
    n_cv = len(cv_Control)
    mass_n = np.zeros((n_cv, n_mv), int)
    s = []
    n_max = 0
    for i in range(n_cv):
        s_per = []
        for j in range(n_mv):
            if dependency[i][j] == 1:
                mass_n[i, j] = math.ceil(T[i][j]/delta_T)
                if mass_n[i, j] > n_max:
                    n_max = mass_n[i, j]
                w = []
                for p in range(0, mass_n[i, j]):
                    w.append(implement_transfer_function_x(character_indicator[i][j], tz[i][j], k[i][j], alpha[i][j], beta[i][j],\
                    delta_T*(p+1)))
                s_per.append(w)
            else:
                s_per.append([])
        s.append(s_per)

    return mass_n, n_max, s


def zn_cv_mv(mass_n, n_max, s, mv_value,  n_mv, n_cv):
    # Заполнение массива разности MV (длина: n_max-1)
    delta_mv_value = np.zeros((n_mv, n_max - 1), float)
    for i in range(n_mv):
        for j in range(n_max - 1):
            delta_mv_value[i, j] = mv_value[i, j + 1] - mv_value[i, j]

    # Определение значения числовой переменной для CV
    cv = []
    for i in range(n_cv):
        cv_per = 0.
        for j in range(n_mv):
            n = mass_n[i][j]
            if n != 0:
                for w in range(n):
                    if w < n-1:
                        cv_per += s[i][j][w] * delta_mv_value[j, n_max - 2 - w]
                    else:
                        cv_per += s[i][j][n - 1] * mv_value[j, n_max - n]
        cv_per = cv_per + (0.5 - random.random()) *  cv_per * 0.03
        cv.append(cv_per)
    return cv

if __name__ == "__main__":

    # Формирование данных из config файлов Sens и Control
    mv_Control, cv_Control, dependency, transfer_name, character_indicator, tz, k, alpha, beta, T = reading_configuration_file()

    n_cv = len(cv_Control)
    n_mv = len(mv_Control)

    # Интервал между выводом значений в БД
    delta_T = 5

    mass_n, n_max, s = determination_n_s(delta_T, dependency, character_indicator, tz, k, \
                                         alpha, beta, T, mv_Control, cv_Control)

    # mv_value
    cv = zn_cv_mv(mass_n, n_max, s, mv_value, n_mv, n_cv)

