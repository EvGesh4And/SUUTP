import numpy as np
from scipy.optimize import minimize
import csv
import math
from scipy.optimize import Bounds
from scipy.optimize import LinearConstraint


def reading_configuration_file():
    """
    Метод считывает из конфигурационных файлов Sens и Control данные, которые были
    сформированы в соответствующих модулях
    :return: параметры, которые характеризуют виртуальные анализаторы, а также переходные функции
    """

    # Считывание формулы модели и названий CV в исходном виде и в виде Xi из конфигурационного файла модуля Sens
    model_sens_str = ''
    comparison_short = ''
    comparison_long = ''
    with open('Sens', encoding='utf-8') as Sens_file:
        file_reader_Sens = csv.reader(Sens_file, delimiter=",")
        sh = 0
        for row in file_reader_Sens:
            if sh == 0:
                model_sens_str = row[0]
            if sh == 1:
                comparison_short = row
            if sh == 2:
                comparison_short = [comparison_short, row]
            if sh == 3:
                comparison_long = row
            if sh == 4:
                comparison_long = [comparison_long, row]
            sh += 1
    Sens_file.close()

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

    with open('Control_MV', encoding='utf-8') as Control_MV_file:
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
                    if str_per[0] != 'no_dependency':
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

                dependency.append(dependency_per)
                transfer_name.append(transfer_name_per)
                character_indicator.append(character_indicator_per)
                tz.append(tz_per)
                k.append(k_per)
                alpha.append(alpha_per)
                beta.append(beta_per)
                T.append(T_per)
    Control_MV_file.close()

    # Считывание данных о переходных функциях для MV из конфигурационного файла модуля Control
    dv_Control = []
    dependency_dv = []
    transfer_name_dv = []
    character_indicator_dv = []
    tz_dv = []
    k_dv = []
    alpha_dv = []
    beta_dv = []
    T_dv = []

    with open('Control_DV', encoding='utf-8') as Control_DV_file:
        file_reader_Control = csv.reader(Control_DV_file, delimiter=",")
        sh = 0
        n_dv = 0
        for row in file_reader_Control:
            dependency_dv_per = []
            transfer_name_dv_per = []
            character_indicator_dv_per = []
            tz_dv_per = []
            k_dv_per = []
            alpha_dv_per = []
            beta_dv_per = []
            T_dv_per = []
            if sh == 0:
                row.pop(0)
                dv_Control = row
                n_dv = len(dv_Control)
                sh += 1
            else:
                for i in range(n_dv):
                    print(i)
                    print(row)
                    str_per = row[i+1].split("_")
                    if str_per[0] != 'no_dependency':
                        dependency_dv_per.append(1)
                        transfer_name_dv_per.append(str_per[0])
                        character_indicator_dv_per.append(str_per[1])
                        tz_dv_per.append(float(str_per[2]))
                        k_dv_per.append(float(str_per[3]))
                        alpha_dv_per.append(float(str_per[4]))
                        beta_dv_per.append(float(str_per[5]))
                        T_dv_per.append(float(str_per[6]))
                    else:
                        dependency_dv_per.append(0)

                dependency_dv.append(dependency_dv_per)
                transfer_name_dv.append(transfer_name_dv_per)
                character_indicator_dv.append(character_indicator_dv_per)
                tz_dv.append(tz_dv_per)
                k_dv.append(k_dv_per)
                alpha_dv.append(alpha_dv_per)
                beta_dv.append(beta_dv_per)
                T_dv.append(T_dv_per)
    Control_DV_file.close()

    return model_sens_str, comparison_short, comparison_long, mv_Control, cv_Control, dependency, \
           transfer_name, character_indicator, tz, k, alpha, beta, T, dv_Control, dependency_dv, transfer_name_dv, \
           character_indicator_dv, tz_dv, k_dv, alpha_dv, beta_dv, T_dv


def implement_transfer_function(character_indicator, k, alpha, beta, time):
    """
    Метод по заданным параметрам возвращает вектор значений передаточной функции для вектора времени time
    с помощью методов и классов библиотеки numpy
    :param character_indicator: характер поведения передаточной функции
    :type character_indicator: str
    :param k: коэффициент усиления
    :type k: float
    :param alpha: параметр передаточной функции
    :type alpha: float
    :param beta: параметр передаточной функции
    :type beta: float
    :param time: вектор времени
    :type time: numpy.ndarray
    :return: значения переходной функции с заданными параметрами в моменты времени time
    :rtype: numpy.ndarray
    """
    cv = []
    if character_indicator == 'growth':
        if beta == 0:
            cv = k*(1-np.exp(-alpha*time))
        else:
            cv = k*(1-np.exp(-alpha*time)*(alpha*np.sin(beta*time)/beta+np.cos(beta*time)))
    if character_indicator == 'decrease':
        cv = k * np.exp(-alpha * time) - k
    return cv


def determination_n_s(delta_T, dependency, character_indicator, k, alpha, beta, T, mv_Control, cv_Control):
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
                    w.append(implement_transfer_function(character_indicator[i][j], k[i][j], alpha[i][j], beta[i][j],\
                    delta_T*(p+1)))
                s_per.append(w)
        s.append(s_per)

    return mass_n, n_max, s


def establishing_connection_cv_mv(mass_n, n_max, s, mv_Control, cv_Control, N):
    n_mv = len(mv_Control)
    n_cv = len(cv_Control)

    a, len_mv_value = mv_value.shape
    b = len_mv_value - n_max

    mv_x = []
    for i in range(n_mv):
        mv_x_per = []
        for j in range(N):
            mv_x_per.append('x['+str(j+i*N)+']')
        mv_x.append(mv_x_per)

    mass_bounds = []
    mass_cv_str = []
    for i in range(n_cv):
        cv_str_per = []
        for k in range(1, N+1):
            mass_bounds_per = []
            str_per = ''
            for j in range(n_mv):
                delta_mv_abc = []
                n = mass_n[i][j]
                for w in range(k):
                    delta_mv_abc.append(mv_x[j][w])
                sch = 0
                for w in range(k):
                    if w < n:
                        mass_bounds_per.append(s[i][j][w])
                        if s[i][j][w] > 0:
                            str_per += "+" + str(s[i][j][w]) + '*' + delta_mv_abc[k-w-1]
                        if s[i][j][w] < 0:
                            str_per += str(s[i][j][w]) + '*' + delta_mv_abc[k-w-1]
                    else:
                        mass_bounds_per.append(s[i][j][n-1])
                        if s[i][j][n-1] > 0:
                            str_per += "+" + str(s[i][j][n-1]) + '*' + delta_mv_abc[k-w-1]
                        if s[i][j][n-1] < 0:
                            str_per += str(s[i][j][n-1]) + '*' + delta_mv_abc[k-w-1]
                    sch += 1
                for w in range(k, N):
                    mass_bounds_per.append(0)
            cv_str_per.append(str_per)
            mass_bounds.append(mass_bounds_per)
        mass_cv_str.append(cv_str_per)
    return mass_cv_str, mass_bounds


def core_optimize(mv_value, mass_cv_str_abc, mass_bounds, n_mv, n_cv, n_max, N, mv_left_bounds, mv_right_bounds,\
                  cv_left_bounds, cv_right_bounds):
    # Определение размеров массива данны MV (мин. длинна n_max)
    a, len_mv_value = mv_value.shape
    b = len_mv_value - n_max

    # Заполнение массива разности MV (длина: n_max-1)
    delta_mv_value = np.zeros((n_mv, n_max - 1), int)
    for i in range(n_mv):
        for j in range(n_max - 1):
            delta_mv_value[i, j] = mv_value[i, b + j + 1] - mv_value[i, b + j]

    # Определение значения числовой переменной для CV
    mass_sum = []
    for i in range(n_cv):
        mass_sum_per = []
        for k in range(1, N + 1):
            sum = 0
            for j in range(n_mv):
                n = mass_n[i][j]
                if k >= n:
                    sum += s[i][j][n - 1] * mv_value[j, len_mv_value - 1]
                else:
                    m = n - k
                    w = 0
                    while w != m:
                        sum += s[i][j][k + w] * delta_mv_value[j, n_max - 2 - w]
                        w += 1
                    sum += s[i][j][n - 1] * mv_value[j, len_mv_value - 1 - w]
            mass_sum_per.append(sum)
        mass_sum.append(mass_sum_per)

    # Делаем замены Xi в целевой функции на формулу с x[j]
    n_comparison_short = len(comparison_short[0])

    fun_model = ''
    for i in range(N):
        model_sens_str_per = model_sens_str
        for j in range(n_cv):
            for k in cv_Control:
                for w in range(n_comparison_short):
                    if k == comparison_short[0][w]:
                        model_sens_str_per = model_sens_str_per.replace(comparison_short[1][w],\
                        '(' + str(mass_sum[j][i]) + mass_cv_str_abc[j][i] + ')')
        fun_model += '+' + '(' + str(k_treb) + '-(' + model_sens_str_per + '))**2'

    fun = lambda x: eval(fun_model)

    # Границы для MV
    mv_left_bounds_full = []
    mv_right_bounds_full = []

    n_mv = len(mv_Control)
    for i in range(n_mv):
        for j in range(N):
            mv_left_bounds_full.append(mv_left_bounds[i])
            mv_right_bounds_full.append(mv_right_bounds[i])

    bounds = Bounds(mv_left_bounds_full, mv_right_bounds_full)

    # Границы для CV
    cv_left_bounds_full = []
    cv_right_bounds_full = []

    n_cv = len(cv_Control)
    for i in range(n_cv):
        for j in range(N):
            cv_left_bounds_full.append(cv_left_bounds[i]-mass_sum[i][j])
            cv_right_bounds_full.append(cv_right_bounds[i]-mass_sum[i][j])

    linear_constraint = LinearConstraint(mass_bounds, cv_left_bounds_full, cv_right_bounds_full)

    x0 = np.zeros(n_mv*N)

    res = minimize(fun, x0, method='trust-constr', bounds=bounds,  constraints=linear_constraint)
    x = res.x

    mv_x_predict = []
    for i in range(n_mv):
        mv_x_predict.append(x[i * N])

    return x, mv_x_predict


if __name__ == "__main__":

    # Формирование данных из config файлов Sens и Control
    model_sens_str, comparison_short, comparison_long, mv_Control, cv_Control, dependency, \
    transfer_name, character_indicator, tz, k, alpha, beta, T, dv_Control, dependency_dv, transfer_name_dv, \
    character_indicator_dv, tz_dv, k_dv, alpha_dv, beta_dv, T_dv = reading_configuration_file()

    # Интервал между выводом оптим. значений в БД
    delta_T = 10

    # "Шаги в будущее"
    N = 5

    # Массив MV из БД длины как минимум n_max
    mv_value = np.array([[2, 5, 6, 8, 10, 11, 12, 15, 14, 13, 16, 13, 14, 16], \
                         [2, 3, 4, 5, 7, 8, 9, 10, 8, 9, 10, 8, 9, 9]])

    mass_n, n_max, s = determination_n_s(delta_T, dependency, character_indicator, k, \
                                         alpha, beta, T, mv_Control, cv_Control)

    mass_cv_str_abc, mass_bounds = establishing_connection_cv_mv(mass_n, n_max, s, mv_Control, cv_Control, N)

    k_treb = 500

    n_cv = len(cv_Control)
    n_mv = len(mv_Control)

    # Границы для MV
    mv_left_bounds = [0, 0]
    mv_right_bounds = [20, 10]

    # Границы для CV
    cv_left_bounds = [0, 0, 0, 0]
    cv_right_bounds = [200, 400, 600, 1400]

    # Границы для
    x, mv_x_predict = core_optimize(mv_value, mass_cv_str_abc, mass_bounds, n_mv, n_cv, n_max, N, mv_left_bounds,
                                    mv_right_bounds, \
                                    cv_left_bounds, cv_right_bounds)