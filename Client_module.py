"""
Модуль Клиентского приложения
"""

from opcua import Client


class ClientApp:

    """
        Класс клиентского приложения
    """

    def __init__(self, url, path_list, timeout = 4):
        """
        :param url: url сервера

        :param path_list: Список абсолютных путей к необходимым узлам

        Функция подключает клиента к OPC-UA серверу и заполняет словарь nodeid_dict,
        у которого ключ -- название узла
                   значение -- [id узла, словарь подобный nodeid_dict только для дочерних узлов]
        либо, если у узла нет дочерних, то
                   значение -- [id узла, None]
        """
        self.url = url
        self.client = Client(self.url, timeout)
        self.client.connect()
        self.nodeid_dict = dict()
        self.completing_NodeID_dict(path_list)


    def client_connect(self):
        """
        Функция подключения клиента к OPC-UA серверу
        """
        self.client.connect()


    def client_disconnect(self):
        """
        Функция отключения клиента от OPC-UA сервера
        """
        self.client.close_session()


    def completing_NodeID_dict(self, path_list):
        """
        :param path_list: Список абсолютных путей к необходимым узлам

        Функция заполняет словарь nodeid_dict
        """
        for name in path_list:
            split_name = name.split('.')
            node = self.client.get_objects_node()
            split_name.pop(0)
            for dir in split_name:
                child_node_list = node.get_children()
                for child_node in child_node_list:
                    if dir == child_node.get_display_name().Text:
                        node = child_node
            if node.get_children():
                self.nodeid_dict[node.get_display_name().Text] = [node.nodeid.to_string(), self.reccursive(node)]
            else:
                self.nodeid_dict[node.get_display_name().Text] = [node.nodeid.to_string(), None]


    def reccursive(self, node):
        """
        :param node: экземпляр класса Node

        Рекурсивная функция, которая возвращает словарь подобный nodeid_dict
        для дочерних узлов узла node
        """
        child_node_list = node.get_children()
        if child_node_list:
            nodeid = dict()
            for child_node in child_node_list:
                nodeid[child_node.get_display_name().Text] = [child_node.nodeid.to_string(), self.reccursive(child_node)]
            return nodeid
        else:
            return None


    def get_value(self, path):
        """
        :param path: относительный от названия переменной путь до узла

        Функция возвращает при успехе значение узла, при ошибке None
        """
        try:
            node = self.client.get_node(self.get_NodeId_from_Path(path))
            return node.get_value()
        except:
            # Bad_AttributeIdInvalid
            return None


    def get_NodeId_from_Path(self, path):
        """
        :param path: относительный от названия переменной путь до узла

        Функция возвращает NodeId узла
        """
        levels = path.split('.')
        list = self.nodeid_dict[levels.pop(0)]
        for level in levels:
            list = list[1][level]
        return list[0]


    def get_values(self, path_list):
        """
        :param path_list: список относительных от названия переменной путей до узла

        Функция возвращает при успехе список значений узлов, при возникновении ошибки None
        """
        try:
            nodes = list()
            for path in path_list:
                nodeid = self.get_NodeId_from_Path(path)
                nodes.append(self.client.get_node(nodeid))
            return self.client.get_values(nodes)
        except:
            None


    def set_value(self, path, value):
        """
        :param path: относительный от названия переменной путь до узла

        :param value: значение

        Функция обновляет значения узла
        """
        nodeid =  self.get_NodeId_from_Path(path)
        node = self.client.get_node(nodeid)
        node.set_value(value, varianttype=node.get_data_type_as_variant_type())



    def set_values(self, path_list, values):
        """
        :param path_list: список относительных от названия переменной путей до узла

        :param values: список значений

        Функция обновляет значения узлов
        """
        for path, value in zip(path_list, values):
            nodeid =  self.get_NodeId_from_Path(path)
            node = self.client.get_node(nodeid)
            node.set_value(value, varianttype=node.get_data_type_as_variant_type())