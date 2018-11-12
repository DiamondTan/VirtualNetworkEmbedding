from src import node_and_edge as ne
import networkx as nx

'''
虚拟网请求类
__init__:构造函数，从filepath路径中读取虚拟网请求文件
id：虚拟网请求id
'''
class Vnr:
    def __init__(self, filepath, id):
        self.id = id  # 虚拟网请求id
        self.time = 0  # 到达时间
        self.duration = 0  # 持续时间
        self.vnode = []  # 虚拟请求节点
        self.vedge = []  # 虚拟请求链路
        with open(filepath) as filevnr:
            vnrmsg = []
            while True:
                lines = filevnr.readline()
                if not lines:
                    break
                p_tmp = [float(value) for value in lines.split()]
                vnrmsg.append(p_tmp)

            self.time = vnrmsg[0][3]
            self.duration = vnrmsg[0][4]

            vnodenumber = vnrmsg[0][0]
            vedegenumber = vnrmsg[0][1]

            for i in range(int(vnodenumber)):
                vnode = ne.Vnode(i, vnrmsg[i+1][2], vnrmsg[i+1][0:2])  # 节点编号，cpu，节点坐标
                self.vnode.append(vnode)

            n = len(self.vnode)
            for i in range(int(vedegenumber)):
                vedge = ne.Vedge(i, vnrmsg[n+1+i][2], vnrmsg[n+1+i][0:2])  # 链路编号，链路带宽，链路两端节点
                self.vedge.append(vedge)

    '''
    getEdegeindex：获取边索引
    u，v：链路节点
    '''
    def getEdegeindex(self, u, v):
        for i in range(len(self.vedge)):
            if (self.vedge[i].link[0] == u and self.vedge[i].link[1] == v) \
                or (self.vedge[i].link[0] == v and self.vedge[i].link[1] == u):
                return i
            else:
                return []

    '''
    构造图：虚拟网
    '''
    def vn2networkG(self):
        g = nx.Graph()
        n = len(self.vnode)  # 节点个数
        for i in range(n):
            g.add_node(i, cpu = self.vnode[i], pos = self.vnode[i].position)

        en = len(self.vedge) # 边条数
        for i in range(en):
            bandwidth = 0
            g.add_weighted_edges_from([(self.vedge[i].link[0], self.vedge[i].link[1], self.vedge[i].bandwidth)])

        return g

'''
虚拟请求类
'''
class VNRS:
    '''
    构造函数
    从文件中读取虚拟请求
    加载虚拟网请求
    '''
    def __init__(self, vnr_file_path, n):
        self.vnr = []
        for i in range(n):
            vnodef = vnr_file_path + "\\req" + str(i) + ".txt"
            self.vnr.append(Vnr(vnodef, i))

    # 输出虚拟请求信息
    def msg(self):
        n = len(self.vnr)
        print('vnr:', n)
        for i in range(n):
            print('---------------------------')
            self.vnr[i].msg()
