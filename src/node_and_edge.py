'''
节点类
index：节点编号
cpu：节点CPU计算资源
position：节点位置
'''
class Node:
    def __init__(self, index, cpu , position):
        self.index = index
        self.cpu = cpu
        self.position = position


'''
虚拟节点，继承node
'''
class Vnode(Node):
    def __init__(self, index, cpu, position):
        Node.__init__(self, index, cpu, position)

    def msg(self):
        print('vnode', self.index, 'cpu', self.cpu, 'position', self.position)

'''
底层节点，继承node
lastcpu:剩余CPU
vnodeindexs:虚拟网节点序列，格式[虚拟网id，虚拟节点id]
open:链路是否开启，布尔值
mappable_flag:是否可以映射，布尔值
'''
class Snode(Node):
    def __init__(self, index, cpu, position):
        Node.__init__(self, index, cpu, position)
        self.lastcpu = cpu
        self.vnodeindexs = []
        self.open = False
        self.mappable_flag = False
        self.baseEnergy = 15
        self.maxEnergy = 300

    def a(self, **args):
        # if args.__len__() == 1:
        print(args)


    def msg(self):
        print('snode:', self.index, 'cpu:', self.cpu, 'lastcpu:',
              self.lastcpu, 'vnodeindexs:',self.vnodeindexs, 'position', self.position)


'''
链路类
index：链路编号
bandwidth：链路带宽
link:链路,使用两端节点表示
'''
class Edge:
    def __init__(self, index, bandwidth, a_2_b):
        self.index = index
        self.bandwidth = bandwidth
        self.link = (int(a_2_b[0]), int(a_2_b[1]))

'''
虚拟请求链路类，继承edge
'''
class Vedge(Edge):
    def msg(self):
        print('vedge', self.index, 'bandwidth', self.bandwidth, 'nodeindex', self.link)

'''
底层网络链路类，继承edge
lastbandwidth：剩余带宽
vedgeindexs：虚拟链路编号
open：是否开启
mappable_flag：是否可以映射标志
'''
class Sedge(Edge):
    def __init__(self, index, bandwidth, a_2_b):
        Edge.__init__(self, index, bandwidth, a_2_b)
        self.lastbandwidth = bandwidth
        self.vedgeindexs = []
        self.open = False
        self.mappable_flag = False

    def msg(self):
        print('sedge:', self.index, 'bandwidth:',self.bandwidth, 'lastbandwidth:',
              self.lastbandwidth, 'link:', self.link, 'vedgeindexs', self.vedgeindexs)

