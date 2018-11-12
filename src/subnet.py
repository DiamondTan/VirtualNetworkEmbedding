from src import node_and_edge as ne
import networkx as nx
import copy as cp
from src import algorithm as al

'''
底层网络类
函数说明：

'''
class SubstrateNetwork:
    # 从文件中读取节点
    def __init__(self, snfilepath):
        self.snode = []  # 底层节点列表
        self.sedge = []  # 底层边列表
        self.p = 0  # 目前底层网络能耗
        self.t = 0  # 目前时刻，为了记录何时的p
        with open(snfilepath) as filesn:
            lines = filesn.readline()
            p_tmp = [float(value) for value in lines.split()]
            snodenumber = int(p_tmp[0])  # 底层节点个数
            sedgenumber = int(p_tmp[1])  # 底层链路条数

            # 读取每个节点信息
            for i in range(snodenumber):
                lines = filesn.readline()
                if not lines:
                    break

                p_tmp = [float(value) for value in lines.split()]
                sn = ne.Snode(i, p_tmp[2], p_tmp[0:2])  # 节点编号，节点CPU，节点位置
                self.snode.append(sn)

            # 读取边信息
            for i in range(sedgenumber):
                lines = filesn.readline()
                if not lines:
                    break

                p_tmp = [float(value) for value in lines.split()]
                sedge = ne.Sedge(i, p_tmp[2], p_tmp[0:2])
                self.sedge.append(sedge)

    '''
    构造图：底层网络
    返回值：图g
    '''
    def sn2networkG(self, bandlimit = 0):
        g = nx.Graph()
        n = len(self.snode)  # 节点个数
        for i in range(n):
            g.add_node(i, cpu = self.snode[i].lastcpu, pos = self.snode[i].position)

        en = len(self.sedge) # 边条数
        for i in range(en):
            bandwidth = 0
            if self.sedge[i].bandwidth > bandlimit:
                g.add_weighted_edges_from([(self.sedge[i].link[0], self.sedge[i].link[1], self.sedge[i].bandwidth)])

        return g

    '''
    get_subnet_link: 获取底层链路信息
    get_flag: 获取链路信息的方式，默认0
              如果get_flag=0，那么按照链路索引进行查找链路信息
              如果get_flag=1，那么按照链路两端点进行查找链路信息
    args: 获取的链路索引或端点信息
    返回值：列表；
        查找成功：链路编号、链路带宽了、链路两端节点编号
        查找失败：空
    '''
    def get_subnet_link(self, get_flag = 0, value = []):
        if get_flag == 0:
            for i in range(len(self.sedge)):
                if self.sedge[i].index == value[0]:
                    return [self.sedge[i].index, self.sedge[i].bandwidth, self.sedge[i].link]

        elif get_flag == 1:
            for i in range(len(self.sedge)):
                if (self.sedge[i].link[0] == value[0] and self.sedge[i].link[1] == value[1]) \
                    or (self.sedge[i].link[0] == value[1] and self.sedge[i].link[1] == value[0]):
                    return [self.sedge[i].index, self.sedge[i].bandwidth, self.sedge[i].link]

        return []

    '''
    get_node: 获取节点信息
    返回值：字典；
            查找成功：
            index: 节点索引
            lastcpu：剩余cpu
            cpu：cpu资源
            vnodeindexs：虚拟网映射索引
            state：节点开启状态状态，True：开启；False：关闭
            查找失败:
            {}
    '''
    def get_node(self, node_index = -1):
        for i in range(len(self.snode)):
            if self.snode[i] == node_index:
                return {"index": self.snode[i].index,
                        "lastcpu": self.snode[i].lastcpu,
                        "cpu": self.snode[i].position,
                        "vnodeindexs": self.snode[i].vnodeindexs,
                        "state": self.snode[i].open
                        }
        else:
            return {}

    def getedge(self, source_node, target_node):
        for i in range(len(self.sedge)):
            for j in range(len(self.sedge[i].link)):
                if (source_node == self.sedge[i].link[0] and target_node == self.sedge[i].link[1]) or\
                    (source_node == self.sedge[i].link[1] and target_node == self.sedge[i].link[0]):
                    return self.sedge[i].index
        return []

    '''
    get_node_neighbors：获取邻居节点
    '''
    def get_node_neighbors(self, n):
        g  = self.sn2networkG()
        iternode = g.neighbors()
        nodes = [x for x in iternode]
        return nodes

    '''
    testnodemapping：虚拟网映射
    参数：
        vnr：虚拟网
        node_map_index：映射序列,一维数组
    返回值：字典
        map_result：映射结果，成功|失败
        nodemapindex：映射索引，映射到该物理节点的虚拟网以及虚拟节点索引，格式[虚拟网id，虚拟节点id]
        snode：底层物理节点
        sedge：底层链路
    '''
    def testnodemapping(self,vnr, node_map_index):
        asnode = cp.deepcopy(self.snode)
        asedge = cp.deepcopy(self.sedge)
        success = True
        n = len(vnr.vnode)
        sn = len(asnode)
        for i in range(n):
            # 如果虚拟节点CPU小于映射的物理节点CPU并且虚拟网id不在物理节点映射索引里
            if (vnr.vnode[i].cpu < asnode[node_map_index[i]].lastcpu) and (vnr.id not in [x[0] for x in asnode[node_map_index[i]].vnodeindexs]):
                asnode[node_map_index[i]].lastcpu = asnode[node_map_index[i]].lastcpu - vnr.vnode[i].cpu
                asnode[node_map_index[i]].vnodeindexs.append([vnr.id, vnr.vnode[i].index])
            else:
                success = False
                return {"map_result":success, "nodemapindex":node_map_index, "snode":asnode, "sedge":asedge}
        return {"map_result":success, "nodemapindex":node_map_index, "snode":asnode, "sedge":asedge}

    def nodemapping(self, vnr, v2sindex):
        vn = len(v2sindex)
        for i in range(vn):
            self.snode[v2sindex[i][1]].lastcpu = self.snode[v2sindex[i][1]].lastcpu - vnr.vnode[v2sindex[i][0]].cpu
            self.snode[v2sindex[i][1]].vnodeindexs.append([vnr.id, vnr.vnode[v2sindex[i][0]].index])
            self.snode[v2sindex[i][1]].open = True
        print('map vnr ', vnr.id, ' success')

    '''
    remove_node_mapping：移除(释放)映射的节点
    按照虚拟节点的id进行查找
    '''
    def remove_node_mapping(self, vnr):
        sn = len(self.snode)
        for i in range(sn):
            xtemp = []
            for x in self.snode[i].vnodeindexs:
                if vnr.id == x[0]:
                    self.snode[i].lastcpu = self.snode[i].lastcpu + vnr.vnode[x[1]].cpu  # 回收映射的CPU
                    xtemp.append(x)
            for x in xtemp:
                    self.snode[i].vnodeindexs.remove(x)
            if not self.snode[i].vnodeindexs:
                self.snode[i].open = False

        print('remove vnr node ', vnr.id, ' success')


    '''
    edgemapping：链路映射
    '''
    def edgemapping(self, vnr, ve2seindex):
        i = 0
        for path in ve2seindex:
            for e in path:
                self.sedge[e].lastbandwidth = self.sedge[e].lastbandwidth - vnr.vedge[i].bandwidth
                self.sedge[e].vedgeindexs.append([vnr.id, vnr.vedge[i].index])
                self.sedge[e].open = True
            i = i + 1
        print('map vnr edge', vnr.id, 'success')

    '''
    remove_edge_mapping：移除(释放)映射的链路
    按照链路索引进行查找
    '''
    def remove_edge_mapping(self, vnr):
        en = len(self.sedge)
        for e in range(en):
            tempx = []
            for x in self.sedge[e].vedgeindexs:
                if vnr.id == x[0]:
                    self.sedge[e].lastbandwidth = self.sedge[e].lastbandwidth + vnr.vedge[x[1]].bandwidth  # 回收链路带宽
                    tempx.append(x)
            for x in tempx:
                self.sedge[e].vedgeindexs.remove(x)
            if not self.sedge[e].vedgeindexs:  # 底层链路没有映射虚拟链路将链路关闭
                self.sedge[e].open = False
        print('remove vnr edge ', vnr.id, ' success')

    def testedegemapping2(self,asnode, asedge, vnr, v2sindex):
        success, ve2seindex, asnode, asedge = al.edegemapping2(asnode, asedge, vnr, v2sindex)
        return success, ve2seindex, asnode, asedge


    '''
    power:计算能耗
    '''
    def power(self, time):
        plink = 15  # 链路功耗w
        pmnode = 300  # 节点最大能耗w
        pbnode = 150  # 节点基本能耗w
        totalp = 0
        # 计算节点电能
        for snode in self.snode:
            if snode.open:
                totalp = totalp + (pmnode - pbnode)*(snode.cpu - snode.lastcpu)/snode.cpu + pbnode
        # 计算链路电能
        for sedge in self.sedge:
            if sedge.open:
                totalp = totalp + plink
        dtenergy = self.p * (time - self.t)
        self.t = time  # 当前时刻
        self.p = totalp  # 当前时刻底层网络总能耗
        return dtenergy




