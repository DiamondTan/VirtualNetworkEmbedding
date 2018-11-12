from src import virnet as vn
from src import subnet as sn
import os
import time as ost
import networkx as nx
import numpy as np
from src import algorithm as al
import copy as cp
import xlwt
import matplotlib.pyplot as plt
import random

'''
Simulation：模拟器类

'''
class Simulation:
    def __init__(self):
        self.lnum = 1
        self.VNRS = []  # 虚拟网请求集合
        self.sn = []  # 物理网
        self.timeframe = []  # 时间窗，统计的频率2000
        self.acceptrate = []  # 接收率
        self.revenue_cost = []  # 收益比
        self.node_success_rate = []  # 节点成功率
        self.node_open = []  # 第i个虚拟网持续时间内节点开启个数
        self.edge_success_rate = []  # 链路映射成功率
        self.edge_open = []  # 第i个虚拟网持续时间内链路开启个数
        self.node_reload_balance = []  # 节点负载均衡
        self.edge_reload_balance = []  # 边负载均衡
        self.map_time = []  # 映射时间，算法效率高低，系统时间
        self.t = []  # 记录当前时刻，虚拟网时间
        self.x = []  # 画图的时候用，基本不用，记录x轴
        self.energy = 0  # 单位瓦特，底层能耗
        self.average_energy = []  # 平均能耗
        self.suipianlv = []  # 碎片率，反映底层网络资源分布零散状况

    '''
    初始化函数
    '''
    def init(self, SNpath, vnr_path, vnr_number, frame_time = 1000):
        self.VNRS = vn.VNRS(vnr_path, vnr_number).vnr  # 加载虚拟网络
        self.sn = sn.SubstrateNetwork(SNpath)  # 加载底层网络
        self.timeframe = frame_time
        self.acceptrate = []
        self.revenue_cost  =[]
        self.node_open = []
        self.edge_open = []

        self.node_reload_balance = []
        self.edge_reload_balance = []

        self.suipianlv = []

    '''
    刷新输出文件
    '''
    def fresh_output(self):
        f = open('.\\output\\acceptrate.txt','wb')
        f.close()
        f = open('.\\output\\revenue_cost.txt','wb')
        f.close()

    '''
    将评价指标保存excle文件
    '''
    def save_excle(self):
        j = 0
        workbook = xlwt.Workbook(encoding='utf-8')
        booksheet = workbook.add_sheet('Sheet 1')
        for i, d in zip(range(len(self.acceptrate)), self.acceptrate):
            booksheet.write(j, i, d)
        workbook.save('.\\评价指标\\请求接收率.xls')

        workbook = xlwt.Workbook(encoding='utf-8')
        booksheet = workbook.add_sheet('Sheet 1')
        for i, d in zip(range(len(self.revenue_cost)),self.revenue_cost):
            booksheet.write(j, i, d)
        workbook.save('.\\评价指标\\收益成本比.xls')

        workbook = xlwt.Workbook(encoding='utf-8')
        booksheet = workbook.add_sheet('Sheet 1')
        for i, d in zip(range(len(self.average_energy)),self.average_energy):
            booksheet.write(j, i, d)
        workbook.save('.\\评价指标\\平均能量消耗.xls')

        workbook = xlwt.Workbook(encoding='utf-8')
        booksheet = workbook.add_sheet('Sheet 1')
        booksheet.write(j, 0, sum(self.map_time)/len(self.map_time))
        workbook.save('.\\评价指标\\映射时间.xls')

        workbook = xlwt.Workbook(encoding='utf-8')
        booksheet = workbook.add_sheet('Sheet 1', cell_overwrite_ok=True)

        j = 0
        for i, d in  zip(range(len(self.node_open)), self.node_open):
            if i % 200 is 0:
                j = j + 1
            booksheet.write(j-1, i - (j - 1)*200, d)
        workbook.save('.\\评价指标\\节点开启量.xls')

        workbook = xlwt.Workbook(encoding='utf-8')
        booksheet = workbook.add_sheet('Sheet 1', cell_overwrite_ok=True)

        j = 0
        for i, d in  zip(range(len(self.edge_open)), self.edge_open):
            if i % 200 is 0:
                j = j + 1
            booksheet.write(j-1, i - (j - 1)*200, d)
        workbook.save('.\\评价指标\\链路开启量.xls')


    '''
    绘制评价指标
    '''
    def draw(self):
        self.save_excle()
        plt.figure(figsize=(30,30))
        plt.subplot(3, 2, 1)
        print(self.acceptrate)
        plt.plot(self.x,self.acceptrate,color="orange")
        plt.title("acceprate")
        plt.xlabel("time")
        plt.ylabel("acceprate")
        plt.legend(["acceprate"], loc="upper right")
        plt.grid(True)

        plt.subplot(3, 2, 2)
        print(self.revenue_cost)
        plt.plot(self.x,self.revenue_cost)
        axes = plt.gca()
        #axes.set_xlim([0, 1])
        axes.set_ylim([0.3, 1])
        plt.title("revenue_cost")
        plt.xlabel("time")
        plt.ylabel("revenue_cost")
        plt.legend(["revenue_cost"], loc="upper right")
        plt.grid(True)

        plt.subplot(3, 2, 3)
        print(self.node_open)
        plt.plot(self.node_open)
        plt.title("node_open")
        plt.xlabel("time")
        plt.ylabel("node_open")
        plt.legend(["node_open"], loc="upper right")
        plt.grid(True)
        
        plt.subplot(3, 2, 4)                                                    
        print(self.edge_open)
        plt.plot(self.edge_open,color="orange")
        plt.title("edegeeopen",)
        plt.xlabel("time")
        plt.ylabel("edegeeopen")
        plt.legend(["edegeeopen"], loc="upper right")
        plt.grid(True)

        plt.subplot(3, 2, 5)
        print(self.average_energy)
        plt.plot(self.x,self.average_energy, color="orange")
        plt.title("average_energy consumption", )
        plt.xlabel("time")
        plt.ylabel("average_energy consumption")
        plt.legend(["average_energy consumption"], loc="upper right")
        plt.grid(True)

        plt.subplot(3, 2, 6)
        print(sum(self.map_time)/len(self.map_time))
        plt.bar(10,self.map_time, width=2)
        plt.xlim(0, 20)
        plt.title("average map time", )
        plt.xlabel("time")
        plt.ylabel("average map time")
        plt.legend(["average map time"], loc="upper right")
        plt.grid(True)
        plt.show()




    '''
    计算代价底层网络代价
    temp1：映射前底层网络能耗
    temp2：映射成功后底层网络能耗
    cost：代价，后-前
    '''
    def cost_sn(self, temp1 = 0):
        temp2 = 0
        sn = len(self.sn.snode)
        for i in range(sn):
            temp2 = temp2 + self.sn.snode[i].lastcpu
        en = len(self.sn.sedge)
        for i in range(en):
            temp2 = temp2 + self.sn.sedge[i].lastbandwidth
        cost = abs(temp2 - temp1)
        return cost, temp2

    '''
    计算虚拟网收益
    虚拟请求节点CPU+链路带宽 = 收益
    '''
    def revenue_vnr(self, vnr):
        revenue = 0
        vn = len(vnr.vnode)
        for i in range(vn):
            revenue = revenue + vnr.vnode[i].cpu
        en = len(vnr.vedge)
        for i in range(en):
            revenue = revenue + vnr.vedge[i].bandwidth
        return revenue

    '''
    nodeopenf：计算节点开启个数
    '''
    def node_openf(self):
        opennode = 0
        sn = len(self.sn.snode)
        for i in range(sn):
            if self.sn.snode[i].open:
                opennode = opennode + 1
        return  opennode

    '''
    edgeopenf：计算链路开启个数
    '''
    def edge_openf(self):
        openedge = 0
        en  = len(self.sn.sedge)
        for i in range(en):
            if self.sn.sedge[i].open:
                openedge = openedge + 1
        return openedge



    '''
    模拟器函数
    algorithm_vne：虚拟网映射算法
    '''
    def simulation(self, algorithm_vne):
        t = self.timeframe
        n_vnr = len(self.VNRS)
        duration = []
        for i in range(n_vnr):
            duration.append(self.VNRS[i].duration)

        reqi = [z for z in range(n_vnr)]
        time = self.VNRS[0].time
        total = 0  # 总虚拟网请求
        node_suc = 0  # 节点映射成功次数
        map_success = 0  # 映射成功次数
        _, base = self.cost_sn()
        revenue = 0
        cost = 0
        self.t = self.VNRS[0].time
        self.energy = self.energy + self.sn.power(self.t)
        for i in range(n_vnr):
            vnr = self.VNRS[i]
            dtime = vnr.time - time  # 持续时间，本次-上次
            # 虚拟网持续时间内
            while dtime > 0:
                dtime = dtime - 1  # 每做一次循环将持续时间-1
                self.t = self.t + 1  # 当前时刻+1
                ii = 0  # 第i个虚拟网请求
                while reqi[ii] < i:  # 遍历第i个虚拟网之前的请求
                    if duration[reqi[ii]] <= 0:  # 第ii个虚拟网持续时间小于0
                        self.sn.remove_node_mapping(self.VNRS[reqi[ii]])  # 移除映射的虚拟节点
                        self.sn.remove_edge_mapping(self.VNRS[reqi[ii]])  # 移除映射的虚拟链路
                        reqi.remove(reqi[ii])  # 移除虚拟网请求
                        ii = ii - 1
                    else:
                        duration[reqi[ii]] = duration[reqi[ii]] - 1  # 把第i条虚拟网请求前的持续时间-1
                    ii = ii + 1
                self.energy = self.energy + self.sn.power(self.t)  # 总能量，计算平均能耗用

            self.node_open.append(self.node_openf())  # 统计节点开启个数
            self.edge_open.append(self.edge_openf())  # 统计链路开启个数
            starttime = ost.time()  # 记录开始时间
            success, v2sindex, ve2seindex = algorithm_vne(vnr)  # 映射算法；
            # success = True
            # v2sindex = [1,2]
            # ve2seindex = []
            if success:
                _, base = self.cost_sn(base)
                node_suc = node_suc + 1
                self.sn.nodemapping(vnr, v2sindex)  # 执行节点映射，传入参数vnr虚拟网以及v2sindex虚拟节点对应底层节点序列
                self.sn.edgemapping(vnr, ve2seindex)  # 执行链路映射，传入参数vnr虚拟网以及ve2seindex虚拟链路对应底层链路序列
                self.map_time.append(ost.time() - starttime)  # 记录映射时间
                self.energy = self.energy + self.sn.power(self.t)  # 总能耗
                tmpcost, base = self.cost_sn(base)
                cost = cost + tmpcost  # 总代价
                tempreven = self.revenue_vnr(vnr)
                revenue = revenue + tempreven  # 总收益
                map_success = map_success + 1  # 总映射成功个数
            else:
                print('map vnr node edge fail ', vnr.id)
            total = total + 1  # 总虚拟网个数
            time = vnr.time  #
            if time > t:
                self.acceptrate.append(map_success / total)  # 接收率
                self.revenue_cost.append(revenue / (cost + 0.01))  # 收益比
                self.node_success_rate.append(node_suc / total)  # 节点成功率
                self.average_energy.append(self.energy / vnr.time)  # 平均能耗
                self.x.append(t)
                t = t + self.timeframe

    '''
    自己写的虚拟网映射
    随机数映射算法
    '''
    def lwtvne(self, vnr):
        n_succ, v2sindex, asnode, asedge = self.lwt_nodemap(vnr)
        self.set_mappable_flags(self.lnum)
        if n_succ:
            e_succ, ve2seindex, asnode, asedge = self.lwt_linkem(vnr, asnode, asedge, v2sindex)
            if e_succ:
                return True, v2sindex, ve2seindex
            else:
                return False, [], []
        else:
            return False, [], []
    '''
    '''
    def lwt_nodemap(self, vnr):
        asnode = cp.deepcopy(self.sn.snode)
        asedege = cp.deepcopy(self.sn.sedge)
        # vnr_len = len(vnr)
        # sn_len = len(vnr)
        g = nx.Graph()
        bandlimit = 0
        v2sindex = []
        '''
        存在的问题：如果所有所有物理节点都不能满足该虚拟网请求节点，那么它就会continue出去到下一个节点而本节点并没有映射
        写函数：同一个虚拟网请求的节点不能映射到同一个节点上
        '''
        for vnode in vnr.vnode:
            flag = False
            for i in range(100):
                if vnode.cpu < asnode[i].lastcpu:
                    v2sindex.append([vnode.index, i])
                    flag = True
                    break
                else:
                    continue
            # while vnode.cpu < asnode[num].lastcpu:
            #     # 生成随机数
            #     num = random.randint(0, 99)
            #     # 如果虚拟节点num可以承载虚拟节点就将其映射
            #
            #     print(vnode.index, num)
            #     break
            # continue
            if flag is False:
                return  False, [], [], []
        return True, v2sindex, asnode, asedege





    def lwt_linkem(self, myvnr, asnode, asedge, v2sindex):

        return al.edgemapping3(asnode, asedge, myvnr, v2sindex)




    def FeedbackControl_based_EEVNE(self, vnr):
        STEP = 1
        while True:
            self.set_mappable_flags(self.lnum)
            em, v2sindex, asnode, asedge = self.NodeEm(vnr)
            if em:
                while True:
                    em, ve2seindex, asnode, asedge = self.LinkEm(vnr, asnode, asedge, v2sindex)
                    if em:
                        return True, v2sindex, ve2seindex
                    if self.lnum > len(self.sn.sedge):
                        return False, [], []
                    break
                continue
            self.lnum = self.lnum + STEP
            if self.lnum > len(self.sn.sedge):
                return False, [], []

    def NodeEm(self,vnr):
        asnode = cp.deepcopy(self.sn.snode)
        asedege = cp.deepcopy(self.sn.sedge)
        g = nx.Graph()
        bandlimit = 0
        v2sindex = []
        a = 0.5
        for sedge in asedege:
            if sedge.bandwidth > bandlimit:
                bandwidth = sedge.bandwidth - bandlimit
                g.add_weighted_edges_from([(sedge.link[0], sedge.link[1], bandwidth)])
        for vnode in vnr.vnode:
            Ncpu = [(snode.index, snode.lastcpu - vnode.cpu) for snode in asnode if snode.lastcpu > vnode.cpu and snode.mappable_flag ]
            if len(Ncpu) is not 0:
                Nbw = [(snodemsg[0], sum([g.get_edge_data(snodemsg[0], v)['weight'] for v in g.neighbors(snodemsg[0])]))
                       for snodemsg in Ncpu]
                NR = [(nc[0], a * nc[1] + (1 - a) * nw[1]) for nc, nw in zip(Ncpu, Nbw)]
                j = NR[np.argmax([i[1] for i in NR])][0]
                asnode[j].lastcpu = asnode[j].lastcpu - vnode.cpu
                asnode[j].vnodeindexs.append([vnr.id, vnode.index])
                v2sindex.append([vnode.index, j])
            else:
                return False, [], [], []
        return True, v2sindex, asnode, asedege


    def LinkEm(self, myvnr, asnode, asedge, v2sindex):
        return al.edgemapping3(asnode, asedge, myvnr, v2sindex)


    def set_mappable_flags(self, nosleep):
        g = nx.Graph()
        bandlimit = 0
        en = len(self.sn.sedge)
        sln = 0
        linksum = en
        sleep = linksum - nosleep
        for i in range(en):
            if self.sn.sedge[i].bandwidth > bandlimit:
                bandwidth = self.sn.sedge[i].bandwidth - bandlimit
                g.add_weighted_edges_from([(self.sn.sedge[i].link[0], self.sn.sedge[i].link[1], bandwidth)])
        snode_degree = [degree for degree in g.degree([i for i in range(len(self.sn.snode))]).values()]
        for i in range(en):
            self.sn.sedge[i].mappable_flag = True
        for i in range(len(self.sn.snode)):
            self.sn.snode[i].mappable_flag = True
        while sln < sleep:
            u = np.argmin(snode_degree)
            if self.sn.snode[u].mappable_flag:
                for v in g.neighbors(u):
                    edgeid = self.sn.getedge(u, v)
                    self.sn.sedge[edgeid].mappable = False
                    sln = sln + 1
                    snode_degree[u] = snode_degree[u] - 1
                    snode_degree[v] = snode_degree[v] - 1
                    if snode_degree[u] is 0:
                        self.sn.snode[u].mappable_flag = False
                    if snode_degree[v] is 0:
                        self.sn.snode[v].mappable_flag = False
                    if sln >= sleep:
                        break
            else:
                break


#Multi-objective enhanced particle swarm optimization in virtual network embedding 2016
    def MO_NPAO_part(self,vnr,pos):
        v2sindex=[]
        asnode=cp.deepcopy(self.sn.snode)
        asedege=cp.deepcopy(self.sn.sedge)
        cost=0
        energy=0
        for vnode,si in zip(vnr.vnode,pos):
            cost=cost+vnode.cpu
            if (vnode.cpu<asnode[si].lastcpu) and ([vnr.id, vnode.index] not in asnode[si].vnodeindexs):
                asnode[si].lastcpu=asnode[si].lastcpu-vnode.cpu
                asnode[si].open=True
                asnode[si].vnodeindexs.append([vnr.id, vnode.index])
                v2sindex.append([vnode.index,si])
            else:
                return float('inf'),float('inf'),[]
        success, ve2seindex, asnode, asedege = self.sn.testedegemapping2(asnode, asedege, vnr, v2sindex)
        for e,setp in zip(vnr.vedge,ve2seindex):
            cost=cost+e.bandwidth*len(setp)
        if success:
            for snode in asnode:
                if snode.open:
                    energy=energy+snode.baseEnergy
            for vnode in vnr.vnode:
                energy=energy+vnode.cpu*asnode[1].maxEnergy
            return cost,energy,ve2seindex
        else:
            return float('inf'),float('inf'),[],


    def MO_NPSO(self,vnr):
        p1=0.1
        p2=0.2
        p3=0.7
        npart = 5
        iteration=30
        X=np.zeros((npart,len(vnr.vnode)),dtype=int)
        Xvalue=[]
        Xve2seindex=[]
        V=np.zeros((npart,len(vnr.vnode)),dtype=int)
        pbest=[]
        pbestx=np.zeros((npart,len(vnr.vnode)))
        pbestve2seindex=[]
        gx=[]
        g=float('inf')
        gve2seindex=[]
        sn=len(self.sn.snode)
        ce=[]
        en=[]
        for i in range(npart):
            position=np.random.choice(range(sn),p=[1/sn for i in range(sn)],size=len(vnr.vnode),replace=False )
            X[i]=position
            pbestx[i]=position
            c,e,ve2seindex=self.MO_NPAO_part(vnr,position)
            ce.append(c)
            en.append(e)
            Xve2seindex.append(ve2seindex)
            pbestve2seindex.append(ve2seindex)
        tmpce=[cee for cee in ce if cee !=float("inf")]
        if len(tmpce) != 0:
            maxce=np.max(tmpce)
            mince=np.min(tmpce)
            if maxce == mince:
                for ii in range(len(ce)):
                    if ce[i] != float('inf'):
                        ce[ii]=0
            else:
                ce = [(cee - mince) / (maxce - mince) for cee in ce]
        tmpen=[enn for enn in en if enn !=float("inf")]
        if len(tmpen) !=0:
            maxen=np.max(tmpen)
            minen=np.min(tmpen)
            if maxen ==minen:
                for ii in range(len(en)):
                    if en[i] != float('inf'):
                        en[ii]=0
            else:
                en=[(enn-minen)/(maxen-minen) for enn in en]
        y=0.5
        for i in range(len(ce)):
            pbest.append(y*ce[i]+(1-y)*en[i])
            Xvalue.append(y*ce[i]+(1-y)*en[i])
            if g>=y*ce[i]+(1-y)*en[i]:
                g=y*ce[i]+(1-y)*en[i]
                gx=X[i]
                gve2seindex=Xve2seindex[i]
        sub=lambda x,y:[int(ii is jj) for ii,jj in zip(x,y)]
        add=lambda p1,x1,p2,x2,p3,x3:[np.random.choice([i,j,k],p=[p1,p2,p3])for i,j,k in zip(x1,x2,x3)]
        for _ in range(iteration):
            ce = []
            en = []
            for i in range(npart):
                V[i]=add(p1, V[i], p2, sub(pbestx[i], X[i]), p3, sub(gx, X[i]))
                for ii in range(len(X[i])):
                    if V[i][ii] is 0:
                        X[i][ii]=np.random.choice(range(len(self.sn.snode)))
                c,e, ve2seindex = self.MO_NPAO_part(vnr,  X[i])
                ce.append(c)
                en.append(e)
                Xve2seindex[i]=ve2seindex
            tmpce = [cee for cee in ce if cee != float("inf")]
            if len(tmpce) != 0:
                maxce = np.max(tmpce)
                mince = np.min(tmpce)
                if maxce == mince:
                    for ii in range(len(ce)):
                        if ce[i] != float('inf'):
                            ce[ii] = 0
                else:
                    ce = [(cee - mince) / (maxce - mince) for cee in ce]
            tmpen = [enn for enn in en if enn != float("inf")]
            if len(tmpen) != 0:
                maxen = np.max(tmpen)
                minen = np.min(tmpen)
                if maxen == minen:
                    for ii in range(len(en)):
                        if en[i] != float('inf'):
                            en[ii] = 0
                else:
                    en = [(enn - minen) / (maxen - minen) for enn in en]
            for i in range(npart):
                Xvalue[i]=y*ce[i]+(1-y)*en[i]
                if pbest[i]>Xvalue[i]:
                    pbest[i]=Xvalue[i]
                    pbestx=X[i]
                    pbestve2seindex[i]=Xve2seindex[i]
                if g>pbest[i]:
                    g=pbest[i]
                    gx=pbestx[i]
                    gve2seindex=pbestve2seindex[i]
        gx=[[i,gx[i]]for i in range(len(vnr.vnode))]
        if g == float('inf'):
            return False,[],[]
        print('---------')
        print(gx)
        print(gve2seindex)
        return True,gx,gve2seindex

    
    def MO_NPSO_simulation(self):
        self.simulation_2step(self.MO_NPSO,"MO_NPSO")








