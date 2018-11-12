from src import node_and_edge as ne
from src import subnet as sn
from src import virnet as vn
import networkx as nx
import matplotlib.pyplot as plt
from src import simulation as sim
from src import algorithm as al
import time

# 底层网络数据路径
sn_path = r"E:\虚拟网数据\sn\sub100-570.txt"

# 虚拟网络数据路径
vnr_path = r"E:\虚拟网数据\newr-2000-0-20-0-25"

# 加载底层网络数据
snet = sn.SubstrateNetwork(sn_path)

# 虚拟网数量
vnrnumber = 2000

mysimulation = sim.Simulation()
mysimulation.init(sn_path, vnr_path, vnrnumber, 1000)
time_start = time.time()
mysimulation.simulation(mysimulation.MO_NPSO)
time_end = time.time()
print("time", time_end-time_start)
mysimulation.draw()


# for i in mysimulation.VNRS:
#     for j in range(len(i.vnode)):
#         print(i.id, i.time, i.vnode[j].index, i.vnode[j].cpu, i.vnode[j].position)


# for i in mysimulation.sn.snode:
#     print(i.index, i.lastcpu, i.vnodeindexs, i.open, i.mappable_flag)
#
# for j in mysimulation.sn.sedge:
#     print(j.index, j.bandwidth, j.vedgeindexs, j.link, j.open, j.mappable_flag)


# SG = sn.sn2networkG()
# pos= {snode.index: snode.position for snode in sn.snode} # 获取节点位置
# nx.draw(SG, pos)
# plt.show()

# 执行映射过程



# VG = vnr.vn2networkG()
# vnr.draw_graph(VG)


# for vd in vnr.vedge:
#     print(vd)
#
# for vn in vnr.vnode:
#     print(vn)
# print(vnr.duration)
# print(vnr.time)
# print(vnr.id)





# n = ne.Vnode(1,20,(100,100))
# n.msg()

