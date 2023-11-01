import copy
import math
import random
import time
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from shapely.geometry import LineString
from shapely.geometry import Polygon as Pol
import numpy as np
from shapely.geometry import Point

# Drawing options
show_animation = True


class RRT:

    def __init__(self, obstacleList, randArea,
                 expandDis=3.0, goalSampleRate=0, maxIter=500):

        self.start = None
        self.goal = None
        self.min_rand = randArea[0]
        self.max_rand = randArea[1]
        self.expand_dis = expandDis
        self.goal_sample_rate = goalSampleRate
        self.max_iter = maxIter
        self.obstacle_list = obstacleList
        self.node_list = None
        self.Ka = 0.01
        self.Kr = 50

    def rrt_star_planning(self, start, goal, animation=True):
        start_time = time.time()
        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.node_list = [self.start]
        path = None
        lastPathLength = float('inf')
        path_sign = False
        plt.figure(figsize=(6, 6))
        a = 0
        iter = 0

        self.start, potential_angle = self.Computational_potential_field(self.start)
        for i in range(self.max_iter):
            iter = iter + 1

            # random sampling
            rnd = self.sample(path_sign, lastPathLength)

            # find the nearest node
            n_ind = self.get_nearest_list_index(self.node_list, rnd, path_sign, lastPathLength)
            nearestNode = self.node_list[n_ind]
            theta = math.atan2(rnd[1] - nearestNode.y, rnd[0] - nearestNode.x)

            angle_diff = 0
            b = 0

            while angle_diff < math.pi/6:
                b = b+1

                # steer
                newNode, angle = self.get_new_node(theta, n_ind, nearestNode, path_sign, b)
                # print(newNode.x, newNode.y)
                angle_diff = abs(angle - potential_angle)
                if angle_diff > math.pi:
                    angle_diff = abs(angle_diff - 2 * math.pi)

                # CollisionFree
                noCollision = self.check_segment_collision(newNode.x, newNode.y, nearestNode.x, nearestNode.y)
                if noCollision:

                    # Calculate the artificial potential field of new node
                    newNode, potential_angle = self.Computational_potential_field(newNode)

                    # Find nearby nodes
                    nearInds = self.find_near_nodes(newNode)

                    # Reselect the parent node
                    newNode = self.choose_parent(newNode, nearInds)

                    self.node_list.append(newNode)

                    # Nearby nodes reselect parent nodes
                    self.rewire(newNode, nearInds)

                    if animation:
                        self.draw_graph(newNode, path, rnd)

                    if path_sign:
                        if newNode.fvalue > lastPathLength:
                            # Rejecting nodes that cost too much
                            newNode.function = False

                    # If the node and the target point are too close together, they can be directly connected
                    if self.is_near_goal(newNode):
                        if self.check_segment_collision(newNode.x, newNode.y, self.goal.x, self.goal.y):
                            lastIndex = len(self.node_list) - 1
                            tempPath = self.get_final_course(lastIndex)
                            tempPathLen = self.get_path_len(tempPath)
                            path_sign = True
                            if lastPathLength > tempPathLen:
                                path = tempPath
                                lastPathLength = tempPathLen
                                costtime = time.time() - start_time
                                print("current path length: {}, It costs {} s".format(tempPathLen, costtime))
                else:
                    break

        print(len(self.node_list))
        print(time.time() - start_time)
        return path


    def sample(self, path_sign, lastPathLength):
        if random.randint(0, 100) >= self.goal_sample_rate:
            while True:
                if path_sign is not True:
                    rnd = [random.uniform(self.min_rand, self.max_rand), random.uniform(self.min_rand, self.max_rand)]
                    break
                else:
                    rnd = [random.uniform(self.min_rand, self.max_rand), random.uniform(self.min_rand, self.max_rand)]
                    # Determine whether the F-value of the sampling point is less than the current shortest path cost
                    if self.check_congfuquyu(rnd, lastPathLength, path_sign):
                        break

        else:  # goal point sampling
            rnd = [self.goal.x, self.goal.y]
        return rnd

    def check_congfuquyu(self, rnd, lastPathLength, path_sign):
        n_qnearest = self.get_nearest_list_index(self.node_list, rnd, path_sign, lastPathLength)
        qnearest = self.node_list[n_qnearest]
        dis = (qnearest.x - rnd[0]) ** 2 + (qnearest.y - rnd[1]) ** 2
        rndfvalue = math.sqrt(dis) + qnearest.cost + math.sqrt((rnd[0] - self.goal.x) ** 2 + (rnd[1] - self.goal.y) ** 2)
        nearnodes = self.find_near_nodes(qnearest)
        step = []
        for i in nearnodes:
            step.append((self.node_list[i].x - qnearest.x) ** 2 + (self.node_list[i].y - qnearest.y) ** 2)
        if dis > min(step) and rndfvalue < lastPathLength:
            return True
        else:
            return False

    def choose_parent(self, newNode, nearInds):
        if len(nearInds) == 0:
            return newNode

        dList = []
        for i in nearInds:
            dx = newNode.x - self.node_list[i].x
            dy = newNode.y - self.node_list[i].y
            d = math.hypot(dx, dy)
            theta = math.atan2(dy, dx)
            if self.check_collision(self.node_list[i], theta, d):
                dList.append(self.node_list[i].cost + d)
            else:
                dList.append(float('inf'))

        minCost = min(dList)
        minInd = nearInds[dList.index(minCost)]

        if minCost == float('inf'):
            print("min cost is inf")
            return newNode

        # Attempts to connect to the parent node on the path
        if minInd != 0:
            minnodeparent = self.node_list[self.node_list[minInd].parent]
            while minInd != 0:
                dx = newNode.x - minnodeparent.x
                dy = newNode.y - minnodeparent.y
                d = math.hypot(dx, dy)
                theta = math.atan2(dy, dx)

                if self.check_collision(minnodeparent, theta, d):
                    minInd = self.node_list[minInd].parent
                    if minInd != 0:
                        minnodeparent = self.node_list[self.node_list[minInd].parent]
                else:
                    break
        dx = newNode.x - self.node_list[minInd].x
        dy = newNode.y - self.node_list[minInd].y
        d = math.hypot(dx, dy)
        minCost = d + self.node_list[minInd].cost

        newNode.cost = minCost
        newNode.parent = minInd
        return newNode

    def find_near_nodes(self, newNode):
        n_node = len(self.node_list)
        r = 50.0 * math.sqrt((math.log(n_node) / n_node))
        if r < 3:
            r = 3
        d_list = [(node.x - newNode.x) ** 2 + (node.y - newNode.y) ** 2
                  for node in self.node_list]
        near_inds = [d_list.index(i) for i in d_list if i <= r ** 2]
        return near_inds

    @staticmethod
    def get_path_len(path):
        pathLen = 0
        for i in range(1, len(path)):
            node1_x = path[i][0]
            node1_y = path[i][1]
            node2_x = path[i - 1][0]
            node2_y = path[i - 1][1]
            pathLen += math.sqrt((node1_x - node2_x) ** 2 + (node1_y - node2_y) ** 2)
        return pathLen

    @staticmethod
    def line_cost(node1, node2):
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2)

    def get_nearest_list_index(self, nodes, rnd, path_sign, lastPathLength):
        dList = []
        for node in nodes:
            dList.append((node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2)
            node.fvalue = node.cost + math.sqrt((node.x - self.goal.x) ** 2 + (node.y - self.goal.y) ** 2)
            if path_sign:
                if node.fvalue > lastPathLength:
                    node.function = False
                else:
                    node.function = True

        # Reject nodes with excessive cost based on F-value
        for i in range(len(nodes)):
            minIndex = dList.index(min(dList))
            if nodes[minIndex].function:
                break
            else:
                dList[minIndex] = float('inf')

        return minIndex

    def get_new_node(self, theta, n_ind, nearestNode, path_sign, b):
        newNode = copy.deepcopy(nearestNode)

        # Calculate the angle of the combined force
        expand_rand = math.sqrt(newNode.potential_x**2 + newNode.potential_y**2)
        expand_randx = expand_rand * math.cos(theta)
        expand_randy = expand_rand * math.sin(theta)
        expand_allx = expand_randx * 0.6 + newNode.potential_x * 0.4
        expand_ally = expand_randy * 0.6 + newNode.potential_y * 0.4
        angle = math.atan2(expand_ally, expand_allx)

        # extend
        if path_sign is not True:
            newNode.x += self.expand_dis*b * math.cos(angle)
            newNode.y += self.expand_dis*b * math.sin(angle)
            newNode.cost += self.expand_dis*b

        else:
            newNode.x += self.expand_dis*b/2 * math.cos(angle)
            newNode.y += self.expand_dis*b/2 * math.sin(angle)
            newNode.cost += self.expand_dis*b/2

        newNode.fvalue = newNode.cost + math.sqrt((newNode.x - self.goal.x) ** 2 + (newNode.y - self.goal.y) ** 2)
        newNode.parent = n_ind
        return newNode, angle

    def is_near_goal(self, node):
        d = self.line_cost(node, self.goal)
        if d < self.expand_dis * 2.5:
            return True
        return False

    def rewire(self, newNode, nearInds):
        n_node = len(self.node_list)
        for i in nearInds:
            nearNode = self.node_list[i]

            d = math.sqrt((nearNode.x - newNode.x) ** 2 + (nearNode.y - newNode.y) ** 2)
            s_cost = newNode.cost + d

            # Nearby nodes reselect parent nodes
            if nearNode.cost > s_cost:
                theta = math.atan2(newNode.y - nearNode.y,
                                   newNode.x - nearNode.x)
                if self.check_collision(nearNode, theta, d):
                    nearNode.parent = n_node - 1
                    nearNode.cost = s_cost
                    if nearNode.parent != 0:
                        minnodeparent = self.node_list[self.node_list[nearNode.parent].parent]
                        while nearNode.parent != 0:
                            dx = nearNode.x - minnodeparent.x
                            dy = nearNode.y - minnodeparent.y
                            d = math.hypot(dx, dy)
                            theta = math.atan2(dy, dx)
                            if self.check_collision(minnodeparent, theta, d):
                                nearNode.parent = self.node_list[nearNode.parent].parent
                                if nearNode.parent != 0:
                                    minnodeparent = self.node_list[self.node_list[nearNode.parent].parent]
                            else:
                                break

    def Computational_potential_field(self, newNode):

        # Calculate attractive force
        dis_Goal_newnode = math.sqrt((self.goal.x - newNode.x) ** 2+ (self.goal.y - newNode.y) ** 2)
        Uattx = 0.5*self.Ka*(dis_Goal_newnode)*(self.goal.x - newNode.x)
        Uatty = 0.5*self.Ka*(dis_Goal_newnode)*(self.goal.y - newNode.y)

        # Calculate repulsive force
        Urepx = 0
        Urepy = 0
        dobs = 3
        for obstacle in self.obstacle_list:
            poly = Pol(obstacle)
            point = Point(newNode.x, newNode.y)
            dis_ob_newnode = point.distance(poly)
            center = Pol(obstacle).centroid

            if dis_ob_newnode < dobs:
                Urepx = Urepx + 0.5 * self.Kr*(1/dis_ob_newnode - 1/dobs)**2/dis_ob_newnode * \
                        (newNode.x - center.x)
                Urepy = Urepy + 0.5 * self.Kr * (1 / dis_ob_newnode - 1 / dobs) ** 2 / dis_ob_newnode * \
                        (newNode.y - center.y)

        newNode.potential_x = Uattx + Urepx
        newNode.potential_y = Uatty + Urepy
        potential_angle = math.atan2(newNode.potential_y, newNode.potential_x)

        return newNode, potential_angle

    def check_segment_collision(self, x1, y1, x2, y2):
        sign = True
        for obstacle in self.obstacle_list:
            ob = Pol(obstacle)
            line = LineString([(x1, y1), (x2, y2)])
            if ob.crosses(line):
                sign = False
                break
        return sign

    def check_collision(self, nearNode, theta, d):
        tmpNode = copy.deepcopy(nearNode)
        end_x = tmpNode.x + math.cos(theta) * d
        end_y = tmpNode.y + math.sin(theta) * d
        return self.check_segment_collision(tmpNode.x, tmpNode.y, end_x, end_y)

    def get_final_course(self, lastIndex):
        path = [[self.goal.x, self.goal.y]]
        while self.node_list[lastIndex].parent is not None:
            node = self.node_list[lastIndex]
            path.append([node.x, node.y])
            lastIndex = node.parent
        path.append([self.start.x, self.start.y])
        return path

    def draw_graph(self, rnd=None, path=None, rnd1=None):
        plt.clf()
        # for stopping simulation with the esc key.
        fig = plt.gca()
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if rnd is not None:
            plt.plot(rnd.x, rnd.y, "^k")

        if rnd1 is not None:
            plt.plot(rnd1[0], rnd1[1], "^r")

        for n in self.node_list:
            plt.plot(n.x, n.y, ".k")

        for node in self.node_list:
            if node.parent is not None:
                if node.x or node.y is not None:
                    plt.plot([node.x, self.node_list[node.parent].x], [
                        node.y, self.node_list[node.parent].y], "-g")

        for k in range(len(self.obstacle_list)):
            fig.add_patch(Polygon(self.obstacle_list[k], color='k'))

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.goal.x, self.goal.y, "xr")

        if path is not None:
            plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')

        plt.axis([0, 100, 0, 100])
        # plt.grid(True)
        plt.pause(0.01)


class Node:

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None
        self.function = True
        self.fvalue = float('inf')
        self.potential_x = None
        self.potential_y = None


def main():
    print("Start rrt planning")

    # map1
    # obstacleList = [np.array([[20, 42], [20, 58], [32, 58], [32, 42]]),
    #                 np.array([[36, 23], [36, 33], [64, 33], [64, 23]]),
    #                 np.array([[68, 42], [68, 58], [80, 58], [80, 42]]),
    #                 np.array([[36, 67], [36, 77], [64, 77], [64, 67]]),
    #                 ]

    # map2
    obstacleList = [np.array([[8, 27], [14, 27], [14, 33], [8, 33]]),
                    np.array([[21, 14], [29, 14], [29, 22], [21, 22]]),
                    np.array([[34, 48], [45, 35], [34, 34]]),
                    np.array([[29, 39], [35, 39], [35, 45], [29, 45]]),
                    np.array([[27, 1], [33, 1], [33, 7], [27, 7]]),
                    np.array([[53, 45], [59, 45], [59, 51], [53, 51]]),
                    np.array([[15, 35], [21, 35], [21, 41], [15, 41]]),
                    np.array([[63, 57], [69, 57], [69, 63], [63, 63]]),
                    np.array([[81, 57], [87, 57], [87, 63], [81, 63]]),
                    np.array([[21, 86], [27, 86], [27, 92], [21, 92]]),
                    np.array([[5, 56], [11, 56], [11, 62], [5, 62]]),
                    np.array([[65, 75], [71, 75], [71, 81], [65, 81]]),
                    np.array([[23, 69], [29, 69], [29, 75], [23, 75]]),
                    np.array([[83, 55], [89, 55], [89, 61], [83, 61]]),
                    np.array([[48, 14], [54, 14], [54, 20], [48, 20]]),
                    np.array([[39, 53], [45, 53], [45, 59], [39, 59]]),
                    np.array([[91, 91], [97, 91], [97, 97], [91, 97]]),
                    np.array([[94, 33], [100, 33], [100, 39], [94, 39]]),
                    np.array([[49, 27], [57, 27], [57, 35], [49, 35]]),
                    np.array([[38, 88], [46, 88], [46, 96], [38, 96]]),
                    np.array([[79, 19], [88, 19], [88, 28], [79, 28]]),
                    np.array([[54, 73], [62, 73], [62, 81], [54, 81]]),
                    np.array([[86, 18], [94, 18], [94, 26], [86, 26]]),
                    np.array([[37, 51], [41, 58], [34, 55]]),
                    np.array([[23, 49], [27, 56], [19, 54]]),
                    np.array([[51, 59], [55, 67], [48, 64]]),
                    np.array([[65, 43], [72, 40], [67, 49]]),
                    ]

    # map3
    # obstacleList = [np.array([[35, 20], [35, 35], [40, 35], [40, 20]]),
    #                 np.array([[35, 35], [35, 40], [0, 40], [0, 35]]),
    #                 np.array([[60, 0], [60, 35], [65, 35], [65, 0]]),
    #                 np.array([[65, 35], [65, 40], [80, 40], [80, 35]]),
    #                 np.array([[100, 60], [65, 60], [65, 65], [100, 65]]),
    #                 np.array([[65, 65], [60, 65], [60, 80], [65, 80]]),
    #                 np.array([[40, 100], [40, 65], [35, 65], [35, 100]]),
    #                 np.array([[35, 65], [35, 60], [20, 60], [20, 65]]),
    #                 np.array([[46, 46], [46, 54], [54, 54], [54, 46]]),
    #                 ]

    # map4
    # obstacleList = [np.array([[18, 18], [18, 38], [21, 38], [21, 18]]),
    #                 np.array([[32, 7], [32, 10], [61, 10], [61, 7]]),
    #                 np.array([[0, 57], [0, 60], [18, 60], [18, 57]]),
    #                 np.array([[18, 52], [18, 72], [21, 72], [21, 52]]),
    #                 np.array([[32, 23], [32, 26], [51, 26], [51, 23]]),
    #                 np.array([[44, 26], [44, 57], [47, 57], [47, 26]]),
    #                 np.array([[32, 61], [32, 80], [35, 80], [35, 61]]),
    #                 np.array([[32, 80], [32, 83], [60, 83], [60, 80]]),
    #                 np.array([[74, 76], [74, 100], [77, 100], [77, 76]]),
    #                 np.array([[71, 73], [71, 76], [82, 76], [82, 73]]),
    #                 np.array([[82, 20], [82, 60], [85, 60], [85, 20]]),
    #                 np.array([[70, 44], [70, 47], [82, 47], [82, 44]]),
    #                 np.array([[61, 0], [61, 22], [64, 22], [64, 0]]),
    #                 np.array([[18, 85], [18, 100], [21, 100], [21, 85]]),
    #                 np.array([[82, 7], [82, 10], [100, 10], [100, 7]]),
    #                 np.array([[27, 42], [27, 48], [33, 48], [33, 42]]),
    #                 np.array([[58, 52], [58, 60], [66, 60], [66, 52]]),
    #                 ]

    # Set params
    rrt = RRT(randArea=[0, 100], obstacleList=obstacleList, maxIter=500)

    # path = rrt.rrt_star_planning(start=[40, 10], goal=[60, 90], animation=show_animation)  # map1
    path = rrt.rrt_star_planning(start=[15, 18], goal=[73, 67], animation=show_animation)  # map2
    # path = rrt.rrt_star_planning(start=[10, 10], goal=[90, 90], animation=show_animation)  # map3
    # path = rrt.rrt_star_planning(start=[10, 10], goal=[90, 90], animation=show_animation)  # map4

    print("Done!!")

    if show_animation and path:
        plt.show()


if __name__ == '__main__':
    main()
