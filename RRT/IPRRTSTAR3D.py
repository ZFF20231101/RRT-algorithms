import copy
import math
import random
import time
import matplotlib.pyplot as plt
import numpy as np

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
        self.start = Node(start[0], start[1], start[2])
        self.goal = Node(goal[0], goal[1], goal[2])
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
            theta = [rnd[0] - nearestNode.x, rnd[1] - nearestNode.y, rnd[2] - nearestNode.z]

            angle_diff = 0
            b = 0

            while angle_diff < math.pi/6:
                b = b+1
                # steer
                newNode, angle = self.get_new_node(theta, n_ind, nearestNode, path_sign, b)
                # print(newNode.x, newNode.y)
                angle_diff = np.arccos(np.dot(angle, potential_angle) / (np.linalg.norm(angle) * np.linalg.norm(potential_angle)))
                if angle_diff > math.pi:
                    angle_diff = abs(angle_diff - 2 * math.pi)

                # CollisionFree
                noCollision = self.check_segment_collision(newNode.x, newNode.y, newNode.z, nearestNode.x, nearestNode.y, nearestNode.z)
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
                        if self.check_segment_collision(newNode.x, newNode.y, newNode.z, self.goal.x, self.goal.y, self.goal.z):
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
                    rnd = [random.uniform(self.min_rand, self.max_rand), random.uniform(self.min_rand, self.max_rand),
                           random.uniform(self.min_rand, self.max_rand)/2.5]  # map5
                    # rnd = [random.uniform(self.min_rand, self.max_rand), random.uniform(self.min_rand, self.max_rand),
                    #        random.uniform(self.min_rand, self.max_rand)/5]  # map6
                    break
                else:
                    rnd = [random.uniform(self.min_rand, self.max_rand), random.uniform(self.min_rand, self.max_rand),
                           random.uniform(self.min_rand, self.max_rand)/2.5]  # map5
                    # rnd = [random.uniform(self.min_rand, self.max_rand), random.uniform(self.min_rand, self.max_rand),
                    #        random.uniform(self.min_rand, self.max_rand)/5]  # map6

                    # Determine whether the F-value of the sampling point is less than the current shortest path cost
                    if self.check_congfuquyu(rnd, lastPathLength, path_sign):
                        break

        else:  # goal point sampling
            rnd = [self.goal.x, self.goal.y, self.goal.z]
        return rnd

    def check_congfuquyu(self, rnd, lastPathLength, path_sign):
        n_qnearest = self.get_nearest_list_index(self.node_list, rnd, path_sign, lastPathLength)
        qnearest = self.node_list[n_qnearest]
        dis = (qnearest.x - rnd[0]) ** 2 + (qnearest.y - rnd[1]) ** 2 + (qnearest.z - rnd[2]) ** 2
        rndfvalue = math.sqrt(dis) + qnearest.cost + math.sqrt((rnd[0] - self.goal.x) ** 2 + (rnd[1] - self.goal.y) ** 2 + (rnd[2] - self.goal.z) ** 2)
        nearnodes = self.find_near_nodes(qnearest)
        step = []
        for i in nearnodes:
            step.append((self.node_list[i].x - qnearest.x) ** 2 + (self.node_list[i].y - qnearest.y) ** 2 + (self.node_list[i].z - qnearest.z) ** 2)
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
            dz = newNode.z - self.node_list[i].z
            d = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            theta = [dx, dy, dz]
            if d == 0:
                continue
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
                dz = newNode.z - minnodeparent.z
                d = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                theta = [dx, dy, dz]
                # 判断当前节点是否和最短距离节点的父节点相连，
                if self.check_collision(minnodeparent, theta, d):
                    minInd = self.node_list[minInd].parent
                    if minInd != 0:
                        minnodeparent = self.node_list[self.node_list[minInd].parent]
                else:
                    break
        dx = newNode.x - self.node_list[minInd].x
        dy = newNode.y - self.node_list[minInd].y
        dz = newNode.z - self.node_list[minInd].z
        d = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
        minCost = d + self.node_list[minInd].cost

        newNode.cost = minCost
        newNode.parent = minInd
        return newNode

    def find_near_nodes(self, newNode):
        n_node = len(self.node_list)
        r = 50.0 * math.sqrt((math.log(n_node) / n_node))
        if r < 4:
            r = 4
        d_list =[]
        for node in self.node_list:
            distance = (node.x - newNode.x) ** 2 + (node.y - newNode.y) ** 2 + (node.z - newNode.z) ** 2
            if distance == 0.0:
                distance = float('inf')
            d_list.append(distance)

        near_inds = [d_list.index(i) for i in d_list if i <= r ** 2]
        return near_inds

    @staticmethod
    def get_path_len(path):
        pathLen = 0
        for i in range(1, len(path)):
            node1_x = path[i][0]
            node1_y = path[i][1]
            node1_z = path[i][2]
            node2_x = path[i - 1][0]
            node2_y = path[i - 1][1]
            node2_z = path[i - 1][2]
            pathLen += math.sqrt((node1_x - node2_x) ** 2 + (node1_y - node2_y) ** 2 + (node1_z - node2_z) ** 2)
        return pathLen

    @staticmethod
    def line_cost(node1, node2):
        return math.sqrt((node1.x - node2.x) ** 2 + (node1.y - node2.y) ** 2 + (node1.z - node2.z) ** 2)

    def get_nearest_list_index(self, nodes, rnd, path_sign, lastPathLength):
        dList = []
        for node in nodes:
            dList.append((node.x - rnd[0]) ** 2 + (node.y - rnd[1]) ** 2 + (node.z - rnd[2]) ** 2)
            node.fvalue = node.cost + self.line_cost(node, self.goal)
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
        expand_rand = math.sqrt(newNode.potential_x**2 + newNode.potential_y**2 + newNode.potential_z**2)
        expand_randx = expand_rand * theta[0]/math.sqrt(theta[0]**2 + theta[1]**2 + theta[2]**2)
        expand_randy = expand_rand * theta[1]/math.sqrt(theta[0]**2 + theta[1]**2 + theta[2]**2)
        expand_randz = expand_rand * theta[2]/math.sqrt(theta[0]**2 + theta[1]**2 + theta[2]**2)
        expand_allx = expand_randx * 0.55 + newNode.potential_x * 0.45
        expand_ally = expand_randy * 0.55 + newNode.potential_y * 0.45
        expand_allz = expand_randz * 0.55+ newNode.potential_y * 0.45
        angle = [expand_allx, expand_ally, expand_allz]

        # extend
        if path_sign is not True:
            newNode.x += self.expand_dis * b * angle[0]/math.sqrt(angle[0]**2 + angle[1]**2 + angle[2]**2)
            newNode.y += self.expand_dis * b * angle[1]/math.sqrt(angle[0]**2 + angle[1]**2 + angle[2]**2)
            newNode.z += self.expand_dis * b * angle[2]/math.sqrt(angle[0]**2 + angle[1]**2 + angle[2]**2)
            newNode.cost += self.expand_dis*b
        else:
            newNode.x += self.expand_dis * b/2 * angle[0]/math.sqrt(angle[0]**2 + angle[1]**2 + angle[2]**2)
            newNode.y += self.expand_dis * b/2 * angle[1]/math.sqrt(angle[0]**2 + angle[1]**2 + angle[2]**2)
            newNode.z += self.expand_dis * b/2 * angle[2]/math.sqrt(angle[0]**2 + angle[1]**2 + angle[2]**2)
            newNode.cost += self.expand_dis*b/2

        newNode.fvalue = newNode.cost + self.line_cost(newNode, self.goal)
        newNode.parent = n_ind
        return newNode, angle

    def is_near_goal(self, node):
        d = self.line_cost(node, self.goal)
        if d < self.expand_dis * 2:
            return True
        return False

    def rewire(self, newNode, nearInds):
        n_node = len(self.node_list)
        for i in nearInds:
            nearNode = self.node_list[i]
            dx = newNode.x - nearNode.x
            dy = newNode.y - nearNode.y
            dz = newNode.z - nearNode.z
            d = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
            if d == 0:
                continue
            s_cost = newNode.cost + d

            # Nearby nodes reselect parent nodes
            if nearNode.cost > s_cost:
                theta = [dx, dy, dz]
                if self.check_collision(nearNode, theta, d):
                    nearNode.parent = n_node - 1
                    nearNode.cost = s_cost
                    if nearNode.parent != 0:
                        minnodeparent = self.node_list[self.node_list[nearNode.parent].parent]
                        while nearNode.parent != 0:
                            dx = nearNode.x - minnodeparent.x
                            dy = nearNode.y - minnodeparent.y
                            dz = nearNode.z - minnodeparent.z
                            d = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                            theta = [dx, dy, dz]
                            if d == 0:
                                continue
                            if self.check_collision(minnodeparent, theta, d):
                                nearNode.parent = self.node_list[nearNode.parent].parent
                                if nearNode.parent != 0:
                                    minnodeparent = self.node_list[self.node_list[nearNode.parent].parent]
                            else:
                                break

    def Computational_potential_field(self, newNode):

        # Calculate attractive force
        dis_Goal_newnode = self.line_cost(newNode, self.goal)
        Uattx = 0.5*self.Ka*(dis_Goal_newnode)*(self.goal.x - newNode.x)
        Uatty = 0.5*self.Ka*(dis_Goal_newnode)*(self.goal.y - newNode.y)
        Uattz = 0.5*self.Ka*(dis_Goal_newnode)*(self.goal.z - newNode.z)

        # Calculate repulsive force
        Urepx = 0
        Urepy = 0
        Urepz = 0
        dobs = 6
        for obstacle in self.obstacle_list:
            center = [obstacle[0]+obstacle[3]/2, obstacle[1]+obstacle[4]/2, obstacle[2]+obstacle[5]/2]
            dis_ob_newnode = math.sqrt((center[0] - newNode.x) ** 2 + (center[1] - newNode.y) ** 2 + (center[2] - newNode.z) ** 2)
            if dis_ob_newnode < dobs:
                Urepx = Urepx + 0.5 * self.Kr*(1/dis_ob_newnode - 1/dobs)**2/dis_ob_newnode * \
                        (newNode.x - center[0])
                Urepy = Urepy + 0.5 * self.Kr * (1 / dis_ob_newnode - 1 / dobs) ** 2 / dis_ob_newnode * \
                        (newNode.y - center[1])
                Urepz = Urepy + 0.5 * self.Kr * (1 / dis_ob_newnode - 1 / dobs) ** 2 / dis_ob_newnode * \
                        (newNode.z - center[2])
        newNode.potential_x = Uattx + Urepx
        newNode.potential_y = Uatty + Urepy
        newNode.potential_z = Uattz + Urepz
        potential_angle = [newNode.potential_x, newNode.potential_y, newNode.potential_z]

        return newNode, potential_angle

    def check_segment_collision(self, x1, y1, z1, x2, y2, z2):
        sign = True

        # Z1 is 20 in map5 and 10 in map6
        if z1 > 20 or z1 < 0:
            sign = False
        else:
            for obstacle in self.obstacle_list:
                box = Box(Point(obstacle[0], obstacle[1], obstacle[2]),
                          Point(obstacle[0] + obstacle[3], obstacle[1] + obstacle[4], obstacle[2] + obstacle[5]))
                line = LineSegment(Point(x1, y1, z1), Point(x2, y2, z2))
                if not box.get_intersect_point(line):
                    sign = False
                    break
        return sign

    def check_collision(self, nearNode, theta, d):
        tmpNode = copy.deepcopy(nearNode)

        end_x = tmpNode.x + d * theta[0]/math.hypot(theta[0], theta[1], theta[2])
        end_y = tmpNode.y + d * theta[1]/math.hypot(theta[0], theta[1], theta[2])
        end_z = tmpNode.z + d * theta[2]/math.hypot(theta[0], theta[1], theta[2])

        return self.check_segment_collision(tmpNode.x, tmpNode.y, tmpNode.z, end_x, end_y, end_z)

    def get_final_course(self, lastIndex):
        path = [[self.goal.x, self.goal.y, self.goal.z]]
        while self.node_list[lastIndex].parent is not None:
            node = self.node_list[lastIndex]
            path.append([node.x, node.y, node.z])
            lastIndex = node.parent
        path.append([self.start.x, self.start.y, self.start.z])
        return path

    def draw_graph(self, rnd=None, path=None, rnd1=None):
        plt.clf()
        fig = plt.gca(projection="3d")

        for ob in self.obstacle_list:
            fig.bar3d(ob[0], ob[1], ob[2], ob[3], ob[4], ob[5], alpha=0.4)

        if rnd is not None:
            fig.scatter(rnd.x, rnd.y, rnd.z, marker='^', color='k')

        if rnd1 is not None:
            fig.scatter(rnd1[0], rnd1[1], rnd1[2], marker='^', color='r')

        for n in self.node_list:
            fig.scatter(n.x, n.y, n.z, marker='.', color='k')

        for node in self.node_list:
            if node.parent is not None:
                if node.x or node.y or node.z is not None:
                    fig.plot3D([node.x, self.node_list[node.parent].x], [node.y, self.node_list[node.parent].y],
                              [node.z, self.node_list[node.parent].z], color='g')

        fig.scatter(self.start.x, self.start.y, self.start.z, marker='x', color='r')
        fig.scatter(self.goal.x, self.goal.y, self.goal.z, marker='x', color='r')

        if path is not None:
            fig.plot3D([x for (x, y, z) in path], [y for (x, y, z) in path], [z for (x, y, z) in path], color='r')

        fig.set_xlim([0, 50])
        fig.set_ylim([0, 50])
        fig.set_zlim([0, 20])
        plt.gca().set_box_aspect((5, 5, 2))

        # fig.set_xlim([0, 50])
        # fig.set_ylim([0, 50])
        # fig.set_zlim([0, 10])
        # plt.gca().set_box_aspect((5, 5, 1))

        plt.grid(True)
        plt.pause(0.01)


class Node:

    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.cost = 0.0
        self.parent = None
        self.function = True
        self.fvalue = float('inf')
        self.potential_x = None
        self.potential_y = None
        self.potential_z = None


# The following is to determine whether the edge and the obstacles collide
class Point:
    def __init__(self, point_x, point_y, point_z):
        self.coord = [point_x, point_y, point_z]


class LineSegment:
    def __init__(self, point_start, point_end):
        origin = []
        direction = []
        for index1 in range(3):
            origin.append(point_start.coord[index1])
            direction.append(point_end.coord[index1] - point_start.coord[index1])

        self.origin = origin
        self.direction = direction

    def get_point(self, coefficient):
        point_coord = []
        for index in range(3):
            point_coord.append(self.origin[index] + coefficient * self.direction[index])
        return Point(point_coord[0], point_coord[1], point_coord[2])


class Box:
    def __init__(self, point_a, point_b):
        self.pA = point_a
        self.pB = point_b

    def get_intersect_point(self, line_segment):
        sign = False
        for index, direction in enumerate(line_segment.direction):
            if direction == 0:
                box_max = max(self.pA.coord[index], self.pB.coord[index])
                box_min = min(self.pA.coord[index], self.pB.coord[index])
                if line_segment.origin[index] > box_max or line_segment.origin[index] < box_min:
                    return True
        t0, t1 = 0., 1.
        for index in range(3):
            if line_segment.direction[index] != 0.:
                inv_dir = 1. / line_segment.direction[index]
                t_near = (self.pA.coord[index] - line_segment.origin[index]) * inv_dir
                t_far = (self.pB.coord[index] - line_segment.origin[index]) * inv_dir
                if t_near > t_far:
                    t_near, t_far = t_far, t_near
                t0 = max(t_near, t0)
                t1 = min(t_far, t1)
                if t0 >= t1:
                    return True
        intersection_point_near = line_segment.get_point(t0)
        intersection_point_far = line_segment.get_point(t1)

        return sign


def main():
    print("Start rrt planning")

    # map5
    obstacleList = [
                    np.array([ 14 , 4 , 0, 4, 4, 17]),
                    np.array([ 21 , 20 , 0, 4, 4, 17]),
                    np.array([ 32 , 3 , 0, 4, 4, 17]),
                    np.array([ 4 , 10 , 0, 5, 5, 18]),
                    np.array([ 40 , 24 , 0, 5, 5, 18]),
                    np.array([ 13 , 15 , 0, 5, 5, 18]),
                    np.array([ 7 , 27 , 0, 5, 5, 18]),
                    np.array([ 31 , 32 , 0, 3, 3, 19]),
                    np.array([ 28 , 12 , 0, 3, 3, 19]),
                    np.array([ 25 , 36 , 0, 3, 3, 19]),
                    np.array([ 29 , 25 , 0, 3, 3, 19]),
                    np.array([ 20 , 11 , 0, 4, 4, 17]),
                    np.array([ 32 , 18 , 0, 4, 4, 17]),
                    np.array([ 22 , 30 , 0, 4, 4, 17]),
                    np.array([ 39 , 38 , 0, 4, 4, 17]),]

    # map6
    # obstacleList = [np.array([3, 20, 0, 10, 2, 10]),
    #                 np.array([17, 20, 0, 10, 2, 10]),
    #                 np.array([31, 20, 0, 10, 2, 10]),
    #                 np.array([7, 28, 0, 10, 2, 10]),
    #                 np.array([21, 28, 0, 10, 2, 10]),
    #                 np.array([35, 28, 0, 10, 2, 10]),
    #                 np.array([4, 5, 0, 2, 12, 10]),
    #                 np.array([8, 32, 0, 2, 12, 10]),
    #                 np.array([15, 5, 0, 2, 12, 10]),
    #                 np.array([19, 32, 0, 2, 12, 10]),
    #                 np.array([27, 5, 0, 2, 12, 10]),
    #                 np.array([31, 32, 0, 2, 12, 10]),
    #                 np.array([37, 5, 0, 2, 12, 10]),
    #                 np.array([41, 32, 0, 2, 12, 10]),
    #                ]

    # Set params
    rrt = RRT(randArea=[0, 50], obstacleList=obstacleList, maxIter=500)

    path = rrt.rrt_star_planning(start=[1, 1, 5], goal=[45, 45, 10], animation=show_animation)  # map5
    # path = rrt.rrt_star_planning(start=[10, 3, 1], goal=[38, 47, 5], animation=show_animation)  # map6

    print("Done!!")

    if show_animation and path:
        plt.show()


if __name__ == '__main__':
    main()
