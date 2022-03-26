import cv2
import numpy as np


def on_change(val):
    imageCopy = img2.copy()
    for i in range(0, val):
        node = weights[i].tree
        if node is not None:
            imageCopy[node.data] = 50
            while node.parent is not None and len(node.parent.children) < 2:
                node = node.parent
                imageCopy[node.data] = 50
            if node.parent is not None:
                node.parent.children.remove(node)
                node.parent = None

    cv2.imshow("skellton", imageCopy)


class Border(object):
    def __init__(self, point, next, prev):
        self.point = point
        self.next = next
        self.prev = prev
        self.weight = None
        self.tree = None


def next_to_zero(img, x, y):
    if img[y, x - 1] == 0:
        return True
    if img[y, x + 1] == 0:
        return True
    if img[y - 1, x] == 0:
        return True
    if img[y + 1, x] == 0:
        return True
    return False


def find_next_border(img, border):
    y = border.point[0]
    x = border.point[1]
    if img[y, x - 1] == 255:
        if border.prev is None or border.prev.point != (y, x - 1):
            return y, x - 1
    elif img[y - 1, x] == 255:
        if border.prev is None or border.prev.point != (y - 1, x):
            return y - 1, x
    elif img[y + 1, x] == 255:
        if border.prev is None or border.prev.point != (y + 1, x):
            return y + 1, x
    elif img[y, x + 1] == 255:
        if border.prev is None or border.prev.point != (y, x + 1):
            return y, x + 1

    for xAdd in range(-1, 2):
        for yAdd in range(-1, 2):
            if (xAdd != 0 or yAdd != 0) and img[y + yAdd, x + xAdd] == 255:
                if border.prev is None or border.prev.point != (y + yAdd, x + xAdd):
                    return y + yAdd, x + xAdd
    return None


def get_border(img):
    # finding a point which we didnt visit before, adding all of it border and marking them with 254, the find the next point
    borders = []

    numofrows, numofcols = img.shape
    first = 0, 0
    for x in range(0, numofcols):
        for y in range(0, numofrows):
            if img[y, x] == 255:
                first = y, x
                border = Border(first, None, None)
                firstBorder = border
                while True:
                    nextPoint = find_next_border(img, border)
                    if nextPoint is None:
                        break
                    img[nextPoint] = 254
                    if nextPoint == first:
                        border.next = firstBorder
                        firstBorder.prev = border
                        break
                    border.next = Border(nextPoint, None, border)
                    border = border.next
                borders.append(border)
    return borders


def find_border(img, imgBorder):
    # marking the borders then send it to getBorder to get the structures
    numofrows, numofcols = img.shape
    for x in range(0, numofcols):
        for y in range(0, numofrows):
            if img[y, x] > 0 and next_to_zero(img, x, y):
                imgBorder[y, x] = 255
            else:
                imgBorder[y, x] = 0
    return get_border(imgBorder)


def border(image):
    image1 = image.copy()
    image2 = image.copy()
    ret, image1 = cv2.threshold(image1, 100, 255, cv2.THRESH_BINARY_INV)

    imgBorder = image2.copy()
    borders = find_border(image1, imgBorder)

    return borders

    # # getting Weight:


def get_weights(borders):
    weights = []
    for border in borders:
        if border is None:
            continue
        while border is not None and border.weight is None:
            if border.next is None or border.prev is None:
                break

            leftVector = (abs(border.point[1] - border.prev.point[1]), abs(border.point[0] - border.prev.point[0]))
            rightVector = (abs(border.point[1] - border.next.point[1]), abs(border.point[0] - border.next.point[0]))

            border.weight = np.dot(leftVector, rightVector)
            weights.append(border)
            border = border.next

    weights.sort(key=lambda border: border.weight, reverse=False)

    return weights


def border_mix(leaves, border):
    firstmatch = None
    while True:
        match = False
        for leaf in leaves:
            if border is not None and border.point == leaf.data:
                if firstmatch is None:
                    firstmatch = border
                border.tree = leaf
                border = border.next
                match = True
                break
        if not match:
            if border is None or border.prev is None or border.next is None:
                return None
            border.prev.next = border.next
            border = border.next

        if border == firstmatch:
            break
    return border


def get_leaves_in_border(borders, leavesGroups):
    outputBorders = []
    usedBorders = []
    for leaves in leavesGroups:
        newBorder = None
        for leaf in leaves:
            for border in borders:
                if border is None or abs(border.point[1] - leaf.data[1]) > 50:
                    continue
                firstBorder = None
                while border is not None and newBorder is None:
                    if border == firstBorder:
                        break
                    if firstBorder is None:
                        firstBorder = border

                    if leaf.data == border.point:
                        newBorder = border_mix(leaves, border)
                        borders.remove(firstBorder)
                        usedBorders.append(firstBorder)
                        outputBorders.append(newBorder)
                        break
                    else:
                        border = border.next

                if newBorder is not None:
                    newBorder = None

            if newBorder is not None:
                break

    return outputBorders


class Tree(object):
    def __init__(self, data, children=None, parent=None):
        self.data = data
        self.children = children or []
        self.parent = parent

    def add_child(self, data):
        new_child = Tree(data, parent=self)
        self.children.append(new_child)
        return new_child

    def is_root(self):
        return self.parent is None

    def find(self, x):
        if self.data == x: return self
        for node in self.children:
            n = node.find(x)
            if n: return n
        return None

    def is_leaf(self):
        return not self.children

    def to_array(self):
        arr = [self.data]
        for node in self.children:
            arr.extend(node.to_array())
        return arr

    def get_leaf_nodes(self):
        queue = [self]
        leafs = []
        while len(queue) > 0:
            node = queue.pop(0)
            if node.is_leaf():
                leafs.append(node)
            else:
                for n in node.children:
                    queue.append(n)
        return leafs

    def __str__(self):
        if self.is_leaf():
            return str(self.data)
        return '{data} [{children}]'.format(data=self.data, children=', '.join(map(str, self.children)))


def rev_tree(node, parent):
    temp = node.parent
    if parent is not None:
        node.children.remove(parent)
    if not node.is_root():
        node.children.append(node.parent)
        rev_tree(temp, node)
    node.parent = parent
    return node


def merge_trees(point, tree1, tree2):
    new_tree1 = rev_tree(tree1, None)
    new_tree2 = rev_tree(tree2, None)
    tree = Tree(point)
    new_tree1.parent = tree
    new_tree2.parent = tree
    tree.children.append(new_tree1)
    tree.children.append(new_tree2)
    return tree


def check_if_same_dic(arr, point):
    arr_of_vars = [[item for item in arr if item[0] == (point[0] - 1)],
                   [item for item in arr if item[0] == (point[0] + 1)],
                   [item for item in arr if item[1] == (point[1] - 1)],
                   [item for item in arr if item[1] == (point[1] + 1)]]
    if len(arr) == 2:
        for var in arr_of_vars:
            if len(var) == 2:
                return True
        return False
    if len(arr) == 3:
        for var in arr_of_vars:
            if len(var) == 3:
                return True
        return False


def search_trees(eight_connectivity, x1, y1):
    temp1 = []
    temp1.extend(zip(*np.where(eight_connectivity < 0)))
    return tuple((a + x1 - 1, b + y1 - 1) for a, b in temp1)


def share_same_dot(dot, point):
    if dot[0] != point[0] and dot[1] != point[1]:
        return False
    return True


def is_child(node, maybe_child):
    for child in node.children:
        if child == maybe_child:
            return True
    return False


def create_tree():
    trees = {}
    trees_counter = -1
    for point in sort_arr:
        if point == (92, 61):
            print(point)
        eight_connectivity = dist[point[0] - 1:point[0] + 2, point[1] - 1:point[1] + 2]
        num_of_trees = search_trees(eight_connectivity, point[0], point[1])
        if len(num_of_trees) == 0:
            new_tree = Tree(point)
            trees[trees_counter] = new_tree
            dist[point] = trees_counter
            trees_counter = trees_counter - 1
        elif len(num_of_trees) == 1:
            trees.get(dist[num_of_trees[0]]).find(num_of_trees[0]).add_child(point)
            dist[point] = dist[num_of_trees[0]]
        elif len(num_of_trees) == 2:
            if dist[num_of_trees[0]] == dist[num_of_trees[1]]:
                if check_if_same_dic(num_of_trees, point):
                    if share_same_dot(num_of_trees[1], point) and \
                            is_child(trees.get(dist[num_of_trees[0]]).find(num_of_trees[0]),
                                     trees.get(dist[num_of_trees[1]]).find(num_of_trees[1])):
                        trees.get(dist[num_of_trees[1]]).find(num_of_trees[1]).add_child(point)
                        dist[point] = dist[num_of_trees[0]]
                    elif share_same_dot(num_of_trees[0], point) and \
                            is_child(trees.get(dist[num_of_trees[1]]).find(num_of_trees[1]),
                                     trees.get(dist[num_of_trees[0]]).find(num_of_trees[0])):
                        trees.get(dist[num_of_trees[0]]).find(num_of_trees[0]).add_child(point)
                        dist[point] = dist[num_of_trees[0]]

            else:
                new_tree = merge_trees(point, trees.get(dist[num_of_trees[1]]).find(num_of_trees[1]),
                                       trees.get(dist[num_of_trees[0]]).find(num_of_trees[0]))
                del trees[dist[num_of_trees[0]]]
                del trees[dist[num_of_trees[1]]]
                trees[trees_counter] = new_tree
                arr_of_tree = new_tree.to_array()
                for dot in arr_of_tree:
                    dist[dot] = trees_counter
                trees_counter = trees_counter - 1

        elif len(num_of_trees) == 3:
            dot1 = trees.get(dist[num_of_trees[0]]).find(num_of_trees[0])
            dot2 = trees.get(dist[num_of_trees[1]]).find(num_of_trees[1])
            dot3 = trees.get(dist[num_of_trees[2]]).find(num_of_trees[2])
            if dist[num_of_trees[0]] == dist[num_of_trees[1]] and dist[num_of_trees[1]] == dist[num_of_trees[2]]:
                if check_if_same_dic(num_of_trees, point):
                    if is_child(dot1, dot2) and is_child(dot2, dot3):
                        dot3.add_child(point)
                        dist[point] = dist[dot3.data]
                    elif is_child(dot3, dot2) and is_child(dot2, dot1):
                        dot1.add_child(point)
                        dist[point] = dist[dot1.data]
                    elif is_child(dot2, dot1) and is_child(dot2, dot3):
                        dot1.add_child(point)
                        dist[point] = dist[dot1.data]
            elif dist[num_of_trees[0]] == dist[num_of_trees[1]]:
                new_tree = None
                if is_child(dot1, dot2):
                    new_tree = merge_trees(point, dot2, dot3)
                else:
                    new_tree = merge_trees(point, dot1, dot3)
                del trees[dist[num_of_trees[0]]]
                del trees[dist[num_of_trees[2]]]
                trees[trees_counter] = new_tree
                arr_of_tree = new_tree.to_array()
                for dot in arr_of_tree:
                    dist[dot] = trees_counter
                trees_counter = trees_counter - 1
            elif dist[num_of_trees[0]] == dist[num_of_trees[2]]:
                new_tree = None
                if is_child(dot1, dot3):
                    new_tree = merge_trees(point, dot1, dot2)
                else:
                    new_tree = merge_trees(point, dot3, dot2)
                del trees[dist[num_of_trees[0]]]
                del trees[dist[num_of_trees[1]]]
                trees[trees_counter] = new_tree
                arr_of_tree = new_tree.to_array()
                for dot in arr_of_tree:
                    dist[dot] = trees_counter
                trees_counter = trees_counter - 1
            elif dist[num_of_trees[1]] == dist[num_of_trees[2]]:
                new_tree = None
                if is_child(dot2, dot3):
                    new_tree = merge_trees(point, dot2, dot1)
                else:
                    new_tree = merge_trees(point, dot3, dot1)
                del trees[dist[num_of_trees[0]]]
                del trees[dist[num_of_trees[1]]]
                trees[trees_counter] = new_tree
                arr_of_tree = new_tree.to_array()
                for dot in arr_of_tree:
                    dist[dot] = trees_counter
                trees_counter = trees_counter - 1

    return trees


if __name__ == '__main__':

    weights = []
    imgBorder = []
    image1 = []

    # Read the image as a grayscale image
    img = cv2.imread('text4.png', 0)
    img2 = img.copy()
    img3 = img.copy()

    # Threshold the image
    ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
    ret, img2 = cv2.threshold(img2, 100, 255, cv2.THRESH_BINARY)

    # Perform the distance transform algorithm
    dist = cv2.distanceTransform(img, cv2.DIST_L2, 3)

    # Creating a sorted dictionary from the highest distance value to the lowest
    sort_arr = {index: x for index, x in np.ndenumerate(dist) if x}
    sort_arr = dict(sorted(sort_arr.items(), key=lambda item: item[1], reverse=True))

    # create tree for each letter
    trees = create_tree()

    # find all the leaves in the tree
    node3 = None
    leavesGroups = []
    for x in trees:
        for y in trees.get(x).to_array():
            if y == (28, 26):
                node3 = trees.get(x).find(y)
                print(y)
            img2[y] = 200
        leafs = trees.get(x).get_leaf_nodes()
        leavesGroups.append(leafs)

    # find the border
    borders = border(img3)
    # filter the border to contain leaves nodes only
    leavesBorder = get_leaves_in_border(borders, leavesGroups)
    # adding weights for each leaf
    weights = get_weights(leavesBorder)

    # show result with bar:
    cv2.imshow('skellton', img2)
    cv2.createTrackbar('slider', "skellton", 0, len(weights), on_change)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
