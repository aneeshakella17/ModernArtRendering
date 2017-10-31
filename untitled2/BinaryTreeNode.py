class Node:
    def __init__(self, hash, length, left = None, right = None):
        self.left = left;
        self.right = right;
        self.hash = hash;
        self.mac = None;
        self.length = length;

    def setLeft(self, newNode):
        self.left = newNode;

    def setRight(self, value):
        self.right = value;

    def setMac(self, value):
        self.mac = value;

    def getLength(self):
        return self.length;

    def getMac(self):
        return self.mac

    def getHash(self):
        return self.hash;



    def getLeft(self):
        return self.left;

    def getRight(self):
        return self.right;

    def constructTree(self, str, chunkSize):
        list = [];
        count = 0;
        start = 0
        while(len(str) > count):
            start = count;
            end = count + chunkSize;
            if(end < len(str)):
                list += [Node(str[start: end], None, None)];
            elif end - len(str) < chunkSize:
                value = str[start: end];
                for i in range(0, end - len(str)):
                         value += " ";
                list += [Node(value, None, None)];
            count = end;
        n = 0;

        while(2**n < len(str)):
            n += 1


        while(len(list)%2**(n - 2) != 0):
            list += [Node(" ", None, None)];

        return self.treeHelper(list, str);

    def treeHelper(self, list, str):
        inventNode = None;
        if(len(list) == 1):
            return list[0];
        while(len(list) > 1):
            tempNodes = [];
            while(len(tempNodes) != len(list)/2):
                current = (2*len(tempNodes));
                next = (2*len(tempNodes) + 1);
                left = list[current];
                right = list[next];
                inventNode = Node(left.hash + right.hash, self.length, left, right)
                tempNodes += [inventNode];
            list = tempNodes;
        return inventNode;

    def update(self, other):
         if(self.left == None and self.right == None):
             self.hash = other.hash;

         else:
            if self.left != None and other.left!= None :
                if(self.left.hash != other.left.hash):
                    self.left.update(other.left);
            if(self.right != None and other.right != None):
                if(self.right.hash != other.right.hash):
                    self.right.update(other.right);
            self.hash = self.left.hash + self.right.hash

    def convertToList(self):
        list = [self.hash]
        if(self.getLeft() != None):
            list += [self.left.convertToList()];
        if(self.getRight() != None):
            list += [self.right.convertToList()];
        return list

def listToTree(list):
    root = Node(list[0]);
    if(len(list) > 1):
        if(len(list[1]) != 0):
            root.left = listToTree(list[1]);
    if(len(list) > 2):
        if(len(list[2]) != 0):
            root.right = listToTree(list[2]);
    return root;







if __name__ == '__main__':
    node = Node("hello", 2, left = None, right = None);
    print(node.getHash())
    node2 = node.constructTree("goodbye",  4)
    node3 = node.constructTree("foodbye", 4);
    print(node2.convertToList())
    print(node3.convertToList())
    node2.update(node3);
    print(node2.convertToList())







