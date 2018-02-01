class NNode:
    def __init__(self , list):
        self.list = list;
        self.name = list[0];
        self.value = list[1];
        self.children = [];

    def add(self, person2):
        self.children += [person2];

    def convertToList(self):
        string = [self.list];
        for children in self.children:
            string += [children.convertToList()]
        return string;

    def find(self, name):
        root = NNode(self.list);
        root.children = self.children;
        S = set([root]);
        while(len(S) != 0):
            v = S.pop()
            if(v.name == name):
                return v;
            for children in v.children:
                S.add(children);
        return None;




    # def delete(self, person2):
    #     for children in self.children:
    #         if(children == person2):
    #             self.children.remove(person2)
    #             return True
    #         if(children.delete() == True):
    #             return True;
    #
    #     return False;

    def delete(self, person):
        root = NNode(self.list)
        root.children = self.children;
        S = set([root])
        while len(S) != 0:
            v = S.pop()
            for children in v.children:
                if children == person:
                    v.children.remove(person)
                    return True;
                S.add(children)
        return False

def convertToNode(list):
    current = NNode(list[0]);
    for i in range(1, len(list)):
        current.children += [convertToNode(list[i])];
    return current;


if __name__ == '__main__':
    A = NNode(["A", "fasdf"]);
    print(A.convertToList());

    # A.add(B);
    # B.add(C);
    #
    # nodeB = A.find("B");
    # nodeB.add(D);
    # print(A.convertToList())
    #
    #
    # A = NNode(["A", "fasdf"]);
    # B = NNode(["B", "fasfdasfd"]);
    # C = NNode(["C", "asfdasfdsaf"]);
    # D = NNode((["D", "asfdasfd"]));
    # A.add(B);
    # A.add(C);
    # nodeB = A.find("B");
    # nodeB.add(D);
    # print(A.convertToList())
    #
    #
    # A = NNode(["A", "fasdf"]);
    # B = NNode(["B", "fasfdasfd"]);
    # C = NNode(["C", "asfdasfdsaf"]);
    # D = NNode((["D", "asfdasfd"]));
    #
    # E = NNode(["E", "fasdf"]);
    # F = NNode(["F", "fasfdasfd"]);
    # G = NNode(["G", "asfdasfdsaf"]);
    # H = NNode((["H", "asfdasfd"]));
    # Q = NNode((["Q", "!"]))
    # A.add(B);
    # A.add(C);
    # A.add(D);
    # C.add(E);
    # E.add(F);
    # F.add(G);
    # G.add(H);
    # nodeF = A.find("F");
    # nodeF.add(Q);
    # print(A.convertToList());
    # help = A.delete(F);
    # print(A.convertToList());


    # yell = A.convertToList();
    # node = convertToNode(yell);
    # print(node.convertToList())
    # A = NNode(["A", "fasdf"]);
    # B = NNode(["B", "fasfdasfd"]);
    # C = NNode(["C", "asfdasfdsaf"]);
    # A.add(B);
    # A.add(C);
    # nodeB = A.find(["B", "fasfdasfd"]);
    #
    # node = convertToNode(yell);


