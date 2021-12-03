
class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def b(root):
    print(id(root))
    root.val = -4
    root = None
    print(id(root))

tree = TreeNode(5)
tree.left = TreeNode(-1)
tree.right = TreeNode(-2)

b(tree.left)
print(tree.left.val, tree.right.val)


# def a(x):
#     print(id(x))
#     x = None
#     print(id(x))

# x = 5
# a(x)



