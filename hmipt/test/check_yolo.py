import torch


a = torch.load("./proto_test.pt")

b = torch.load("./proto.pt")[-1][0]
c = torch.load("./proto_my.pt")

# print(a)
# print("------")
print(b-c)
print(b.shape)
print(c.shape)