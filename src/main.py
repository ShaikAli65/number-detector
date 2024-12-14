class A:
    def __getitem__(self,slice):
        print(slice)

A()[1,1:1,1:1:1,1]