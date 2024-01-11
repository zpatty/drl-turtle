import numpy as np
def main(args=None):
    def squeeze(lst):
        print(f"input: {lst}")
        nq = 6
        q = np.array(lst).T
        print(f"q strucut: {q.shape}\n")
        qsqueezed = np.reshape(q, (nq * q.shape[1]))
        return list(qsqueezed)
    
    # def squeeze(lst):
    #     nq = 6
    #     q = np.array(lst).T
    #     qsqueezed = np.reshape(q, (nq * q.shape[1]))
    #     return list(qsqueezed)
# nq = 6
    num_samples = len([[0,1,2,3,4,5], [6,7,8,9,10,11]])
    # q = np.array([[0,1,2,3,4,5], [6,7,8,9,10,11]]).T
    listy = squeeze([[0,1,2,3,4,5], [6,7,8,9,10,11]])
    # # q = np.array([1,2])
    # print(f"q strucut: {q.shape}\n")
    # qsqueezed = np.reshape(q, (nq * q.shape[1]))
    # # qsqueezed = np.reshape(q, (q.shape[0]))
    # print(f"q squeezed: {qsqueezed.shape}\n")
    # # listy = list(qsqueezed)
    # listy = list(q)
    print(f"list : {listy}\n")
    back = np.array(listy).reshape(6,2)
    print(f"back at it: {back}\n")
    print(f"back at it shape: {back.shape}")
    print(f"back test: {back[:,0]}")
    # tback = np.array(listy).reshape(1, 2)
    # print(f"back at it: {tback}\n")
    # print(f"back at it shape: {tback.shape}")
    # print(f"back test: {tback[:,0]}")
if __name__ == '__main__':
    main()

