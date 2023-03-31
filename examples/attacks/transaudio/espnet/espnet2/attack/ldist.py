DEBUG = False
ncells = 0

def recursive_ldist(A, B):
    global ncells
    ncells = 0
    memo = {}
    def get_memo(i, j):
        global ncells
        if i < 0 or j < 0:
            r = max(i, j) + 1
        elif (i, j) in memo:
            r = memo[(i, j)]
        else:
            if DEBUG: 
                print("%d %d" % (i, j), end = "")
                r = memo[(i, j)] = _ldist(i, j)
            if DEBUG: 
                print(" %d" % r)
                ncells += 1
        return r
    def min3(id1, id2, id3):
        id1, id3 = sorted([id1, id3], key = lambda c: abs(c[0] - c[1]))
        if get_memo(*id1) < get_memo(*id2):
            r = get_memo(*id1)
        else:
            r = min(get_memo(*id2), get_memo(*id3))
        return r
    def _ldist(i, j):
        if A[i] == B[j]:
            return get_memo(i-1, j-1)
        else:
            return 1 + min3((i, j-1), (i-1, j-1), (i-1, j))

    return get_memo(len(A) - 1, len(B) - 1)

def iterative_ldist(A, B):
    INIT = 0
    COPY = 1
    CMP_12 = 2
    CMP_23 = 3

    global ncells
    ncells = 0
    memo = {}
    stack = []
    in_stack = set()
    
    def get_memo(i, j):
        r = None
        if i < 0 or j < 0:
            r = max(i, j) + 1
        elif (i, j) in memo:
            r = memo[(i, j)]
        elif (i, j) not in in_stack:
            append(i, j, INIT)
        return r
    
    def set_memo(i, j, val):
        global ncells
        ncells += 1
        memo[(i, j)] = val

    def append(i, j, state):
        if INIT == state:
            in_stack.add((i, j))
        stack.append((i, j, state))

    def gen_ids(i, j):
        ids = [(i-1, j), (i, j-1)]
        ids.sort(key = lambda c: abs(c[0] - c[1]))
        ids.insert(1, (i-1, j-1))
        return ids

    targ_ids = (len(A) - 1, len(B) - 1)
    get_memo(*targ_ids)

    while len(stack) > 0:
        (i, j, state) = stack.pop()
        if INIT == state:
            if A[i] == B[j]:
                append(i, j, COPY)
                get_memo(i-1, j-1)
            else:
                ids = gen_ids(i, j)
                append(i, j, CMP_12)
                get_memo(*ids[0])
                get_memo(*ids[1])
        elif COPY == state:
            set_memo(i, j, get_memo(i-1, j-1))
        elif CMP_12 == state:
            ids = gen_ids(i, j)
            if get_memo(*ids[0]) < get_memo(*ids[1]):
                set_memo(i, j, get_memo(*ids[0]) + 1)
            else:
                append(i, j, CMP_23)
                get_memo(*ids[2])
        elif CMP_23 == state:
            ids = gen_ids(i, j)
            set_memo(i, j, 1 + min(get_memo(*ids[1]), get_memo(*ids[2])))
        else:
            raise Exception("Invalid state")

    return get_memo(*targ_ids)
